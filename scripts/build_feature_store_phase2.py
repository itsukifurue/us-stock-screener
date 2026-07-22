"""Version2 Phase2 Step1: 銘柄数・期間を拡張したfeature store構築。

Phase1(5銘柄・1年の証明実行)から、以下を拡張する:
  - 対象銘柄: backtest.watchlist.PHASE2_WATCHLIST(既定46銘柄、Version1の29銘柄を含む・
    上場時期/セクターを多様化)
  - 対象期間: 既定5年
  - 2層ユニバース設計: daily_universe(全営業日・軽量)/ candidate_snapshots(候補日のみ・
    Version1採用ルールとの突き合わせ結果つき)を追加保存
  - Version1の採用ルール(backtest.engine.simulate_symbol + backtest.portfolio.simulate_portfolio、
    max_concurrent=3)を実際に同じ設定で再実行し、「採用された候補」「枠不足で見送られた候補」を
    signal_date単位で突き合わせる
  - yfinance直接依存を feature_store.providers 経由に統一(Provider抽象化)
  - Train(60%)/Validation(20%)/Test(20%)の日付境界をbuild_runsに記録する
    (Test期間は本スクリプトでは対象成績を計算・表示しない。品質指標のみ集計可能)

実行:
    python scripts/build_feature_store_phase2.py
    python scripts/build_feature_store_phase2.py --years 5
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from backtest.engine import simulate_symbol
from backtest.portfolio import simulate_portfolio
from backtest.watchlist import PHASE2_WATCHLIST
from feature_store.database import FeatureStoreDB
from feature_store.features import FEATURE_VERSION, build_feature_row, compute_feature_frame
from feature_store.labels import LABEL_VERSION, compute_labels_for_signal
from feature_store.market_regime import MarketContext
from feature_store.providers import (
    FixedWatchlistUniverseProvider,
    YFinanceFundamentalDataProvider,
    YFinanceMarketDataProvider,
)
from feature_store.quality_checks import format_report, run_quality_checks
from feature_store.sector import SectorContext

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"
MIN_PRICE = config.SCREEN_MIN_PRICE
MIN_MARKET_CAP = config.SCREEN_MIN_MARKET_CAP
MIN_AVG_VOLUME = config.SCREEN_MIN_AVG_VOLUME
LABEL_HORIZON_BUFFER_DAYS = 25
MIN_ROWS = 200  # ma200等のため、features/labels/tradesの全てでこの行数以上の履歴を要求する(統一)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="feature store構築(Phase2 Step1)")
    parser.add_argument("--symbols", type=str, default=",".join(PHASE2_WATCHLIST))
    parser.add_argument("--years", type=float, default=5.0)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    build_run_id = f"phase2_step1_{date.today().isoformat()}_{uuid.uuid4().hex[:8]}"

    end_date = date.today() - timedelta(days=LABEL_HORIZON_BUFFER_DAYS)
    start_date = end_date - timedelta(days=int(365 * args.years))
    fetch_start = (start_date - timedelta(days=420)).isoformat()  # ma200等のための遡り取得(約287営業日)
    fetch_end = date.today().isoformat()

    total_days = (end_date - start_date).days
    train_end = start_date + timedelta(days=int(total_days * 0.6))
    val_end = start_date + timedelta(days=int(total_days * 0.8))
    print(f"build_run_id: {build_run_id}")
    print(f"対象銘柄({len(symbols)}件): {symbols}")
    print(f"特徴量生成期間: {start_date.isoformat()} 〜 {end_date.isoformat()}")
    print(f"Train: {start_date.isoformat()} 〜 {train_end.isoformat()}")
    print(f"Validation: {train_end.isoformat()} 〜 {val_end.isoformat()}")
    print(f"Test(封印、品質指標のみ): {val_end.isoformat()} 〜 {end_date.isoformat()}")

    market_provider = YFinanceMarketDataProvider()
    fundamental_provider = YFinanceFundamentalDataProvider()
    universe_provider = FixedWatchlistUniverseProvider(symbols, MIN_PRICE, MIN_MARKET_CAP, MIN_AVG_VOLUME)

    # ---------- 価格・属性データ取得 + universe_membership ----------
    print("価格データ取得中(providers経由)...")
    histories: dict[str, list[dict]] = {}
    metas: dict[str, dict] = {}
    membership_rows: list[dict] = []
    failed_symbols: list[str] = []

    for sym in symbols:
        h = market_provider.fetch_historical_prices(sym, fetch_start, fetch_end)
        meta = fundamental_provider.get_company_meta(sym)
        if not h:
            failed_symbols.append(sym)
            membership_rows.append({
                "ticker": sym, "build_run_id": build_run_id,
                "first_available_date": None, "last_available_date": None,
                "sector": meta.get("sector"), "exchange": meta.get("exchange"),
                "data_fetch_status": "failed",
            })
            continue
        histories[sym] = h
        metas[sym] = meta
        membership_rows.append({
            "ticker": sym, "build_run_id": build_run_id,
            "first_available_date": h[0]["date"], "last_available_date": h[-1]["date"],
            "sector": meta.get("sector"), "exchange": meta.get("exchange"),
            "data_fetch_status": "ok",
        })

    print(f"取得できた銘柄: {len(histories)}件 (失敗: {failed_symbols})")

    # 生の価格履歴をキャッシュ保存する(逆選択分析スクリプト等で同じ46銘柄・同じ期間を
    # 再利用する際に、yfinanceへの再フェッチ(数分かかる)を避けるため。gitignore対象)。
    import pickle
    cache_path = config.DATA_DIR / "phase2_price_histories.pkl"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"histories": histories, "metas": metas, "fetch_start": fetch_start, "fetch_end": fetch_end}, f)
    print(f"価格履歴キャッシュ保存: {cache_path}")

    # ---------- daily_universe(軽量。全営業日、featureが計算できない銘柄日も含む) ----------
    print("daily_universe(全営業日レイヤー)構築中...")
    daily_universe_rows: list[dict] = []
    for sym, h in histories.items():
        df = pd.DataFrame(h)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["avg_volume_20d"] = df["volume"].rolling(20, min_periods=20).mean()
        meta = metas[sym]
        for _, row in df.iterrows():
            d = row["date"].strftime("%Y-%m-%d")
            if not (start_date.isoformat() <= d <= end_date.isoformat()):
                continue
            probe = pd.Series({"close": row["close"], "avg_volume20": row["avg_volume_20d"]})
            is_candidate, reason = universe_provider.evaluate_candidate(probe, meta.get("market_cap"))
            daily_universe_rows.append({
                "ticker": sym, "date": d, "build_run_id": build_run_id,
                "close": float(row["close"]), "volume": float(row["volume"]),
                "dollar_volume": float(row["close"] * row["volume"]),
                "universe_included_flag": 1,
                "candidate_flag": 1 if is_candidate else 0,
                "candidate_reason": reason,
                "universe_version": universe_provider.universe_version,
            })
    print(f"daily_universeレコード数: {len(daily_universe_rows)}")

    # ---------- 特徴量フレーム(min_rows=200を満たす銘柄のみ) ----------
    print("特徴量フレーム計算中...")
    feature_frames: dict[str, pd.DataFrame] = {}
    for sym in list(histories.keys()):
        df = compute_feature_frame(histories[sym], min_rows=MIN_ROWS)
        if df is None:
            failed_symbols.append(sym)
            continue
        feature_frames[sym] = df

    active_symbols = list(feature_frames.keys())
    print(f"特徴量計算できた銘柄: {len(active_symbols)}件 (失敗/計算不能を含む全失敗銘柄: {failed_symbols})")

    print("市場環境・セクターコンテキスト構築中...")
    market_ctx = MarketContext(start_date.isoformat(), end_date.isoformat())
    sectors_in_use = [metas[s].get("sector") for s in active_symbols if metas[s].get("sector")]
    sector_ctx = SectorContext(start_date.isoformat(), end_date.isoformat(), sectors_in_use)

    # ---------- 工程1: 特徴量生成(全営業日) ----------
    print("[工程1/2] 特徴量生成(point-in-time、全営業日)...")
    all_features: list[dict] = []
    signal_indices: dict[str, list[int]] = {}

    for sym in active_symbols:
        df = feature_frames[sym]
        meta = metas[sym]
        indices_in_window = [
            i for i in range(len(df))
            if start_date.isoformat() <= df.iloc[i]["date"].strftime("%Y-%m-%d") <= end_date.isoformat()
        ]
        signal_indices[sym] = []
        for i in indices_in_window:
            row = df.iloc[i]
            is_candidate, reason = universe_provider.evaluate_candidate(row, meta.get("market_cap"))
            target_date_str = row["date"].strftime("%Y-%m-%d")
            mf = market_ctx.get_features(target_date_str)
            sf = sector_ctx.get_sector_return(meta.get("sector"), target_date_str)
            feature = build_feature_row(
                df, i, sym, meta.get("exchange"), meta.get("sector"), meta.get("industry"),
                meta.get("market_cap"), universe_provider.candidate_source_label, mf, sf,
                universe_included_flag=1,
                candidate_flag=1 if is_candidate else 0,
                candidate_reason=reason,
                universe_version=universe_provider.universe_version,
            )
            all_features.append(feature)
            signal_indices[sym].append(i)

    by_date: dict[str, list[dict]] = defaultdict(list)
    for f in all_features:
        by_date[f["signal_date"]].append(f)
    for day_features in by_date.values():
        candidates_that_day = [f for f in day_features if f.get("candidate_flag") == 1]
        ranked = sorted(
            [f for f in candidates_that_day if f.get("stock_vs_sector_20d") is not None],
            key=lambda f: f["stock_vs_sector_20d"],
        )
        n = len(ranked)
        for rank, f in enumerate(ranked):
            f["sector_rank"] = round((rank + 1) / n, 3) if n > 0 else None

    num_candidates = sum(1 for f in all_features if f.get("candidate_flag") == 1)
    print(f"生成した特徴量レコード数(全営業日): {len(all_features)} (うち候補日: {num_candidates})")

    # ---------- 工程2: ラベル生成 ----------
    print("[工程2/2] ラベル生成...")
    all_labels: list[dict] = []
    for sym in active_symbols:
        df = feature_frames[sym]
        for i in signal_indices[sym]:
            label = compute_labels_for_signal(df, i, sym)
            if label is not None:
                all_labels.append(label)
    print(f"生成したラベルレコード数: {len(all_labels)}")

    # ---------- Version1採用ルールの再現(同一設定でsimulate_symbol + simulate_portfolio) ----------
    print("Version1採用ルール(スコア閾値+ポートフォリオ枠)を再現中...")
    trades_all: list[dict] = []
    price_frames: dict[str, pd.DataFrame] = {}
    for sym in active_symbols:
        trades = simulate_symbol(
            sym, histories[sym],
            min_technical_score=config.BACKTEST_MIN_TECHNICAL_SCORE,
            max_holding_days=config.BACKTEST_MAX_HOLDING_DAYS,
            min_rows=MIN_ROWS,
            entry_mode="next_open",
            slippage_pct=config.BACKTEST_SLIPPAGE_PCT,
            commission_pct=config.BACKTEST_COMMISSION_PCT,
        )
        trades = [t for t in trades if start_date.isoformat() <= t["signal_date"] <= end_date.isoformat()]
        trades_all.extend(trades)
        price_frames[sym] = feature_frames[sym][["date", "close"]].copy()

    portfolio_result = simulate_portfolio(
        trades_all, price_frames,
        max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS,
        cash_reuse_timing="next_day",
    )
    admitted_keys = {(t["symbol"], t["signal_date"]) for t in portfolio_result["admitted_trades"]}
    trade_by_key = {(t["symbol"], t["signal_date"]): t for t in trades_all}
    # 見送り理由の内訳(no_slot/cash_insufficient)。同一(symbol,signal_date)が複数回見送られる
    # ことはない(1シグナル1トレードのため)前提でdictにする。
    skip_reason_by_key = {
        (s["symbol"], s["signal_date"]): s["reason"] for s in portfolio_result["skipped_trades"]
    }

    trades_by_entry_date: dict[str, list[dict]] = defaultdict(list)
    for t in trades_all:
        trades_by_entry_date[t["entry_date"]].append(t)
    rank_within_entry_day: dict[tuple, int] = {}
    for day_trades in trades_by_entry_date.values():
        ranked = sorted(day_trades, key=lambda t: -(t.get("signal_score") or 0))
        for rank, t in enumerate(ranked, start=1):
            rank_within_entry_day[(t["symbol"], t["signal_date"])] = rank

    print(
        f"シグナル(score>={config.BACKTEST_MIN_TECHNICAL_SCORE})件数: {len(trades_all)} / "
        f"採用: {portfolio_result['trades_taken']} / "
        f"見送り(枠不足): {portfolio_result['trades_skipped_no_slot']} / "
        f"見送り(現金不足): {portfolio_result['trades_skipped_cash']}"
    )

    # ---------- candidate_snapshots ----------
    print("candidate_snapshots構築中...")
    candidates_by_date: dict[str, list[dict]] = defaultdict(list)
    for f in all_features:
        if f.get("candidate_flag") == 1:
            candidates_by_date[f["signal_date"]].append(f)

    candidate_snapshot_rows: list[dict] = []
    for signal_date, day_features in candidates_by_date.items():
        ranked = sorted(day_features, key=lambda f: -(f.get("technical_score_v1") or 0))
        for rank, f in enumerate(ranked, start=1):
            key = (f["ticker"], f["signal_date"])
            # qualified: technical_score_v1>=45 で信号化され、かつ同一銘柄内で保有期間が重複して
            # いない(=trades_allに実際にトレードとして現れた)候補。信号化されなかった
            # (スコア未達 or 銘柄内クールダウン中の)universe candidateはqualified=Falseとなり、
            # selected_rank_v1/rejected_reasonともにNone(=競合に参加すらしていない)のまま。
            qualified = key in trade_by_key
            selected = key in admitted_keys
            reason = skip_reason_by_key.get(key) if (qualified and not selected) else None
            candidate_snapshot_rows.append({
                "ticker": f["ticker"], "signal_date": signal_date, "build_run_id": build_run_id,
                "candidate_source": f.get("candidate_source"),
                "candidate_rank": rank,
                "candidate_reason": f.get("candidate_reason"),
                "technical_score_v1": f.get("technical_score_v1"),
                "signaled_flag": 1 if qualified else 0,
                "selected_by_v1_flag": 1 if selected else 0,
                "selected_rank_v1": rank_within_entry_day.get(key) if qualified else None,
                "selection_capacity": config.BACKTEST_MAX_CONCURRENT_POSITIONS,
                "rejected_due_to_capacity_flag": 1 if (qualified and not selected) else 0,
                "rejected_reason": reason,
                "feature_version": f.get("feature_version"),
                "universe_version": f.get("universe_version"),
            })
    print(f"candidate_snapshotsレコード数: {len(candidate_snapshot_rows)}")

    # ---------- DB保存 ----------
    print("DBへ保存中...")
    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    db.upsert_features_bulk(all_features)
    db.upsert_labels_bulk(all_labels)
    db.upsert_universe_membership_bulk(membership_rows)
    db.upsert_daily_universe_bulk(daily_universe_rows)
    db.upsert_candidate_snapshots_bulk(candidate_snapshot_rows)
    db.upsert_build_run({
        "build_run_id": build_run_id,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "symbols": ",".join(symbols),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "universe_version": universe_provider.universe_version,
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "train_start": start_date.isoformat(), "train_end": train_end.isoformat(),
        "val_start": train_end.isoformat(), "val_end": val_end.isoformat(),
        "test_start": val_end.isoformat(), "test_end": end_date.isoformat(),
        "notes": "Phase2 Step1: 30-50銘柄・5年拡張、2層ユニバース設計、Version1採用ルール再現",
    })

    # ---------- データ品質チェック ----------
    print("データ品質チェック実行中...")
    report = run_quality_checks(all_features, all_labels, db=db, failed_symbols=failed_symbols)
    report_text = format_report(report)
    print(report_text)

    report_path = config.REPORTS_DIR / f"feature_store_phase2_quality_{date.today().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Feature Store Phase2 Step1 データ品質レポート ({date.today().isoformat()})\n\n"
        f"build_run_id: {build_run_id}\n\n```\n{report_text}\n```\n",
        encoding="utf-8",
    )

    db.close()
    print(f"\n特徴量DB: {FEATURE_STORE_DB_PATH}")
    print(f"品質レポート: {report_path}")
    print(
        f"\n合計: features={len(all_features)}件, labels={len(all_labels)}件, "
        f"daily_universe={len(daily_universe_rows)}件, candidate_snapshots={len(candidate_snapshot_rows)}件"
    )
    print(f"build_run_id: {build_run_id}")


if __name__ == "__main__":
    main()
