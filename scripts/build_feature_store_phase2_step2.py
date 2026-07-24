"""Version2 Phase2 Step2: 100〜300銘柄(目標200)・5年へのユニバース拡張。

目的はモデル学習ではなく、Phase3で使えるだけのpoint-in-time特徴量ストアを完成させること。
Train/Validation/Testの日付境界はPhase2 Step1から一切変更しない(直前のbuild_runsから
そのままコピーする)。

再開可能性: build_run_idは対象範囲(universe_version・期間)から決定的に生成するため、
同じ条件で再実行すると同じbuild_run_idになる。銘柄ごとのfeatures/labels計算が完了済み
(build_run_symbol_status.stage='compute_features_labels'がcompleted)であればスキップする。
価格取得はfeature_store.providers.CachingMarketDataProvider経由でディスクキャッシュされる
ため、再実行時にネットワーク再取得が最小限になる。

実行:
    python scripts/build_feature_store_phase2_step2.py
    python scripts/build_feature_store_phase2_step2.py --symbols AAPL,MSFT  # 動作確認用
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from backtest.engine import simulate_symbol
from backtest.portfolio import simulate_portfolio
from feature_store.database import FeatureStoreDB
from feature_store.feature_metadata import seed_feature_metadata
from feature_store.features import (
    FEATURE_VERSION,
    build_feature_row,
    compute_cross_sectional_percentiles,
    compute_feature_frame,
)
from feature_store.labels import LABEL_VERSION, compute_labels_for_signal
from feature_store.market_regime import MarketContext
from feature_store.providers import (
    FixedWatchlistUniverseProvider,
    YFinanceFundamentalDataProvider,
    default_market_data_provider,
)
from feature_store.quality_checks import format_report, run_quality_checks
from feature_store.sector import SectorContext
from feature_store.universe_source import build_candidate_universe

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"
MIN_PRICE = config.SCREEN_MIN_PRICE
MIN_MARKET_CAP = config.SCREEN_MIN_MARKET_CAP
MIN_AVG_VOLUME = config.SCREEN_MIN_AVG_VOLUME
MIN_ROWS = 200

# 既知のティッカー変更(手動確認済み)。価格系列の自動接続は行わない。
KNOWN_TICKER_ALIASES = [
    {
        "old_ticker": "SQ", "new_ticker": "XYZ", "effective_date": "2025-01-01",
        "company_name": "Block, Inc.", "reason": "ticker_rename",
        "source": "manual_research_2026-07-22", "manual_override_flag": 1,
        "price_series_spliced_flag": 0,
        "notes": "Block Inc(旧Square)がティッカーをSQからXYZへ変更。新ティッカーの価格履歴のみ"
                 "取得し、旧ティッカー時代の価格とは接続しない(継続性未検証のため)。",
    },
]


def seed_ticker_aliases(db: FeatureStoreDB) -> None:
    for row in KNOWN_TICKER_ALIASES:
        db.upsert_ticker_alias(row)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="feature store構築(Phase2 Step2)")
    parser.add_argument("--symbols", type=str, default=None, help="未指定ならuniverse_source.pyの候補一覧を使う")
    args = parser.parse_args()

    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    seed_ticker_aliases(db)
    seed_feature_metadata(db)

    # Train/Validation/Testの日付境界はStep1から一切変更しない(直近のPhase2 build_runsから複製)
    prior_run = db.conn.execute(
        "SELECT * FROM build_runs WHERE build_run_id LIKE 'phase2_step1%' ORDER BY run_at DESC LIMIT 1"
    ).fetchone()
    if prior_run is None:
        raise RuntimeError("Phase2 Step1のbuild_runsが見つかりません。先にStep1を実行してください。")
    prior_run = dict(prior_run)
    start_date, end_date = prior_run["start_date"], prior_run["end_date"]
    train_start, train_end = prior_run["train_start"], prior_run["train_end"]
    val_start, val_end = prior_run["val_start"], prior_run["val_end"]
    test_start, test_end = prior_run["test_start"], prior_run["test_end"]
    fetch_start = (date.fromisoformat(start_date) - timedelta(days=420)).isoformat()
    fetch_end = date.today().isoformat()

    # 銘柄選定パイプライン(universe source → normalize → ETF除外 → 重複除去)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        excluded_from_selection: list[dict] = []
    else:
        symbols, excluded_from_selection = build_candidate_universe()

    universe_provider = FixedWatchlistUniverseProvider(symbols, MIN_PRICE, MIN_MARKET_CAP, MIN_AVG_VOLUME)
    universe_version = universe_provider.universe_version
    # build_run_idは対象範囲から決定的に生成する(再実行時に同じIDへ再開できるようにするため)
    build_run_id = f"phase2_step2_{universe_version}_{start_date}_{end_date}"

    print(f"build_run_id: {build_run_id}")
    print(f"対象銘柄候補: {len(symbols)}件(選定除外: {len(excluded_from_selection)}件)")
    print(f"期間: {start_date}〜{end_date}(Step1と同一の境界を維持)")
    print(f"Train: {train_start}〜{train_end} / Validation: {val_start}〜{val_end} / Test(封印): {test_start}〜{test_end}")

    market_provider = default_market_data_provider(db=db)
    fundamental_provider = YFinanceFundamentalDataProvider()

    membership_rows: list[dict] = []
    daily_universe_rows: list[dict] = []
    failed_symbols: list[str] = []
    aliased_symbols: list[str] = []
    active_symbols: list[str] = []
    metas: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}

    print("\n[段階1/4] 価格データ取得(キャッシュ・リトライ・ティッカー変更対応)...")
    for sym in symbols:
        db.set_symbol_stage_status(build_run_id, sym, "fetch_price", "running", started_at=datetime.now(timezone.utc).isoformat())
        try:
            h = market_provider.fetch_historical_prices(sym, fetch_start, fetch_end)
            used_symbol = sym
            if not h:
                alias = db.get_ticker_alias_for(sym)
                if alias:
                    h = market_provider.fetch_historical_prices(alias["new_ticker"], fetch_start, fetch_end)
                    used_symbol = alias["new_ticker"]
                    if h:
                        aliased_symbols.append(f"{sym}->{alias['new_ticker']}")
            if not h:
                failed_symbols.append(sym)
                db.set_symbol_stage_status(
                    build_run_id, sym, "fetch_price", "failed",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error_type="empty_history", error_message="fetch_historical_pricesが空を返した",
                )
                membership_rows.append({
                    "ticker": sym, "build_run_id": build_run_id,
                    "first_available_date": None, "last_available_date": None,
                    "sector": None, "exchange": None, "data_fetch_status": "failed",
                })
                continue
            meta = fundamental_provider.get_company_meta(used_symbol)
            histories[sym] = h
            metas[sym] = meta
            db.set_symbol_stage_status(
                build_run_id, sym, "fetch_price", "completed",
                completed_at=datetime.now(timezone.utc).isoformat(), rows_written=len(h),
            )
            membership_rows.append({
                "ticker": sym, "build_run_id": build_run_id,
                "first_available_date": h[0]["date"], "last_available_date": h[-1]["date"],
                "sector": meta.get("sector"), "exchange": meta.get("exchange"),
                "data_fetch_status": "ok" if used_symbol == sym else "ok_via_alias",
            })
        except Exception as e:  # noqa: BLE001
            failed_symbols.append(sym)
            db.set_symbol_stage_status(
                build_run_id, sym, "fetch_price", "failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error_type=type(e).__name__, error_message=str(e),
            )
            membership_rows.append({
                "ticker": sym, "build_run_id": build_run_id,
                "first_available_date": None, "last_available_date": None,
                "sector": None, "exchange": None, "data_fetch_status": "failed",
            })

    db.upsert_universe_membership_bulk(membership_rows)
    print(f"取得成功: {len(histories)}件 / 失敗: {len(failed_symbols)}件 / エイリアス経由: {len(aliased_symbols)}件 {aliased_symbols}")

    print("\n[段階2/4] daily_universe(全営業日レイヤー)構築中...")
    for sym, h in histories.items():
        df = pd.DataFrame(h)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["avg_volume_20d"] = df["volume"].rolling(20, min_periods=20).mean()
        meta = metas[sym]
        for _, row in df.iterrows():
            d = row["date"].strftime("%Y-%m-%d")
            if not (start_date <= d <= end_date):
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
                "universe_version": universe_version,
            })
    db.upsert_daily_universe_bulk(daily_universe_rows)
    print(f"daily_universeレコード数: {len(daily_universe_rows)}")

    print("\n[段階3/4] 特徴量・ラベル計算(銘柄単位、再開可能)...")
    print("市場環境・セクターコンテキスト構築中...")
    market_ctx = MarketContext(start_date, end_date, market_data_provider=market_provider)
    all_sectors = [m.get("sector") for m in metas.values() if m.get("sector")]
    sector_ctx = SectorContext(start_date, end_date, all_sectors, market_data_provider=market_provider)
    all_features_for_quality: list[dict] = []
    all_labels_for_quality: list[dict] = []
    n_skipped_resumed = 0

    for sym in list(histories.keys()):
        status = db.get_symbol_stage_status(build_run_id, sym, "compute_features_labels")
        if status is not None and status["status"] == "completed":
            n_skipped_resumed += 1
            active_symbols.append(sym)
            continue

        db.set_symbol_stage_status(build_run_id, sym, "compute_features_labels", "running", started_at=datetime.now(timezone.utc).isoformat())
        try:
            df = compute_feature_frame(histories[sym], min_rows=MIN_ROWS)
            if df is None:
                db.set_symbol_stage_status(
                    build_run_id, sym, "compute_features_labels", "skipped",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error_type="insufficient_history", error_message=f"総データ行数がmin_rows={MIN_ROWS}未満",
                )
                continue

            meta = metas[sym]
            indices_in_window = [
                i for i in range(len(df))
                if start_date <= df.iloc[i]["date"].strftime("%Y-%m-%d") <= end_date
            ]
            sym_features = []
            sym_labels = []
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
                    universe_version=universe_version,
                )
                sym_features.append(feature)
                label = compute_labels_for_signal(df, i, sym, as_of_date=fetch_end)
                sym_labels.append(label)

            db.upsert_features_bulk(sym_features)
            db.upsert_labels_bulk(sym_labels)
            all_features_for_quality.extend(sym_features)
            all_labels_for_quality.extend(sym_labels)
            active_symbols.append(sym)

            db.set_symbol_stage_status(
                build_run_id, sym, "compute_features_labels", "completed",
                completed_at=datetime.now(timezone.utc).isoformat(), rows_written=len(sym_features),
            )
        except Exception as e:  # noqa: BLE001
            db.set_symbol_stage_status(
                build_run_id, sym, "compute_features_labels", "failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error_type=type(e).__name__, error_message=str(e),
            )
            logging.exception("特徴量/ラベル計算に失敗: %s", sym)

    print(f"特徴量計算できた銘柄: {len(active_symbols)}件(うち再開によりスキップ: {n_skipped_resumed}件)")

    # ---------- 横断percentile特徴量(全銘柄処理完了後に日付単位でまとめて計算) ----------
    print("\n[段階4/4] 横断percentile特徴量・セクターランク・Version1採用ルールの計算中...")
    all_features = db.get_all_features()  # 今回のビルド分だけでなく既存分も含めて全件(日付単位でグループ化するため)
    by_date: dict[str, list[dict]] = defaultdict(list)
    for f in all_features:
        if start_date <= f["signal_date"] <= end_date:
            by_date[f["signal_date"]].append(f)

    for day_features in by_date.values():
        compute_cross_sectional_percentiles(day_features)
        candidates_that_day = [f for f in day_features if f.get("candidate_flag") == 1]
        ranked = sorted(
            [f for f in candidates_that_day if f.get("stock_vs_sector_20d") is not None],
            key=lambda f: f["stock_vs_sector_20d"],
        )
        n = len(ranked)
        for rank, f in enumerate(ranked):
            f["sector_rank"] = round((rank + 1) / n, 3) if n > 0 else None

    updated_rows = [f for day_features in by_date.values() for f in day_features]
    db.upsert_features_bulk(updated_rows)
    print(f"percentile付与済みレコード数: {len(updated_rows)}")

    # ---------- Version1採用ルールの再現(candidate_snapshots用) ----------
    trades_all: list[dict] = []
    price_frames: dict[str, pd.DataFrame] = {}
    for sym in active_symbols:
        if sym not in histories:
            continue
        trades = simulate_symbol(
            sym, histories[sym],
            min_technical_score=config.BACKTEST_MIN_TECHNICAL_SCORE,
            max_holding_days=config.BACKTEST_MAX_HOLDING_DAYS,
            min_rows=MIN_ROWS, entry_mode="next_open",
            slippage_pct=config.BACKTEST_SLIPPAGE_PCT, commission_pct=config.BACKTEST_COMMISSION_PCT,
        )
        trades = [t for t in trades if start_date <= t["signal_date"] <= end_date]
        trades_all.extend(trades)
        df = compute_feature_frame(histories[sym], min_rows=MIN_ROWS)
        if df is not None:
            price_frames[sym] = df[["date", "close"]].copy()

    portfolio_result = simulate_portfolio(
        trades_all, price_frames, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS,
        cash_reuse_timing="next_day",
    )
    admitted_keys = {(t["symbol"], t["signal_date"]) for t in portfolio_result["admitted_trades"]}
    trade_by_key = {(t["symbol"], t["signal_date"]): t for t in trades_all}
    skip_reason_by_key = {(s["symbol"], s["signal_date"]): s["reason"] for s in portfolio_result["skipped_trades"]}
    trades_by_entry_date: dict[str, list[dict]] = defaultdict(list)
    for t in trades_all:
        trades_by_entry_date[t["entry_date"]].append(t)
    rank_within_entry_day: dict[tuple, int] = {}
    for day_trades in trades_by_entry_date.values():
        ranked = sorted(day_trades, key=lambda t: -(t.get("signal_score") or 0))
        for rank, t in enumerate(ranked, start=1):
            rank_within_entry_day[(t["symbol"], t["signal_date"])] = rank

    candidate_snapshot_rows: list[dict] = []
    for signal_date, day_features in by_date.items():
        candidates_that_day = [f for f in day_features if f.get("candidate_flag") == 1]
        ranked = sorted(candidates_that_day, key=lambda f: -(f.get("technical_score_v1") or 0))
        for rank, f in enumerate(ranked, start=1):
            key = (f["ticker"], f["signal_date"])
            qualified = key in trade_by_key
            selected = key in admitted_keys
            reason = skip_reason_by_key.get(key) if (qualified and not selected) else None
            candidate_snapshot_rows.append({
                "ticker": f["ticker"], "signal_date": signal_date, "build_run_id": build_run_id,
                "candidate_source": f.get("candidate_source"), "candidate_rank": rank,
                "candidate_reason": f.get("candidate_reason"), "technical_score_v1": f.get("technical_score_v1"),
                "signaled_flag": 1 if qualified else 0, "selected_by_v1_flag": 1 if selected else 0,
                "selected_rank_v1": rank_within_entry_day.get(key) if qualified else None,
                "selection_capacity": config.BACKTEST_MAX_CONCURRENT_POSITIONS,
                "rejected_due_to_capacity_flag": 1 if (qualified and not selected) else 0,
                "rejected_reason": reason,
                "feature_version": f.get("feature_version"), "universe_version": f.get("universe_version"),
            })
    db.upsert_candidate_snapshots_bulk(candidate_snapshot_rows)
    print(f"candidate_snapshotsレコード数(今回分): {len(candidate_snapshot_rows)}")
    print(
        f"シグナル(score>={config.BACKTEST_MIN_TECHNICAL_SCORE}、全期間): {len(trades_all)} / "
        f"採用: {portfolio_result['trades_taken']} / 見送り(枠不足+現金不足): {len(portfolio_result['skipped_trades'])}"
    )

    # ---------- build_run記録(Train/Val/Testの境界はStep1から不変) ----------
    db.upsert_build_run({
        "build_run_id": build_run_id, "run_at": datetime.now(timezone.utc).isoformat(),
        "symbols": ",".join(symbols), "start_date": start_date, "end_date": end_date,
        "universe_version": universe_version, "feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION,
        "train_start": train_start, "train_end": train_end, "val_start": val_start, "val_end": val_end,
        "test_start": test_start, "test_end": test_end,
        "notes": "Phase2 Step2: 100-300銘柄への拡張(Step1の日付境界を維持)",
    })

    # ---------- データ品質チェック ----------
    print("\nデータ品質チェック実行中...")
    report = run_quality_checks(all_features_for_quality, all_labels_for_quality, db=db, failed_symbols=failed_symbols)
    report_text = format_report(report)
    print(report_text)

    report_path = config.REPORTS_DIR / f"feature_store_phase2_step2_quality_{date.today().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    excluded_text = "\n".join(f"- {e['symbol']}({e['category']}): {e['reason']}" for e in excluded_from_selection)
    report_path.write_text(
        f"# Feature Store Phase2 Step2 データ品質レポート ({date.today().isoformat()})\n\n"
        f"build_run_id: {build_run_id}\n\n"
        f"## 選定除外銘柄\n\n{excluded_text or '(なし)'}\n\n"
        f"## 取得失敗銘柄\n\n{failed_symbols}\n\n"
        f"## エイリアス経由取得\n\n{aliased_symbols}\n\n"
        f"```\n{report_text}\n```\n",
        encoding="utf-8",
    )

    db.close()
    print(f"\n特徴量DB: {FEATURE_STORE_DB_PATH}")
    print(f"品質レポート: {report_path}")
    print(f"\nbuild_run_id: {build_run_id}")


if __name__ == "__main__":
    main()
