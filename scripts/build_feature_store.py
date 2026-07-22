"""Version2 Phase1: point-in-time feature storeの構築スクリプト(証明実行)。

Phase1の完了条件(最低ライン)として、既定では5銘柄・直近1年分で実行する。
特徴量生成とラベル生成を明確に別工程(別ループ)として実行し、リークを防ぐ。

実行:
    python scripts/build_feature_store.py
    python scripts/build_feature_store.py --symbols AAPL,MSFT,NVDA,TSLA,AMD --years 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

import config
from api.yfinance_client import fetch_historical_prices
from feature_store.database import FeatureStoreDB
from feature_store.features import compute_feature_frame, build_feature_row
from feature_store.labels import compute_labels_for_signal
from feature_store.market_regime import MarketContext
from feature_store.quality_checks import format_report, run_quality_checks
from feature_store.sector import SectorContext
from feature_store.universe import APPROX_SOURCE_LABEL, UNIVERSE_VERSION, evaluate_candidate

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"

MIN_PRICE = config.SCREEN_MIN_PRICE
MIN_MARKET_CAP = config.SCREEN_MIN_MARKET_CAP
MIN_AVG_VOLUME = config.SCREEN_MIN_AVG_VOLUME

# ラベル確定に必要な将来の営業日数(最長15日先まで見るため、期間末尾はこの分だけ余裕を持たせる)
LABEL_HORIZON_BUFFER_DAYS = 25


def get_company_meta(symbol: str) -> dict:
    """yfinanceからセクター/業種/時価総額/取引所を取得する(現在の分類を過去に遡って適用する
    簡易近似。真のpoint-in-time企業属性ではない点に注意)。"""
    try:
        info = yf.Ticker(symbol).info
    except Exception:
        info = {}
    return {
        "exchange": info.get("exchange"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="feature store構築(Phase1証明実行)")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--years", type=float, default=1.0)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    end_date = date.today() - timedelta(days=LABEL_HORIZON_BUFFER_DAYS)  # ラベル確定用の余裕を持たせる
    start_date = end_date - timedelta(days=int(365 * args.years))
    # ma200(200営業日)等のための遡り取得。米国market は年間約252営業日 ≒ 365日中69%なので、
    # 420暦日 ≒ 290営業日分の遡りとなり、要件の「開始日より最低250〜300営業日前」を満たす。
    fetch_start = (start_date - timedelta(days=420)).isoformat()
    fetch_end = date.today().isoformat()  # ラベル計算には最新日までの価格データが必要

    print(f"対象銘柄: {symbols}")
    print(f"特徴量生成期間: {start_date.isoformat()} 〜 {end_date.isoformat()}")

    print("価格データ取得中...")
    histories: dict[str, list[dict]] = {}
    metas: dict[str, dict] = {}
    failed_symbols: list[str] = []
    for sym in symbols:
        h = fetch_historical_prices(sym, fetch_start, fetch_end)
        if not h:
            failed_symbols.append(sym)
            continue
        histories[sym] = h
        metas[sym] = get_company_meta(sym)

    print("特徴量フレーム計算中...")
    feature_frames = {}
    for sym in list(histories.keys()):
        df = compute_feature_frame(histories[sym], min_rows=200)
        if df is None:
            failed_symbols.append(sym)
            continue
        feature_frames[sym] = df

    active_symbols = list(feature_frames.keys())
    print(f"特徴量計算できた銘柄: {active_symbols} (失敗: {failed_symbols})")

    print("市場環境・セクターコンテキスト構築中...")
    market_ctx = MarketContext(start_date.isoformat(), end_date.isoformat())
    sectors_in_use = [metas[s].get("sector") for s in active_symbols if metas[s].get("sector")]
    sector_ctx = SectorContext(start_date.isoformat(), end_date.isoformat(), sectors_in_use)

    # ---------- 工程1: 特徴量生成(未来情報は一切使わない) ----------
    # 重要: ここではウィンドウ内の「全営業日」についてレコードを生成する(候補日だけに絞らない)。
    # 候補判定(evaluate_candidate)の結果はcandidate_flag/candidate_reasonとして各行に付与するだけで、
    # 非候補日のレコードも保存する。これにより後から「全営業日」「候補日のみ」「Version1条件を
    # 満たした日のみ」のいずれの集計もこのfeaturesテーブル1つから正確に抽出できる。
    print("[工程1/2] 特徴量生成(point-in-time、全営業日を保存)...")
    all_features: list[dict] = []
    signal_indices: dict[str, list[int]] = {}  # symbol -> [df中のインデックス](全営業日分)

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
            is_candidate, reason = evaluate_candidate(row, MIN_PRICE, MIN_MARKET_CAP, MIN_AVG_VOLUME, meta.get("market_cap"))
            target_date_str = row["date"].strftime("%Y-%m-%d")
            mf = market_ctx.get_features(target_date_str)
            sf = sector_ctx.get_sector_return(meta.get("sector"), target_date_str)
            feature = build_feature_row(
                df, i, sym, meta.get("exchange"), meta.get("sector"), meta.get("industry"),
                meta.get("market_cap"), APPROX_SOURCE_LABEL, mf, sf,
                universe_included_flag=1,  # このビルドで対象とした固定ウォッチリストに含まれる銘柄
                candidate_flag=1 if is_candidate else 0,
                candidate_reason=reason,
                universe_version=UNIVERSE_VERSION,
            )
            all_features.append(feature)
            signal_indices[sym].append(i)

    # セクター相対強度の順位付け(その日の"候補"銘柄内でのstock_vs_sector_20dの順位、0〜1に正規化)
    # 非候補日のレコードはsector_rank=None(比較対象の候補ユニバースに含まれないため)のまま。
    by_date: dict[str, list[dict]] = {}
    for f in all_features:
        by_date.setdefault(f["signal_date"], []).append(f)
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

    # ---------- 工程2: ラベル生成(工程1とは完全に別ループ。未来データのみ参照) ----------
    print("[工程2/2] ラベル生成(future labels、工程1とは独立)...")
    all_labels: list[dict] = []
    for sym in active_symbols:
        df = feature_frames[sym]
        for i in signal_indices[sym]:
            label = compute_labels_for_signal(df, i, sym)
            if label is not None:
                all_labels.append(label)

    print(f"生成したラベルレコード数: {len(all_labels)}")

    # ---------- 保存 ----------
    print("DBへ保存中...")
    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    db.upsert_features_bulk(all_features)
    db.upsert_labels_bulk(all_labels)

    # ---------- データ品質チェック ----------
    print("データ品質チェック実行中...")
    report = run_quality_checks(all_features, all_labels, db=db, failed_symbols=failed_symbols)
    report_text = format_report(report)
    print(report_text)

    report_path = config.REPORTS_DIR / f"feature_store_quality_{date.today().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"# Feature Store データ品質レポート ({date.today().isoformat()})\n\n```\n{report_text}\n```\n",
        encoding="utf-8",
    )

    db.close()
    print(f"\n特徴量DB: {FEATURE_STORE_DB_PATH}")
    print(f"品質レポート: {report_path}")
    print(f"\n合計: features={len(all_features)}件, labels={len(all_labels)}件")


if __name__ == "__main__":
    main()
