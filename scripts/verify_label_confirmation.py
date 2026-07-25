"""Phase3A着手前の最終確認: 全confirmedラベルについて、必要な将来営業日数が
実際の価格データ内に存在することを機械的に検証する。

target_15pct_within_10d(主要ラベル1)は「翌営業日エントリーから10営業日」の将来ウィンドウを
要求する。target_trade_success(主要ラベル2)は最大保有10営業日のATRトレードのため、
同じく最低10営業日分の将来データを要求する(日中にstop/target到達すればそれ未満でも
確定できるが、未到達の場合は10営業日分のデータが必要)。

この検証は、feature_store.labels.compute_labels_for_signal が実際に使ったのと同じ
価格データ(data/price_cache/*.pkl、ビルド時のキャッシュ)を再度読み込み、
各confirmedラベルについて「signal_dateの翌営業日(entry)から数えて10営業日分のデータが
実際に存在するか」を独立に再計算して突き合わせる。

実行:
    python scripts/verify_label_confirmation.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from feature_store.database import FeatureStoreDB

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"
PRICE_CACHE_DIR = config.DATA_DIR / "price_cache"
REQUIRED_FUTURE_DAYS = 10  # target_15pct_within_10d/target_trade_successとも10営業日ウィンドウ


def build_date_index(history: list[dict]) -> list[str]:
    return sorted(row["date"] for row in history)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    build_run = dict(db.conn.execute(
        "SELECT * FROM build_runs WHERE build_run_id LIKE 'phase2_step2%' ORDER BY run_at DESC LIMIT 1"
    ).fetchone())
    start_date, end_date = build_run["start_date"], build_run["end_date"]
    print(f"対象期間: {start_date} 〜 {end_date}")
    print(f"build_run_id: {build_run['build_run_id']}")

    # キャッシュされた価格履歴を読み込む(CachingMarketDataProvider経由でビルド時に
    # 実際に使われたのと同じディスクキャッシュ、data/price_cache/*.pkl)
    histories = {}
    for f in PRICE_CACHE_DIR.glob("*.pkl"):
        with open(f, "rb") as fh:
            cache = pickle.load(fh)
        histories[f.stem] = cache["history"]
    print(f"価格キャッシュから{len(histories)}銘柄分の履歴を読み込みました")

    date_index_by_ticker = {t: build_date_index(h) for t, h in histories.items()}

    # ティッカー変更(エイリアス)銘柄は、features/labelsにはold_tickerで保存されているが、
    # 実際に使われた価格データはnew_ticker名でキャッシュされている(build_feature_store_phase2_step2.py
    # のフォールバック取得ロジック参照)。検証時もそのマッピングを辿る必要がある。
    aliases = db.get_all_ticker_aliases()
    alias_map = {a["old_ticker"]: a["new_ticker"] for a in aliases}
    print(f"ティッカーエイリアスマッピング: {alias_map}")

    rows = db.conn.execute(
        "SELECT ticker, signal_date, label_status FROM labels WHERE signal_date BETWEEN ? AND ? AND label_status = 'confirmed'",
        (start_date, end_date),
    ).fetchall()
    print(f"検証対象(label_status='confirmed'): {len(rows)}件")

    mismatches = []
    checked = 0
    for r in rows:
        ticker, signal_date = r["ticker"], r["signal_date"]
        lookup_ticker = alias_map.get(ticker, ticker)
        dates = date_index_by_ticker.get(lookup_ticker)
        if dates is None:
            mismatches.append((ticker, signal_date, "価格キャッシュに銘柄が見つからない"))
            continue
        try:
            sig_idx = dates.index(signal_date)
        except ValueError:
            mismatches.append((ticker, signal_date, "signal_dateが価格データ内に見つからない"))
            continue
        entry_idx = sig_idx + 1
        required_last_idx = entry_idx + REQUIRED_FUTURE_DAYS - 1
        checked += 1
        if required_last_idx > len(dates) - 1:
            mismatches.append((
                ticker, signal_date,
                f"confirmedだが将来データ不足(必要idx={required_last_idx}, 実際の最大idx={len(dates)-1})",
            ))

    print(f"実際に検証できた件数: {checked}件")
    print(f"不整合件数: {len(mismatches)}件")
    for m in mismatches[:20]:
        print(" -", m)

    # 参考情報: 最終signal_date・最終価格取得日・確保している営業日数バッファ
    max_signal_date = max(dates[-1] for dates in date_index_by_ticker.values() if dates)
    print(f"\n価格データの最終取得日(全銘柄中の最大値): {max_signal_date}")
    print(f"特徴量保存対象の最終signal_date(end_date): {end_date}")
    print(
        f"end_date以降、価格データ最終日までの営業日バッファ: "
        f"end_date={end_date} から実際の価格データ最終日={max_signal_date} までの間隔"
    )

    db.close()

    if mismatches:
        print("\n[FAIL] confirmedなのに将来データが不足しているラベルが見つかりました。")
        sys.exit(1)
    else:
        print("\n[PASS] 全confirmedラベルについて、必要な将来営業日数が価格データ内に存在することを確認しました。")


if __name__ == "__main__":
    main()
