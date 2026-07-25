"""Phase3A: データ読み込み・特徴量ホワイトリスト・Test封印。

【使用禁止列(必ずXから除外)】
- 識別子: ticker, signal_date, build_run_id, feature_version, universe_version
  (company_nameはfeaturesテーブルに存在しないため対象外)
- 未来情報・ラベル関連: labelsテーブルの全列(対象targetを除く)、future_return系、
  future_max/min_return系、days_to_target、exit_reason、label_status、target_trade_pnl_pct
- 採用結果・ポートフォリオ結果: candidate_snapshots由来の列(selected_by_v1_flag等)は
  そもそもfeaturesテーブルに存在しないため、featuresだけをSELECTする本モジュールの
  設計上、構造的に混入しない。
- 取得後に確定する運用情報: data_collected_at、build_run_symbol_status由来の列
  (completed_at/rows_written/error_type等)も同様に構造的に混入しない。

【追加の除外(設計判断、レポートに明記する)】
- universe_included_flag/candidate_flag/eligible_flag: eligible_flag=1で絞り込むため
  定数列になり予測力を持たない。
- signal_v1_flag/signal_reason: technical_score_v1から決定的に導出される
  (score>=45の別表現)ため、technical_score_v1と冗長。
- candidate_source/candidate_reason/sector_etf: 予測的意味を持たないブックキーピング文字列
  (sector_etfはsectorから1:1で決まる冗長列)。
- open/high/low/close/price: 銘柄間で価格水準が大きく異なり横断比較に適さないため除外
  (株価収益率・pctrank系の正規化済み特徴量を使う設計判断)。

technical_score_v1は比較用の通常特徴量として残す。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from feature_store.database import FeatureStoreDB

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"

IDENTIFIER_COLUMNS = ["ticker", "signal_date"]

EXCLUDE_ALWAYS = {
    "feature_version", "universe_version", "data_collected_at",
    "candidate_source", "candidate_reason", "sector_etf",
    "universe_included_flag", "candidate_flag", "eligible_flag",
    "signal_v1_flag", "signal_reason",
    "price", "open", "high", "low", "close",
}

CATEGORICAL_COLUMNS = ["sector", "industry", "exchange", "market_regime"]

TARGET_COLUMNS = ["target_trade_success", "target_15pct_within_10d"]
ECONOMIC_COLUMN = "target_trade_pnl_pct"  # 分類学習には使わず、経済成績評価専用


def get_build_run(db: FeatureStoreDB) -> dict:
    row = db.conn.execute(
        "SELECT * FROM build_runs WHERE build_run_id LIKE 'phase2_step2%' ORDER BY run_at DESC LIMIT 1"
    ).fetchone()
    return dict(row)


def get_feature_whitelist(db: FeatureStoreDB) -> tuple[list[str], list[str]]:
    """(数値特徴量列, カテゴリ特徴量列)を返す。featuresテーブルの全列からEXCLUDE_ALWAYSと
    識別子・カテゴリ列を除いた残りを数値特徴量とする(スキーマドリフトに強くするため
    ハードコードの列挙ではなく、実際のテーブル定義から導出する)。
    """
    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(features)").fetchall()]
    excluded = set(IDENTIFIER_COLUMNS) | EXCLUDE_ALWAYS | set(CATEGORICAL_COLUMNS)
    numeric_cols = [c for c in cols if c not in excluded]
    return numeric_cols, list(CATEGORICAL_COLUMNS)


def load_trainval_dataset(
    db: FeatureStoreDB,
    universe_variant: str = "A_eligible_universe",
    cooldown_days: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Train+Validation期間のデータをSQLのWHERE句の段階でTest除外して読み込む
    (表示時に隠すのではなく、探索・学習処理そのものにTestを一切取り込まない)。

    universe_variant: "A_eligible_universe"(eligible_flag=1) または
                       "B_signal_v1_subset"(signal_v1_flag=1もかつ満たす部分集合)。
    cooldown_days: 0なら全eligible営業日、5/10等を指定すると同一tickerについて
                   その日数以内の連続日を間引く(重複サンプル影響の確認用)。
    """
    build_run = get_build_run(db)
    test_start = build_run["test_start"]
    numeric_cols, categorical_cols = get_feature_whitelist(db)
    all_cols = IDENTIFIER_COLUMNS + numeric_cols + categorical_cols + ["technical_score_v1" if "technical_score_v1" not in numeric_cols else None]
    all_cols = [c for c in dict.fromkeys(all_cols) if c]  # 重複除去(technical_score_v1は既にnumeric_colsに含まれる想定)

    col_list_sql = ", ".join(f"f.{c}" for c in all_cols)
    query = f"""
        SELECT {col_list_sql},
               l.target_trade_success, l.target_15pct_within_10d, l.target_trade_pnl_pct,
               l.label_status
        FROM features f
        LEFT JOIN labels l ON f.ticker = l.ticker AND f.signal_date = l.signal_date
        WHERE f.signal_date < :test_start
          AND f.eligible_flag = 1
          AND l.label_status = 'confirmed'
    """
    rows = db.conn.execute(query, {"test_start": test_start}).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    assert df["signal_date"].max() < test_start, "Test期間のレコードが混入しています(監査失敗)"

    if universe_variant == "B_signal_v1_subset":
        # signal_v1_flagはSELECTしていないため、technical_score_v1>=45で同等の条件を再現する
        df = df[df["technical_score_v1"] >= 45].reset_index(drop=True)

    if cooldown_days > 0:
        df = _apply_cooldown(df, cooldown_days)

    meta = {
        "build_run": build_run, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
        "test_start": test_start, "universe_variant": universe_variant, "cooldown_days": cooldown_days,
    }
    return df, meta


def _apply_cooldown(df: pd.DataFrame, cooldown_days: int) -> pd.DataFrame:
    """同一tickerについて、直前の採用済みsignal_dateからcooldown_days営業日以内の行を間引く。
    「営業日」は当該tickerの実際のsignal_date系列上の間隔(暦日ではなく行の間隔)で数える。
    """
    keep_rows = []
    df_sorted = df.sort_values(["ticker", "signal_date"])
    last_kept_date: dict[str, int] = {}
    for ticker, group in df_sorted.groupby("ticker"):
        dates = group["signal_date"].tolist()
        idxs = group.index.tolist()
        last_kept_pos = -10**9
        for pos, (idx, d) in enumerate(zip(idxs, dates)):
            if pos - last_kept_pos > cooldown_days:
                keep_rows.append(idx)
                last_kept_pos = pos
    return df.loc[sorted(keep_rows)].reset_index(drop=True)


def train_val_masks(df: pd.DataFrame, build_run: dict) -> tuple[pd.Series, pd.Series]:
    train_mask = (df["signal_date"] >= build_run["train_start"]) & (df["signal_date"] < build_run["train_end"])
    val_mask = (df["signal_date"] >= build_run["val_start"]) & (df["signal_date"] < build_run["val_end"])
    return train_mask, val_mask
