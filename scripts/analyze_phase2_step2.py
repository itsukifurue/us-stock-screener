"""Version2 Phase2 Step2: Train/Validationの特徴量品質・期間安定性の確認分析。

目的はVersion1スコアの改善ではなく、Phase3で使う特徴量のデータ品質と期間安定性を
100〜300銘柄規模で確認すること。Test期間はSQLのWHERE句の段階で除外し(表示時に隠すのではない)、
探索処理そのものに一切含めない。

実行:
    python scripts/analyze_phase2_step2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config
from feature_store.database import FeatureStoreDB

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"

QUANTILE_FEATURES = [
    "technical_score_v1", "rsi_14", "atr_pct", "return_5d", "volume_ratio_5d",
    "return_1d_pctrank", "return_5d_pctrank", "volume_ratio_5d_pctrank", "atr_pct_pctrank",
    "rsi_14_pctrank", "dollar_volume_pctrank", "market_cap_pctrank",
]
N_BINS = 5


def load_trainval_only(db: FeatureStoreDB, test_start: str) -> pd.DataFrame:
    rows = db.conn.execute(
        """
        SELECT f.*, l.future_return_10d, l.target_15pct_within_10d, l.target_trade_success,
               l.target_trade_pnl_pct, l.label_status
        FROM features f
        LEFT JOIN labels l ON f.ticker = l.ticker AND f.signal_date = l.signal_date
        WHERE f.signal_date < :test_start AND f.eligible_flag = 1
        ORDER BY f.signal_date, f.ticker
        """,
        {"test_start": test_start},
    ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def pf_ev(pnl: pd.Series) -> dict:
    s = pnl.dropna()
    if len(s) == 0:
        return {"n": 0, "pf": None, "ev_pct": None, "win_rate": None}
    wins, losses = s[s > 0].sum(), -s[s < 0].sum()
    pf = round(wins / losses, 3) if losses > 0 else None
    return {"n": len(s), "pf": pf, "ev_pct": round(float(s.mean()), 3), "win_rate": round(float((s > 0).mean() * 100), 1)}


def quantile_table(df: pd.DataFrame, col: str) -> list[str]:
    vals = df[col].dropna()
    if len(vals) < 50:
        return [f"### {col}\n(サンプル不足のためスキップ: {len(vals)}件)\n"]
    try:
        binned, edges = pd.qcut(vals, N_BINS, retbins=True, duplicates="drop")
    except ValueError:
        return [f"### {col}\n(値のばらつきが小さくbin分割不能)\n"]
    lines = [f"### {col}", "", "| 分位 | n | target_15pct陽性率 | 将来10日リターン平均 | PF | EV(%) | 勝率(%) |", "|---|---|---|---|---|---|---|"]
    sub = df.loc[vals.index].copy()
    sub["_bin"] = binned
    for b, group in sub.groupby("_bin", observed=True):
        stats = pf_ev(group["target_trade_pnl_pct"])
        pos = group["target_15pct_within_10d"].dropna()
        pos_rate = f"{round(pos.mean()*100,1)}%" if len(pos) else "N/A"
        ret10 = group["future_return_10d"].dropna()
        ret10_mean = round(float(ret10.mean()), 2) if len(ret10) else "N/A"
        lines.append(f"| {b} | {len(group)} | {pos_rate} | {ret10_mean} | {stats['pf']} | {stats['ev_pct']} | {stats['win_rate']} |")
    lines.append("")
    return lines


def category_table(df: pd.DataFrame, col: str, title: str) -> list[str]:
    lines = [f"### {title}({col}別)", "", "| 区分 | n | target_15pct陽性率 | PF | EV(%) | 勝率(%) |", "|---|---|---|---|---|---|"]
    for cat, group in df.groupby(col, dropna=False):
        stats = pf_ev(group["target_trade_pnl_pct"])
        pos = group["target_15pct_within_10d"].dropna()
        pos_rate = f"{round(pos.mean()*100,1)}%" if len(pos) else "N/A"
        lines.append(f"| {cat} | {len(group)} | {pos_rate} | {stats['pf']} | {stats['ev_pct']} | {stats['win_rate']} |")
    lines.append("")
    return lines


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)
    build_run = dict(db.conn.execute(
        "SELECT * FROM build_runs WHERE build_run_id LIKE 'phase2_step2%' ORDER BY run_at DESC LIMIT 1"
    ).fetchone())
    test_start = build_run["test_start"]

    df = load_trainval_only(db, test_start)
    assert df["signal_date"].max() < test_start, "Test期間のレコードが混入しています(監査失敗)"
    print(f"[Test封印チェック] 読み込み件数: {len(df)}件、最大signal_date: {df['signal_date'].max()} (< {test_start})")

    train_mask = (df["signal_date"] >= build_run["train_start"]) & (df["signal_date"] < build_run["train_end"])
    val_mask = (df["signal_date"] >= build_run["val_start"]) & (df["signal_date"] < build_run["val_end"])
    train_df, val_df = df[train_mask], df[val_mask]

    lines: list[str] = [f"# Phase2 Step2 Train/Validation分析 ({pd.Timestamp.today().date().isoformat()})", ""]
    lines.append(f"build_run_id: {build_run['build_run_id']}")
    lines.append(f"対象: eligible_universeのみ(一次スクリーニング通過)。Train={len(train_df)}件 / Validation={len(val_df)}件")
    lines.append("")

    lines.append("## 特徴量別分位分析(Train)")
    lines.append("")
    for col in QUANTILE_FEATURES:
        if col in train_df.columns:
            lines.extend(quantile_table(train_df, col))

    lines.append("## 市場環境別(market_regime)")
    lines.extend(category_table(train_df, "market_regime", "市場環境別"))

    lines.append("## セクター別")
    lines.extend(category_table(train_df, "sector", "セクター別"))

    lines.append("## 時価総額帯別(market_cap_pctrank)")
    if "market_cap_pctrank" in train_df.columns:
        train_df = train_df.copy()
        train_df["_mcap_band"] = pd.cut(train_df["market_cap_pctrank"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        lines.extend(category_table(train_df, "_mcap_band", "時価総額帯別"))

    lines.append("## ATR%帯別")
    if "atr_pct" in train_df.columns:
        train_df["_atr_band"] = pd.qcut(train_df["atr_pct"], 5, duplicates="drop")
        lines.extend(category_table(train_df, "_atr_band", "ATR%帯別"))

    lines.append("## 同日候補数別(cross_section_universe_size)")
    if "cross_section_universe_size" in train_df.columns:
        train_df["_n_bucket"] = pd.cut(
            train_df["cross_section_universe_size"], [0, 50, 100, 150, 200, 1000],
            labels=["<=50", "51-100", "101-150", "151-200", "200+"],
        )
        lines.extend(category_table(train_df, "_n_bucket", "同日候補数(全銘柄規模)別"))

    lines.append("## candidate全体 vs Version1シグナル(signal_v1_flag)の比較")
    lines.extend(category_table(train_df, "signal_v1_flag", "signal_v1_flag別"))

    lines.append("## Validation側の同項目(Trainとの比較用)")
    lines.append("")
    lines.append("### 市場環境別(Validation)")
    lines.extend(category_table(val_df, "market_regime", "市場環境別(Validation)"))
    lines.append("### signal_v1_flag別(Validation)")
    lines.extend(category_table(val_df, "signal_v1_flag", "signal_v1_flag別(Validation)"))

    report_path = config.REPORTS_DIR / f"phase2_step2_trainval_report_{pd.Timestamp.today().date().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    db.close()
    print(f"レポート出力: {report_path}")


if __name__ == "__main__":
    main()
