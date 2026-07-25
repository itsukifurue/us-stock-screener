"""Phase3A.5: 市場環境依存・ATR依存・日次TopN実運用評価の診断分析。

Phase3Aで確認された予測力(target_trade_success中心、target_15pct_within_10dは補助)が、
市場全体の上げ下げやATR(値動きの大きさ)だけによる見かけの効果ではなく、
同じ市場環境・同じボラティリティ条件の中でも銘柄選別能力として残るかを検証する。

Feature Set v1は変更しない。Phase3Aで選ばれた最良モデル(logreg_l2, C=0.01, 同一前処理・
同一特徴量・同一Train/Validation境界)を固定して使う(再探索なし)。

Test期間はSQLの段階で除外し、探索・学習・閾値決定・ATR分位点決定のいずれにも一切使わない。

実行:
    python -u scripts/run_phase3a5_diagnostics.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import psutil
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import config
from feature_store.database import FeatureStoreDB
from modeling import evaluate
from modeling.data import ECONOMIC_COLUMN, get_build_run, load_trainval_dataset, train_val_masks
from modeling.preprocessing import build_preprocessor

FEATURE_STORE_DB_PATH = config.DATA_DIR / "feature_store.db"
FEATURE_SET_VERSION = "v1"
MODEL_VERSION = "phase3a_logreg_l2_C0.01"
RANDOM_SEED = 42
N_RANDOM_TRIALS = 1000

MARKET_COLUMNS = [
    "spy_close", "spy_return_1d", "spy_return_5d", "spy_above_ma20", "spy_above_ma50",
    "spy_above_ma200", "spy_ma20_slope", "spy_ma200_slope", "qqq_return_5d", "iwm_return_5d",
    "vix_close", "vix_return_5d",
]
MARKET_CATEGORICAL = ["market_regime"]

PROCESS = psutil.Process()


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    mem_mb = PROCESS.memory_info().rss / 1024 / 1024
    print(f"[{ts}] [mem={mem_mb:.0f}MB] {msg}", flush=True)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent).decode().strip()
    except Exception:
        return "unknown"


# ============================================================
# 汎用ヘルパー
# ============================================================

def fit_logreg_l2_c001(train_df, y_train, numeric_cols, categorical_cols=None, include_categorical=False):
    cols = numeric_cols if not categorical_cols else numeric_cols
    pre = build_preprocessor(numeric_cols, categorical_cols or [], include_categorical)
    clf = LogisticRegression(solver="lbfgs", max_iter=300, C=0.01, class_weight=None, l1_ratio=0.0)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train_df, y_train)
    return pipe


def pf_ev_stats(pnl: pd.Series) -> dict:
    s = pnl.dropna()
    if len(s) == 0:
        return {"n": 0, "pf": None, "ev_pct": None, "win_rate_pct": None, "median_pct": None, "max_losing_streak": None}
    wins, losses = s[s > 0].sum(), -s[s < 0].sum()
    pf = round(float(wins / losses), 3) if losses > 0 else None
    return {
        "n": int(len(s)), "pf": pf, "ev_pct": round(float(s.mean()), 3), "median_pct": round(float(s.median()), 3),
        "win_rate_pct": round(float((s > 0).mean() * 100), 1),
        "max_losing_streak": max_losing_streak(s),
    }


def max_losing_streak(pnl_ordered: pd.Series) -> int:
    streak = 0
    max_streak = 0
    for v in pnl_ordered:
        if v <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def month_key(date_str: str) -> str:
    return date_str[:7]


def half_key(date_str: str) -> str:
    y, m = date_str[:4], int(date_str[5:7])
    return f"{y}-H{1 if m <= 6 else 2}"


def quarter_key(date_str: str) -> str:
    y, m = date_str[:4], int(date_str[5:7])
    return f"{y}-Q{(m - 1) // 3 + 1}"


def period_breakdown(df: pd.DataFrame, prob_col: str, target_col: str, key_fn) -> list[dict]:
    d = df.copy()
    d["_period"] = d["signal_date"].apply(key_fn)
    rows = []
    for period, g in d.groupby("_period"):
        stats = pf_ev_stats(g[ECONOMIC_COLUMN])
        pos = g[target_col].dropna()
        try:
            pr_auc = evaluate.classification_metrics(g[target_col].values, g[prob_col].values)["pr_auc"]
        except Exception:
            pr_auc = None
        rows.append({
            "period": period, "n": len(g),
            "positive_rate": round(float(pos.mean()), 4) if len(pos) else None,
            "pr_auc": pr_auc, "mean_predicted_prob": round(float(g[prob_col].mean()), 4),
            "pf": stats["pf"], "ev_pct": stats["ev_pct"],
        })
    return rows


def daily_select(df: pd.DataFrame, score_col: str, target_col: str, n: int | None = None, pct: float | None = None) -> pd.DataFrame:
    picks = []
    for _, g in df.groupby("signal_date"):
        if n is not None:
            k = min(n, len(g))
        else:
            k = max(1, int(round(len(g) * pct)))
        picks.append(g.nlargest(k, score_col))
    return pd.concat(picks) if picks else df.iloc[0:0]


def summarize_selection(picked: pd.DataFrame, target_col: str) -> dict:
    stats = pf_ev_stats(picked.sort_values("signal_date")[ECONOMIC_COLUMN])
    pos = picked[target_col].dropna()
    return {
        "n_trades": len(picked), "n_days": picked["signal_date"].nunique(),
        "positive_rate": round(float(pos.mean()), 4) if len(pos) else None,
        **stats,
        "monthly_pf": {k: v["pf"] for k, v in _group_pf(picked, month_key).items()},
        "halfyear_pf": {k: v["pf"] for k, v in _group_pf(picked, half_key).items()},
    }


def _group_pf(df: pd.DataFrame, key_fn) -> dict:
    d = df.copy()
    d["_period"] = d["signal_date"].apply(key_fn)
    out = {}
    for period, g in d.groupby("_period"):
        out[period] = pf_ev_stats(g[ECONOMIC_COLUMN])
    return out


def random_daily_topn_distribution(df: pd.DataFrame, target_col: str, n: int, n_trials: int, seed0: int) -> dict:
    day_groups = list(df.groupby("signal_date"))
    rng = np.random.default_rng(seed0)
    pfs, evs, pos_rates, streaks = [], [], [], []
    for _ in range(n_trials):
        picks = []
        for _, g in day_groups:
            k = min(n, len(g))
            idx = rng.choice(len(g), size=k, replace=False)
            picks.append(g.iloc[idx])
        picked = pd.concat(picks)
        stats = pf_ev_stats(picked[ECONOMIC_COLUMN])
        pfs.append(stats["pf"] if stats["pf"] is not None else np.nan)
        evs.append(stats["ev_pct"] if stats["ev_pct"] is not None else np.nan)
        pos = picked[target_col].dropna()
        pos_rates.append(float(pos.mean()) if len(pos) else np.nan)
        streaks.append(stats["max_losing_streak"])
    return {"pf": pfs, "ev": evs, "pos_rate": pos_rates, "max_losing_streak": streaks}


def percentile_of(value, population) -> float | None:
    pop = [p for p in population if p is not None and not (isinstance(p, float) and np.isnan(p))]
    if value is None or not pop:
        return None
    return round(sum(1 for p in pop if p <= value) / len(pop) * 100, 1)


def daily_ic_stats(df: pd.DataFrame, prob_col: str, outcome_col: str, min_candidates: int = 5) -> dict:
    rows = []
    for d, g in df.groupby("signal_date"):
        gg = g.dropna(subset=[prob_col, outcome_col])
        if len(gg) < min_candidates:
            continue
        try:
            ic, _ = spearmanr(gg[prob_col], gg[outcome_col])
        except Exception:
            ic = np.nan
        if not np.isnan(ic):
            rows.append({"signal_date": d, "n": len(gg), "ic": ic})
    if not rows:
        return {"mean_ic": None, "median_ic": None, "std_ic": None, "icir": None, "pct_positive_days": None, "n_days": 0, "daily": []}
    ic_series = pd.Series([r["ic"] for r in rows])
    mean_ic, std_ic = float(ic_series.mean()), float(ic_series.std())
    return {
        "mean_ic": round(mean_ic, 4), "median_ic": round(float(ic_series.median()), 4),
        "std_ic": round(std_ic, 4), "icir": round(mean_ic / std_ic, 4) if std_ic > 0 else None,
        "pct_positive_days": round(float((ic_series > 0).mean() * 100), 1), "n_days": len(rows),
        "daily": rows,
    }


def topbottom_spread(df: pd.DataFrame, prob_col: str, frac: float) -> dict:
    diffs = []
    for _, g in df.groupby("signal_date"):
        n = len(g)
        k = max(1, int(round(n * frac)))
        if n < 2 * k:
            continue
        top = g.nlargest(k, prob_col)[ECONOMIC_COLUMN].mean()
        bottom = g.nsmallest(k, prob_col)[ECONOMIC_COLUMN].mean()
        diffs.append(top - bottom)
    if not diffs:
        return {"mean_spread": None, "n_days": 0}
    return {"mean_spread": round(float(np.mean(diffs)), 3), "n_days": len(diffs)}


def full_regime_metrics(g: pd.DataFrame, prob_col: str, target_col: str) -> dict:
    cm = evaluate.classification_metrics(g[target_col].values, g[prob_col].values)
    out = {**cm}
    for pct in [0.01, 0.03, 0.05, 0.10]:
        out[f"top{int(pct*100)}pct"] = evaluate.topk_economic_summary(g, prob_col, target_col, ECONOMIC_COLUMN, pct)
    for n in [1, 3, 5]:
        out[f"daily_top{n}"] = summarize_selection(daily_select(g, prob_col, target_col, n=n), target_col)
    return out


def method_score_column(df: pd.DataFrame, method: str, prob_col: str, rng) -> pd.Series:
    if method == "model":
        return df[prob_col]
    if method == "random":
        return pd.Series(rng.random(len(df)), index=df.index)
    if method == "score_desc":
        return df["technical_score_v1"]
    if method == "score_asc":
        return -df["technical_score_v1"]
    if method == "return_5d":
        return df["return_5d"]
    if method == "breakout":
        return df["breakout_close_20d_pct"]
    if method == "atr_desc":
        return df["atr_pct"]
    if method == "atr_pctrank_desc":
        return df["atr_pct_pctrank"]
    if method == "realized_vol_desc":
        return df["historical_volatility_10d"]
    raise ValueError(method)


def run_diagnostics_for_target(db: FeatureStoreDB, target_col: str, experiment_batch_id: str, run_at: str, git_commit: str, library_versions: str, is_primary: bool) -> list[str]:
    lines = [f"## 目的変数: `{target_col}`{'(主目的変数)' if is_primary else '(補助分析、ATR依存の切り分け中心)'}", ""]
    log(f"===== {target_col} 診断開始 =====")

    df, meta = load_trainval_dataset(db, universe_variant="A_eligible_universe")
    build_run = meta["build_run"]
    numeric_cols, cat_cols = meta["numeric_cols"], meta["categorical_cols"]
    train_mask, val_mask = train_val_masks(df, build_run)
    train_df, val_df = df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)

    # ---- Test封印チェック(このtarget用データロードについて) ----
    assert df["signal_date"].max() < build_run["test_start"], "Test期間が混入しています"
    assert val_df["signal_date"].max() < build_run["test_start"], "Validation最大日がtest_start以上です"
    log(f"[Test封印チェック] OK train={len(train_df)} val={len(val_df)} max_date={df['signal_date'].max()} < test_start={build_run['test_start']}")

    # ---- 1. Phase3Aモデル固定(再学習、探索なし) ----
    t0 = time.time()
    full_pipe = fit_logreg_l2_c001(train_df, train_df[target_col], numeric_cols)
    prob_val = full_pipe.predict_proba(val_df)[:, 1]
    prob_train = full_pipe.predict_proba(train_df)[:, 1]
    log(f"[固定モデル再学習] {round(time.time()-t0,1)}s train_n={len(train_df)}")

    val_eval = val_df.copy()
    val_eval[ECONOMIC_COLUMN] = val_df[ECONOMIC_COLUMN]
    val_eval["_prob"] = prob_val
    base_metrics = evaluate.classification_metrics(val_eval[target_col].values, prob_val)
    lines.append(f"- サンプル数: train={len(train_df)} val={len(val_df)} / Validation全体: ROC-AUC={base_metrics['roc_auc']} PR-AUC={base_metrics['pr_auc']} Brier={base_metrics['brier']}")
    lines.append("")

    experiment_id = f"phase3a5_{experiment_batch_id}_{target_col}"
    db.upsert_phase3a5_experiment({
        "experiment_id": experiment_id, "run_at": run_at, "git_commit": git_commit,
        "feature_set_version": FEATURE_SET_VERSION, "model_version": MODEL_VERSION, "target_variable": target_col,
        "start_date": build_run["start_date"], "end_date": build_run["end_date"],
        "train_start": build_run["train_start"], "train_end": build_run["train_end"],
        "val_start": build_run["val_start"], "val_end": build_run["val_end"],
        "test_start": build_run["test_start"], "test_end": build_run["test_end"],
        "n_train": len(train_df), "n_val": len(val_df), "used_columns": json.dumps(numeric_cols),
        "excluded_columns": json.dumps(MARKET_COLUMNS), "atr_quantile_edges": None, "probability_thresholds": None,
        "random_seed": RANDOM_SEED, "library_versions": library_versions, "notes": "Phase3A.5 診断分析",
    })

    # ============================================================
    # 2. 市場環境別評価
    # ============================================================
    log("[2] 市場環境別評価")
    lines.append("### 2. 市場環境別評価\n")
    lines.append("| 区分 | 値 | n | 陽性率 | ROC-AUC | PR-AUC | Brier | top5%陽性率 | 日次Top3 PF | 警告 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    regime_dims = {
        "market_regime": val_eval["market_regime"].astype(str),
        "spy_vs_ma200": val_eval["spy_above_ma200"].map({1: "above_ma200", 0: "below_ma200"}),
        "spy_ma200_slope_sign": (val_eval["spy_ma200_slope"] > 0).map({True: "slope_positive", False: "slope_nonpositive"}),
    }
    for dim_name, series in regime_dims.items():
        for val, idx in val_eval.groupby(series).groups.items():
            g = val_eval.loc[idx]
            warn = 1 if len(g) < 500 else 0
            m = full_regime_metrics(g, "_prob", target_col)
            lines.append(
                f"| {dim_name} | {val} | {len(g)} | {m['positive_rate']} | {m['roc_auc']} | {m['pr_auc']} | {m['brier']} | "
                f"{m['top5pct']['target_positive_rate']} | {m['daily_top3']['pf']} | {'少数注意' if warn else ''} |"
            )
            db.insert_phase3a5_regime_metric(experiment_id, dim_name, str(val), len(g), json.dumps(m, default=str), warn)
    lines.append("")

    # ============================================================
    # 3. 同日・同市場内ランキング評価(ベースライン比較)
    # ============================================================
    log("[3] 同日内ランキング評価")
    lines.append("### 3. 同日内ランキング評価(ベースライン比較)\n")
    rng = np.random.default_rng(RANDOM_SEED)
    methods = ["model", "random", "score_desc", "score_asc", "return_5d", "breakout"]
    lines.append("| 方式 | 日次Top1 PF | 日次Top3 PF | 日次Top5 PF | 日次Top10 PF | 上位1% PF | 上位5% PF | 上位10% PF |")
    lines.append("|---|---|---|---|---|---|---|---|")
    daily_rank_results = {}
    for method in methods:
        tmp = val_eval.copy()
        tmp["_score"] = method_score_column(tmp, method, "_prob", rng)
        row = {}
        for n in [1, 3, 5, 10]:
            sel = daily_select(tmp, "_score", target_col, n=n)
            s = summarize_selection(sel, target_col)
            row[f"top{n}"] = s
            db.insert_phase3a5_topn_metric(experiment_id, method, f"daily_top{n}", s["n_trades"], json.dumps(s, default=str))
        for pct in [0.01, 0.05, 0.10]:
            sel = daily_select(tmp, "_score", target_col, pct=pct)
            s = summarize_selection(sel, target_col)
            row[f"pct{int(pct*100)}"] = s
            db.insert_phase3a5_topn_metric(experiment_id, method, f"daily_top{int(pct*100)}pct", s["n_trades"], json.dumps(s, default=str))
        daily_rank_results[method] = row
        lines.append(
            f"| {method} | {row['top1']['pf']} | {row['top3']['pf']} | {row['top5']['pf']} | {row['top10']['pf']} | "
            f"{row['pct1']['pf']} | {row['pct5']['pf']} | {row['pct10']['pf']} |"
        )
    lines.append("")

    # ============================================================
    # 4. 市場中立化評価(日次IC)
    # ============================================================
    log("[4] 日次IC")
    lines.append("### 4. 市場中立化評価(日次Spearman IC、モデル予測確率 vs target_trade_pnl_pct)\n")
    ic = daily_ic_stats(val_eval, "_prob", ECONOMIC_COLUMN, min_candidates=5)
    lines.append(
        f"平均IC={ic['mean_ic']} / 中央値IC={ic['median_ic']} / 標準偏差={ic['std_ic']} / ICIR={ic['icir']} / "
        f"IC>0の日の割合={ic['pct_positive_days']}% / 対象日数={ic['n_days']}\n"
    )
    for row in ic["daily"]:
        db.insert_phase3a5_daily_ic(experiment_id, row["signal_date"], row["n"], row["ic"])
    ic_monthly = defaultdict(list)
    for row in ic["daily"]:
        ic_monthly[month_key(row["signal_date"])].append(row["ic"])
    ic_half = defaultdict(list)
    for row in ic["daily"]:
        ic_half[half_key(row["signal_date"])].append(row["ic"])
    lines.append("月別IC平均: " + json.dumps({k: round(float(np.mean(v)), 4) for k, v in sorted(ic_monthly.items())}))
    lines.append("")
    lines.append("半期別IC平均: " + json.dumps({k: round(float(np.mean(v)), 4) for k, v in sorted(ic_half.items())}))
    lines.append("")
    spread10 = topbottom_spread(val_eval, "_prob", 0.10)
    spread20 = topbottom_spread(val_eval, "_prob", 0.20)
    lines.append(f"Top10%-Bottom10%平均PnL差: {spread10['mean_spread']}(対象日数{spread10['n_days']}) / Top20%-Bottom20%: {spread20['mean_spread']}(対象日数{spread20['n_days']})\n")
    db.insert_phase3a5_metric(experiment_id, "daily_ic", "overall", "mean_ic", ic["mean_ic"], json.dumps(ic))
    db.insert_phase3a5_metric(experiment_id, "daily_ic", "overall", "icir", ic["icir"], None)
    db.insert_phase3a5_metric(experiment_id, "topbottom_spread", "10pct", "mean_spread", spread10["mean_spread"], json.dumps(spread10))
    db.insert_phase3a5_metric(experiment_id, "topbottom_spread", "20pct", "mean_spread", spread20["mean_spread"], json.dumps(spread20))

    # ============================================================
    # 5. 市場特徴量除外モデル(診断用)
    # ============================================================
    log("[5] 市場特徴量除外モデル")
    lines.append("### 5. 市場特徴量除外モデル(診断用、Feature Set v1は変更しない)\n")
    numeric_cols_excl = [c for c in numeric_cols if c not in MARKET_COLUMNS]
    lines.append(f"除外列: {MARKET_COLUMNS}\n")
    pipe_excl = fit_logreg_l2_c001(train_df, train_df[target_col], numeric_cols_excl)
    prob_val_excl = pipe_excl.predict_proba(val_df)[:, 1]
    val_excl_eval = val_eval.copy()
    val_excl_eval["_prob_excl"] = prob_val_excl
    m_excl = evaluate.classification_metrics(val_excl_eval[target_col].values, prob_val_excl)
    ic_excl = daily_ic_stats(val_excl_eval, "_prob_excl", ECONOMIC_COLUMN, min_candidates=5)
    top1_excl = summarize_selection(daily_select(val_excl_eval, "_prob_excl", target_col, n=1), target_col)
    top3_excl = summarize_selection(daily_select(val_excl_eval, "_prob_excl", target_col, n=3), target_col)
    top5_excl = summarize_selection(daily_select(val_excl_eval, "_prob_excl", target_col, n=5), target_col)
    lines.append("| モデル | ROC-AUC | PR-AUC | Brier | 平均IC | 日次Top1 PF | 日次Top3 PF | 日次Top5 PF |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(f"| Aフルモデル | {base_metrics['roc_auc']} | {base_metrics['pr_auc']} | {base_metrics['brier']} | {ic['mean_ic']} | {daily_rank_results['model']['top1']['pf']} | {daily_rank_results['model']['top3']['pf']} | {daily_rank_results['model']['top5']['pf']} |")
    lines.append(f"| B市場特徴量除外 | {m_excl['roc_auc']} | {m_excl['pr_auc']} | {m_excl['brier']} | {ic_excl['mean_ic']} | {top1_excl['pf']} | {top3_excl['pf']} | {top5_excl['pf']} |")
    lines.append("")
    db.insert_phase3a5_metric(experiment_id, "market_excluded_model", "validation", "pr_auc", m_excl["pr_auc"], json.dumps(m_excl))
    db.insert_phase3a5_metric(experiment_id, "market_excluded_model", "validation", "mean_ic", ic_excl["mean_ic"], json.dumps(ic_excl))

    # ============================================================
    # 6. ATR層別評価(Train基準の分位点)
    # ============================================================
    log("[6] ATR層別評価")
    lines.append("### 6. ATR層別評価(Train基準の5分位、atr_pct)\n")
    train_atr = train_df["atr_pct"].dropna()
    _, atr_edges = pd.qcut(train_atr, 5, retbins=True, duplicates="drop")
    atr_edges = np.array(atr_edges, dtype=float)
    atr_edges[0], atr_edges[-1] = -np.inf, np.inf
    val_eval["_atr_bucket"] = pd.cut(val_eval["atr_pct"], bins=atr_edges, labels=[f"Q{i+1}" for i in range(len(atr_edges) - 1)])
    lines.append("| ATR層 | n | 陽性率 | ROC-AUC | PR-AUC | Brier | 上位5%陽性率 | 日次Top3 PF | 平均PnL | EV |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for bucket, g in val_eval.groupby("_atr_bucket", observed=True):
        if len(g) == 0:
            continue
        m = full_regime_metrics(g, "_prob", target_col)
        pf_stats = pf_ev_stats(g[ECONOMIC_COLUMN])
        lines.append(
            f"| {bucket} | {len(g)} | {m['positive_rate']} | {m['roc_auc']} | {m['pr_auc']} | {m['brier']} | "
            f"{m['top5pct']['target_positive_rate']} | {m['daily_top3']['pf']} | {pf_stats['ev_pct']} | {pf_stats['ev_pct']} |"
        )
        db.insert_phase3a5_atr_bucket_metric(experiment_id, str(bucket), len(g), json.dumps(m, default=str))
    lines.append("")
    db.upsert_phase3a5_experiment({"experiment_id": experiment_id, "atr_quantile_edges": json.dumps(atr_edges.tolist())})

    atr_related_lines = []
    if target_col == "target_15pct_within_10d":
        # ============================================================
        # 7. ATRベースライン比較
        # ============================================================
        log("[7] ATRベースライン比較")
        atr_related_lines.append("### 7. ATR単独ベースラインとの比較(target_15pct_within_10d)\n")
        atr_baselines = ["atr_desc", "atr_pctrank_desc", "realized_vol_desc"]
        atr_related_lines.append("| 方式 | ROC-AUC | PR-AUC | 上位5%陽性率 | 日次Top3 PF |")
        atr_related_lines.append("|---|---|---|---|---|")
        for method in atr_baselines:
            tmp = val_eval.copy()
            tmp["_score"] = method_score_column(tmp, method, "_prob", rng)
            cm = evaluate.classification_metrics(tmp[target_col].values, tmp["_score"].values)
            top5 = evaluate.topk_economic_summary(tmp, "_score", target_col, ECONOMIC_COLUMN, 0.05)
            top3sel = summarize_selection(daily_select(tmp, "_score", target_col, n=3), target_col)
            atr_related_lines.append(f"| {method} | {cm['roc_auc']} | {cm['pr_auc']} | {top5['target_positive_rate']} | {top3sel['pf']} |")
            db.insert_phase3a5_metric(experiment_id, "atr_baseline", method, "pr_auc", cm["pr_auc"], json.dumps(cm))

        atr_simple_cols = ["atr_pct"]
        pipe_atr_only = fit_logreg_l2_c001(train_df, train_df[target_col], atr_simple_cols)
        prob_atr_only = pipe_atr_only.predict_proba(val_df)[:, 1]
        cm_atr_only = evaluate.classification_metrics(val_df[target_col].values, prob_atr_only)
        atr_related_lines.append(f"| LR(atr_pctのみ) | {cm_atr_only['roc_auc']} | {cm_atr_only['pr_auc']} | - | - |")
        db.insert_phase3a5_metric(experiment_id, "atr_baseline", "logreg_atr_only", "pr_auc", cm_atr_only["pr_auc"], json.dumps(cm_atr_only))

        atr_regime_cols = ["atr_pct"]
        pipe_atr_regime = fit_logreg_l2_c001(train_df, train_df[target_col], atr_regime_cols, categorical_cols=["market_regime"], include_categorical=True)
        prob_atr_regime = pipe_atr_regime.predict_proba(val_df)[:, 1]
        cm_atr_regime = evaluate.classification_metrics(val_df[target_col].values, prob_atr_regime)
        atr_related_lines.append(f"| LR(atr_pct+market_regime) | {cm_atr_regime['roc_auc']} | {cm_atr_regime['pr_auc']} | - | - |")
        atr_related_lines.append(f"| **フルモデル(参考)** | {base_metrics['roc_auc']} | {base_metrics['pr_auc']} | {full_regime_metrics(val_eval,'_prob',target_col)['top5pct']['target_positive_rate']} | {daily_rank_results['model']['top3']['pf']} |")
        atr_related_lines.append("")
        db.insert_phase3a5_metric(experiment_id, "atr_baseline", "logreg_atr_regime", "pr_auc", cm_atr_regime["pr_auc"], json.dumps(cm_atr_regime))

        judgement = "フルモデルがATR単独モデルを明確に上回る" if (base_metrics["pr_auc"] or 0) > (cm_atr_regime["pr_auc"] or 0) + 0.02 else "フルモデルとATRベースラインの差が小さく、方向予測というよりボラティリティ予測に近い可能性"
        atr_related_lines.append(f"判定: {judgement}\n")

        # ============================================================
        # 8. ATR条件付き残差分析
        # ============================================================
        log("[8] ATR残差分析")
        atr_related_lines.append("### 8. ATR条件付き残差分析(residual_score = フルモデル確率 - ATRベースライン確率)\n")
        residual_score = prob_val - prob_atr_regime
        val_resid = val_eval.copy()
        val_resid["_resid"] = residual_score
        atr_related_lines.append("| 指標 | 陽性率 | PF |")
        atr_related_lines.append("|---|---|---|")
        for pct in [0.01, 0.03, 0.05, 0.10]:
            s = evaluate.topk_economic_summary(val_resid, "_resid", target_col, ECONOMIC_COLUMN, pct)
            atr_related_lines.append(f"| 上位{int(pct*100)}% | {s['target_positive_rate']} | {s['pf']} |")
        for n in [1, 3, 5]:
            sel = summarize_selection(daily_select(val_resid, "_resid", target_col, n=n), target_col)
            atr_related_lines.append(f"| 日次Top{n} | {sel['positive_rate']} | {sel['pf']} |")
        resid_top5 = evaluate.topk_economic_summary(val_resid, "_resid", target_col, ECONOMIC_COLUMN, 0.05)
        resid_judgement = "残差スコアでも改善が見られ、ATR以外の方向性情報がある可能性" if (resid_top5["pf"] or 0) > 1.2 else "残差スコアでは明確な改善が見られず、主な予測力はATR由来と判断"
        atr_related_lines.append(f"\n判定: {resid_judgement}\n")
        db.insert_phase3a5_metric(experiment_id, "atr_residual", "top5pct", "pf", resid_top5["pf"], json.dumps(resid_top5))

    lines.extend(atr_related_lines)

    # ============================================================
    # 9. 予測確率閾値評価(Train基準)
    # ============================================================
    log("[9] 確率閾値評価")
    lines.append("### 9. 予測確率閾値評価(Train基準で決定)\n")
    train_prob_quantile_thresholds = {
        f"train_top{int(p*100)}pct": float(np.quantile(prob_train, 1 - p)) for p in [0.01, 0.03, 0.05, 0.10]
    }
    fixed_thresholds = {f"prob{t}": t for t in [0.50, 0.55, 0.60, 0.65, 0.70]}
    all_thresholds = {**train_prob_quantile_thresholds, **fixed_thresholds}
    lines.append("| 閾値名 | 閾値 | シグナル日数 | 総シグナル数 | 1日平均 | シグナルなし日数 | 陽性率 | 平均PnL | PF |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    threshold_results = {}
    n_val_days = val_eval["signal_date"].nunique()
    for name, thresh in all_thresholds.items():
        sig = val_eval[val_eval["_prob"] >= thresh]
        n_days_signal = sig["signal_date"].nunique()
        stats = pf_ev_stats(sig[ECONOMIC_COLUMN])
        pos = sig[target_col].dropna()
        row = {
            "threshold": thresh, "n_days_signal": n_days_signal, "n_total_signals": len(sig),
            "avg_signals_per_day": round(len(sig) / n_days_signal, 2) if n_days_signal else None,
            "n_days_no_signal": n_val_days - n_days_signal,
            "positive_rate": round(float(pos.mean()), 4) if len(pos) else None,
            **stats, "monthly": {k: v["pf"] for k, v in _group_pf(sig, month_key).items()},
        }
        threshold_results[name] = row
        lines.append(f"| {name} | {round(thresh,4)} | {n_days_signal} | {len(sig)} | {row['avg_signals_per_day']} | {row['n_days_no_signal']} | {row['positive_rate']} | {stats['ev_pct']} | {stats['pf']} |")
        db.insert_phase3a5_metric(experiment_id, "probability_threshold", name, "pf", stats["pf"], json.dumps(row, default=str))
    lines.append("")
    db.upsert_phase3a5_experiment({"experiment_id": experiment_id, "probability_thresholds": json.dumps(all_thresholds)})

    # ============================================================
    # 10. 日次TopN + 閾値併用
    # ============================================================
    log("[10] TopN+閾値併用")
    lines.append("### 10. 日次TopN + 確率閾値併用\n")
    lines.append("| 条件 | 取引件数 | 取引日数 | シグナルなし日数 | 陽性率 | 平均PnL | PF | 最大連敗 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n in [1, 3, 5]:
        for thresh_name in ["train_top10pct", "train_top5pct"]:
            thresh = all_thresholds[thresh_name]
            def _select_with_threshold(g, n=n, thresh=thresh):
                top = g.nlargest(min(n, len(g)), "_prob")
                return top[top["_prob"] >= thresh]
            picks = [_select_with_threshold(g) for _, g in val_eval.groupby("signal_date")]
            picked = pd.concat(picks) if picks else val_eval.iloc[0:0]
            n_days_traded = picked["signal_date"].nunique()
            stats = pf_ev_stats(picked.sort_values("signal_date")[ECONOMIC_COLUMN])
            pos = picked[target_col].dropna()
            label = f"Top{n} かつ prob>={thresh_name}"
            lines.append(f"| {label} | {stats['n']} | {n_days_traded} | {n_val_days - n_days_traded} | {round(float(pos.mean()),4) if len(pos) else None} | {stats['ev_pct']} | {stats['pf']} | {stats['max_losing_streak']} |")
            db.insert_phase3a5_topn_metric(experiment_id, "model", f"top{n}_and_{thresh_name}", stats["n"], json.dumps(stats, default=str))
    lines.append("")

    # ============================================================
    # 11. 時間安定性
    # ============================================================
    log("[11] 時間安定性")
    lines.append("### 11. 時間安定性(月別・半期別)\n")
    lines.append("#### 月別\n")
    lines.append("| 月 | n | 陽性率 | PR-AUC | 日次Top3 PF | 予測確率平均 |")
    lines.append("|---|---|---|---|---|---|")
    monthly = period_breakdown(val_eval, "_prob", target_col, month_key)
    val_eval["_month"] = val_eval["signal_date"].apply(month_key)
    for row in monthly:
        top3_month = summarize_selection(daily_select(val_eval[val_eval["_month"] == row["period"]], "_prob", target_col, n=3), target_col)
        lines.append(f"| {row['period']} | {row['n']} | {row['positive_rate']} | {row['pr_auc']} | {top3_month['pf']} | {row['mean_predicted_prob']} |")
        db.insert_phase3a5_metric(experiment_id, "time_stability_monthly", row["period"], "pr_auc", row["pr_auc"], json.dumps(row))
    lines.append("\n#### 半期別\n")
    lines.append("| 半期 | n | 陽性率 | PR-AUC | 日次Top3 PF |")
    lines.append("|---|---|---|---|---|")
    halfy = period_breakdown(val_eval, "_prob", target_col, half_key)
    val_eval["_half"] = val_eval["signal_date"].apply(half_key)
    for row in halfy:
        top3_half = summarize_selection(daily_select(val_eval[val_eval["_half"] == row["period"]], "_prob", target_col, n=3), target_col)
        lines.append(f"| {row['period']} | {row['n']} | {row['positive_rate']} | {row['pr_auc']} | {top3_half['pf']} |")
        db.insert_phase3a5_metric(experiment_id, "time_stability_halfyear", row["period"], "pr_auc", row["pr_auc"], json.dumps(row))
    lines.append("")

    # ============================================================
    # 12. 銘柄集中度
    # ============================================================
    log("[12] 銘柄集中度")
    lines.append("### 12. 銘柄集中度(日次Top3選出、フルモデル)\n")
    top3_all = daily_select(val_eval, "_prob", target_col, n=3)
    ticker_counts = top3_all["ticker"].value_counts()
    total_sel = len(top3_all)
    top10_share = round(float(ticker_counts.head(10).sum() / total_sel * 100), 1) if total_sel else None
    hhi = round(float(((ticker_counts / total_sel) ** 2).sum()), 4) if total_sel else None
    lines.append(f"総選出数={total_sel} / 上位10銘柄の構成比={top10_share}% / HHI={hhi}\n")
    lines.append("| 銘柄 | 選出数 | PF | 平均PnL | 勝率 |")
    lines.append("|---|---|---|---|---|")
    per_ticker_stats = {}
    for ticker, g in top3_all.groupby("ticker"):
        stats = pf_ev_stats(g[ECONOMIC_COLUMN])
        per_ticker_stats[ticker] = stats
        db.insert_phase3a5_symbol_metric(experiment_id, ticker, len(g), json.dumps(stats))
    for ticker in ticker_counts.head(10).index:
        s = per_ticker_stats[ticker]
        lines.append(f"| {ticker} | {ticker_counts[ticker]} | {s['pf']} | {s['ev_pct']} | {s['win_rate_pct']} |")
    overall_stats = pf_ev_stats(top3_all[ECONOMIC_COLUMN])
    lines.append(f"\n全体PF(除外なし): {overall_stats['pf']}\n")
    lines.append("| 除外銘柄 | 除外後PF |")
    lines.append("|---|---|")
    for ticker in ticker_counts.head(5).index:
        remaining = top3_all[top3_all["ticker"] != ticker]
        s = pf_ev_stats(remaining[ECONOMIC_COLUMN])
        lines.append(f"| {ticker} | {s['pf']} |")
    lines.append("")

    # ============================================================
    # 13. セクター別評価
    # ============================================================
    log("[13] セクター別評価")
    lines.append("### 13. セクター別評価(日次Top3選出)\n")
    lines.append("| セクター | モデル | n | 陽性率 | PF | 平均PnL |")
    lines.append("|---|---|---|---|---|---|")
    top3_excl_all = daily_select(val_excl_eval, "_prob_excl", target_col, n=3)
    for label, sel in [("フルモデル", top3_all), ("市場特徴量除外", top3_excl_all)]:
        for sector, g in sel.groupby("sector"):
            stats = pf_ev_stats(g[ECONOMIC_COLUMN])
            pos = g[target_col].dropna()
            lines.append(f"| {sector} | {label} | {len(g)} | {round(float(pos.mean()),4) if len(pos) else None} | {stats['pf']} | {stats['ev_pct']} |")
            db.insert_phase3a5_sector_metric(experiment_id, str(sector), "full" if label == "フルモデル" else "market_excluded", len(g), json.dumps(stats))
    lines.append("")

    # ============================================================
    # 14. ランダム比較(1000回)
    # ============================================================
    log("[14] ランダム比較1000回")
    lines.append("### 14. 日次TopNのランダム比較(1000回試行)\n")
    lines.append("| N | モデルPF | ランダムPF中央値 | モデルPFパーセンタイル | モデル陽性率パーセンタイル |")
    lines.append("|---|---|---|---|---|")
    for n in [1, 3, 5]:
        model_sel = daily_select(val_eval, "_prob", target_col, n=n)
        model_stats = pf_ev_stats(model_sel[ECONOMIC_COLUMN])
        pos = model_sel[target_col].dropna()
        model_pos_rate = float(pos.mean()) if len(pos) else None
        rand_dist = random_daily_topn_distribution(val_eval, target_col, n=n, n_trials=N_RANDOM_TRIALS, seed0=RANDOM_SEED)
        pf_pctl = percentile_of(model_stats["pf"], rand_dist["pf"])
        pos_pctl = percentile_of(model_pos_rate, rand_dist["pos_rate"])
        lines.append(f"| {n} | {model_stats['pf']} | {round(float(np.nanmedian(rand_dist['pf'])),3)} | {pf_pctl} | {pos_pctl} |")
        db.insert_phase3a5_metric(experiment_id, "random_comparison", f"daily_top{n}", "pf_percentile", pf_pctl, json.dumps({"model_pf": model_stats["pf"], "random_pf_median": float(np.nanmedian(rand_dist["pf"]))}))
    lines.append("")

    log(f"===== {target_col} 診断完了 =====")
    return lines


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    experiment_batch_id = uuid.uuid4().hex[:8]
    run_at = datetime.now(timezone.utc).isoformat()
    git_commit = git_commit_hash()
    library_versions = json.dumps({
        "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "python": platform.python_version(),
    })

    db = FeatureStoreDB(FEATURE_STORE_DB_PATH)

    report_lines = [
        f"# Phase3A.5 診断分析: 市場環境依存・ATR依存・日次TopN実運用評価 ({pd.Timestamp.today().date().isoformat()})",
        "", f"experiment_batch_id: {experiment_batch_id}", f"git_commit: {git_commit}", "",
        "Phase3Aで選定した最良モデル(logreg_l2, C=0.01, 同一前処理・同一特徴量・同一Train/Validation境界)を",
        "固定して使用(このフェーズでの再探索は行っていない)。Feature Set v1は変更していない。", "",
    ]

    for target_col, is_primary in [("target_trade_success", True), ("target_15pct_within_10d", False)]:
        section = run_diagnostics_for_target(db, target_col, experiment_batch_id, run_at, git_commit, library_versions, is_primary)
        report_lines.extend(section)
        report_lines.append("")

    db.close()

    report_path = config.REPORTS_DIR / f"phase3a5_diagnostics_{pd.Timestamp.today().date().isoformat()}.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"レポート出力: {report_path}")


if __name__ == "__main__":
    main()
