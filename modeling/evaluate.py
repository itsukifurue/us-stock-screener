"""Phase3A: 評価指標(分類指標・確率校正・順位付け評価)。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "n": len(y_true),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "mean_predicted_prob": round(float(np.mean(y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    except ValueError:
        out["pr_auc"] = None
    try:
        out["log_loss"] = round(float(log_loss(y_true, y_prob, labels=[0, 1])), 4)
    except ValueError:
        out["log_loss"] = None
    return out


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """予測確率decile別の実際の陽性率とExpected Calibration Errorの元データを返す。"""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    try:
        df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], n_bins)
    rows = []
    for b, g in df.groupby("bin", observed=True):
        rows.append({
            "bin": str(b), "n": len(g),
            "mean_predicted": round(float(g["p"].mean()), 4),
            "actual_positive_rate": round(float(g["y"].mean()), 4),
        })
    return rows


def expected_calibration_error(cal_table: list[dict], total_n: int) -> float:
    ece = sum(row["n"] / total_n * abs(row["mean_predicted"] - row["actual_positive_rate"]) for row in cal_table)
    return round(ece, 4)


def topk_economic_summary(df: pd.DataFrame, score_col: str, target_col: str, pnl_col: str, k_frac: float) -> dict:
    """予測確率(またはスコア)上位k_frac割合(プール全体基準)の経済成績を計算する。"""
    n = len(df)
    k = max(1, int(n * k_frac))
    top = df.nlargest(k, score_col)
    pnl = top[pnl_col].dropna()
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = round(wins / losses, 3) if losses > 0 else None
    return {
        "k_frac": k_frac, "n": len(top),
        "target_positive_rate": round(float(top[target_col].mean()), 4) if top[target_col].notna().any() else None,
        "pnl_mean": round(float(pnl.mean()), 3) if len(pnl) else None,
        "pnl_median": round(float(pnl.median()), 3) if len(pnl) else None,
        "pf": pf,
    }


def daily_topn_economic_summary(df: pd.DataFrame, score_col: str, target_col: str, pnl_col: str, n_per_day: int) -> dict:
    """各営業日ごとに上位n_per_day銘柄を選び、その経済成績を集計する
    (候補数の多い日に支配されないよう、日単位で選抜してからまとめて集計する)。
    """
    picks = []
    for _, group in df.groupby("signal_date"):
        picks.append(group.nlargest(min(n_per_day, len(group)), score_col))
    picked = pd.concat(picks) if picks else df.iloc[0:0]
    pnl = picked[pnl_col].dropna()
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = round(wins / losses, 3) if losses > 0 else None
    return {
        "n_per_day": n_per_day, "n_total_picks": len(picked),
        "target_positive_rate": round(float(picked[target_col].mean()), 4) if picked[target_col].notna().any() else None,
        "pnl_mean": round(float(pnl.mean()), 3) if len(pnl) else None,
        "pnl_median": round(float(pnl.median()), 3) if len(pnl) else None,
        "pf": pf,
    }
