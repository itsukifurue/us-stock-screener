"""Phase3A: 比較ベースライン(Dummy・Version1スコア順・ランダム・return_5d順・breakout順)。

いずれも「予測確率」ではなく「順位付けスコア」を返す(値が大きいほど上位)。
DummyClassifierは確率としては全体陽性率を返す定数モデルとして別途scikit-learnで実装する
(evaluate.py側で分類指標を計算する際に使う)。ここではランキング系ベースラインのみを扱う。
"""
from __future__ import annotations

import random
import statistics

import numpy as np
import pandas as pd


def score_rank_desc(df: pd.DataFrame) -> pd.Series:
    return df["technical_score_v1"]


def score_rank_asc(df: pd.DataFrame) -> pd.Series:
    return -df["technical_score_v1"]


def return_5d_rank_desc(df: pd.DataFrame) -> pd.Series:
    return df["return_5d"]


def breakout_rank_desc(df: pd.DataFrame) -> pd.Series:
    return df["breakout_close_20d_pct"]


def random_rank(df: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(len(df)), index=df.index)


def overall_positive_rate(y: pd.Series) -> float:
    return float(y.mean())


def random_rank_topk_distribution(
    df: pd.DataFrame, target_col: str, pnl_col: str, k_frac: float, n_trials: int = 1000, seed0: int = 0,
) -> dict:
    """ランダム順位でtop-k_frac(日次ではなく全体プールに対する割合)を選んだ場合の
    陽性率・pnl平均の分布をn_trials回試行して返す(median/p5/p95)。
    """
    n = len(df)
    k = max(1, int(n * k_frac))
    pos_rates, pnl_means = [], []
    y = df[target_col].values
    pnl = df[pnl_col].values
    rng_master = np.random.default_rng(seed0)
    for _ in range(n_trials):
        idx = rng_master.choice(n, size=k, replace=False)
        pos_rates.append(float(np.nanmean(y[idx])))
        pnl_means.append(float(np.nanmean(pnl[idx])))
    return {
        "pos_rate_median": statistics.median(pos_rates),
        "pos_rate_p5": np.percentile(pos_rates, 5),
        "pos_rate_p95": np.percentile(pos_rates, 95),
        "pnl_mean_median": statistics.median(pnl_means),
        "pnl_mean_p5": np.percentile(pnl_means, 5),
        "pnl_mean_p95": np.percentile(pnl_means, 95),
        "pos_rates_raw": pos_rates,
        "pnl_means_raw": pnl_means,
    }


def percentile_rank_of_value(value: float, population: list[float]) -> float:
    return round(sum(1 for v in population if v <= value) / len(population) * 100, 1)
