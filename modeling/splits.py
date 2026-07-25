"""Phase3A: Train内部のwalk-forward分割(embargo付き)。

ランダムなtrain_test_splitは使わない。Train期間内の実際の営業日(signal_dateの値そのもの)を
昇順に並べ、拡大窓(expanding window)方式でfoldを作る: fold kの学習データは
Train開始からfold kの学習終了日まで、検証データはそこからembargo_trading_days営業日space
空けた次のブロック。embargo期間はラベルの将来ウィンドウ(最大10営業日)が学習データと
検証データの間で重複しないようにするためのもの。
"""
from __future__ import annotations

import pandas as pd

DEFAULT_EMBARGO_TRADING_DAYS = 10  # 最大保有10営業日のラベルウィンドウが跨がないようにする


def build_walk_forward_folds(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    n_folds: int = 5,
    embargo_trading_days: int = DEFAULT_EMBARGO_TRADING_DAYS,
) -> list[dict]:
    """Train期間内の実際のsignal_date値からfoldを構築する。
    戻り値: [{"fold": 1, "train_dates": set[str], "val_dates": set[str],
              "train_range": (start,end), "val_range": (start,end)}, ...]
    """
    dates = sorted(
        df.loc[(df["signal_date"] >= train_start) & (df["signal_date"] < train_end), "signal_date"].unique()
    )
    n = len(dates)
    piece = n // (n_folds + 1)
    if piece == 0:
        raise ValueError(f"Train期間の営業日数({n})がn_folds({n_folds})に対して少なすぎます")

    folds = []
    for k in range(1, n_folds + 1):
        train_end_idx = piece * k
        val_start_idx = train_end_idx + embargo_trading_days
        val_end_idx = min(val_start_idx + piece, n)
        if val_start_idx >= n or val_end_idx <= val_start_idx:
            break
        train_dates = set(dates[:train_end_idx])
        val_dates = set(dates[val_start_idx:val_end_idx])
        folds.append({
            "fold": k,
            "train_dates": train_dates,
            "val_dates": val_dates,
            "train_range": (dates[0], dates[train_end_idx - 1]),
            "val_range": (dates[val_start_idx], dates[val_end_idx - 1]),
            "embargo_trading_days": embargo_trading_days,
        })
    return folds


def split_by_dates(df: pd.DataFrame, dates: set[str]) -> pd.DataFrame:
    return df[df["signal_date"].isin(dates)]
