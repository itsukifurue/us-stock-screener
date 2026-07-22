"""テクニカル指標の計算。pandasのrolling/ewmで自前実装する(外部TAライブラリ不使用)。

compute_indicator_frame() は全期間分をベクトル化(rolling)で一気に計算する。
- 本番パイプライン(最新日だけ知りたい)は compute_indicators() を使う
- バックテスト(過去の各日について知りたい)は compute_indicator_frame() + row_to_indicators() を使う
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _to_dataframe(price_history: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(price_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss=0のケース: 上昇のみなら100(買われすぎ)、値動きなし(avg_gainも0)なら中立50
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    return pd.Series(rsi, index=close.index).fillna(50)  # データ不足分は中立の50とする


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_indicator_frame(price_history: list[dict], min_rows: int = 30) -> Optional[pd.DataFrame]:
    """price_history: [{date, open, high, low, close, volume}, ...] (順不同で可、内部で日付昇順に整列)

    各日についてテクニカル指標を計算した列を追加したDataFrameを返す。
    データが min_rows 未満の場合は None を返す。
    """
    if len(price_history) < min_rows:
        return None

    df = _to_dataframe(price_history)
    close = df["close"]
    volume = df["volume"]

    df["sma5"] = close.rolling(5, min_periods=5).mean()
    df["sma15"] = close.rolling(15, min_periods=15).mean()
    df["sma25"] = close.rolling(25, min_periods=25).mean()
    df["sma50"] = close.rolling(50, min_periods=50).mean()
    df["avg_volume20"] = volume.rolling(20, min_periods=20).mean()
    df["rsi14"] = _rsi(close, 14)
    macd_line, signal_line, hist = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["atr14"] = _atr(df, 14)
    bb_upper, bb_mid, bb_lower = _bollinger(close, 20)
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower

    df["volume_ratio"] = volume / df["avg_volume20"]

    # 「当日を含まない直近19日の最高値」を当日終値が上回っていれば20日高値更新
    prev19_high = close.shift(1).rolling(19, min_periods=19).max()
    df["new_20d_high"] = close > prev19_high
    df["breakout_pct"] = (close - prev19_high) / prev19_high * 100  # 20日高値からの突破率(マイナスも許容)

    # 52週(直近252営業日、無ければあるだけ)の高値と、そこからの距離
    df["high_52w"] = df["high"].rolling(window=252, min_periods=1).max()
    df["distance_from_52w_high_pct"] = (close - df["high_52w"]) / df["high_52w"] * 100

    return df


def row_to_indicators(df: pd.DataFrame, idx: int) -> dict:
    """compute_indicator_frame()が返すDataFrameの特定行を、compute_indicators()と同じ形式の辞書にする。"""
    row = df.iloc[idx]
    current_close = float(row["close"])
    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "current_close": current_close,
        "current_volume": float(row["volume"]),
        "sma5": _safe_float(row["sma5"]),
        "sma15": _safe_float(row["sma15"]),
        "sma25": _safe_float(row["sma25"]),
        "sma50": _safe_float(row["sma50"]),
        "avg_volume20": _safe_float(row["avg_volume20"]),
        "volume_ratio": _safe_float(row["volume_ratio"]),
        "rsi14": _safe_float(row["rsi14"]),
        "macd": _safe_float(row["macd"]),
        "macd_signal": _safe_float(row["macd_signal"]),
        "macd_hist": _safe_float(row["macd_hist"]),
        "atr14": _safe_float(row["atr14"]),
        "bb_upper": _safe_float(row["bb_upper"]),
        "bb_mid": _safe_float(row["bb_mid"]),
        "bb_lower": _safe_float(row["bb_lower"]),
        "high_52w": _safe_float(row["high_52w"]),
        "distance_from_52w_high_pct": _safe_float(row["distance_from_52w_high_pct"]),
        "new_20d_high": bool(row["new_20d_high"]) if pd.notna(row["new_20d_high"]) else False,
        "breakout_pct": _safe_float(row["breakout_pct"]),
    }


def compute_indicators(price_history: list[dict], min_rows: int = 30) -> Optional[dict]:
    """最新日の指標だけを取得する(本番パイプライン用)。"""
    df = compute_indicator_frame(price_history, min_rows=min_rows)
    if df is None:
        return None
    return row_to_indicators(df, len(df) - 1)


def _safe_float(v) -> Optional[float]:
    return float(v) if pd.notna(v) else None
