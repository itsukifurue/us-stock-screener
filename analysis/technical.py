"""テクニカル指標の計算。pandasのrolling/ewmで自前実装する(外部TAライブラリ不使用)。"""
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


def compute_indicators(price_history: list[dict], min_rows: int = 30) -> Optional[dict]:
    """price_history: [{date, open, high, low, close, volume}, ...] (順不同で可、内部で日付昇順に整列)

    データが min_rows 未満の場合は None を返す(指標計算に不十分なため)。
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
    df["rolling_high_20"] = close.rolling(20, min_periods=20).max()
    lookback_52w = min(len(df), 252)
    high_52w = df["high"].tail(lookback_52w).max()

    latest = df.iloc[-1]
    prev20_high_excl_today = close.iloc[-20:-1].max() if len(df) >= 21 else np.nan

    current_close = float(latest["close"])
    avg_volume20 = float(latest["avg_volume20"]) if pd.notna(latest["avg_volume20"]) else None
    current_volume = float(latest["volume"])

    return {
        "date": latest["date"].strftime("%Y-%m-%d"),
        "current_close": current_close,
        "current_volume": current_volume,
        "sma5": _safe_float(latest["sma5"]),
        "sma15": _safe_float(latest["sma15"]),
        "sma25": _safe_float(latest["sma25"]),
        "sma50": _safe_float(latest["sma50"]),
        "avg_volume20": avg_volume20,
        "volume_ratio": (current_volume / avg_volume20) if avg_volume20 else None,
        "rsi14": _safe_float(latest["rsi14"]),
        "macd": _safe_float(latest["macd"]),
        "macd_signal": _safe_float(latest["macd_signal"]),
        "macd_hist": _safe_float(latest["macd_hist"]),
        "atr14": _safe_float(latest["atr14"]),
        "bb_upper": _safe_float(latest["bb_upper"]),
        "bb_mid": _safe_float(latest["bb_mid"]),
        "bb_lower": _safe_float(latest["bb_lower"]),
        "high_52w": float(high_52w) if pd.notna(high_52w) else None,
        "distance_from_52w_high_pct": (
            (current_close - high_52w) / high_52w * 100 if pd.notna(high_52w) and high_52w else None
        ),
        "new_20d_high": bool(pd.notna(prev20_high_excl_today) and current_close > prev20_high_excl_today),
    }


def _safe_float(v) -> Optional[float]:
    return float(v) if pd.notna(v) else None
