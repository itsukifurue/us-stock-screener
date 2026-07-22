"""市場環境(SPY/QQQ/IWM/VIX)の特徴量とレジーム分類。

market_regime分類ルール(明示・固定、Version2 Phase1の初期定義):
  spy_above_ma200 かつ spy_above_ma50 の場合:
      spy_return_5d > +2.0% なら "strong_bull"、それ以外は "bull"
  spy_above_ma200 が偽 かつ spy_above_ma50 が偽 の場合:
      spy_return_5d < -2.0% なら "strong_bear"、それ以外は "bear"
  上記いずれにも当てはまらない場合(200日線と50日線で判定が割れている過渡期): "neutral"
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from api.yfinance_client import fetch_historical_prices

MARKET_TICKERS = ["SPY", "QQQ", "IWM", "^VIX"]


def classify_regime(spy_above_ma200: Optional[bool], spy_above_ma50: Optional[bool], spy_return_5d: Optional[float]) -> str:
    if spy_above_ma200 is None or spy_above_ma50 is None:
        return "neutral"
    r5 = spy_return_5d if spy_return_5d is not None else 0.0
    if spy_above_ma200 and spy_above_ma50:
        return "strong_bull" if r5 > 2.0 else "bull"
    if (not spy_above_ma200) and (not spy_above_ma50):
        return "strong_bear" if r5 < -2.0 else "bear"
    return "neutral"


def _build_frame(price_history: list[dict]) -> Optional[pd.DataFrame]:
    if not price_history:
        return None
    df = pd.DataFrame(price_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    close = df["close"]
    df["ma20"] = close.rolling(20, min_periods=20).mean()
    df["ma50"] = close.rolling(50, min_periods=50).mean()
    df["ma200"] = close.rolling(200, min_periods=200).mean()
    df["return_1d"] = close.pct_change(1) * 100
    df["return_5d"] = close.pct_change(5) * 100
    df["ma20_slope"] = df["ma20"].diff(5)
    df["ma200_slope"] = df["ma200"].diff(5)
    return df


class MarketContext:
    """SPY/QQQ/IWM/VIXの日次データを事前ロードし、日付指定で市場環境特徴量を返す。"""

    def __init__(self, start_date: str, end_date: str):
        lookback_start = (date.fromisoformat(start_date) - timedelta(days=400)).isoformat()
        self.frames: dict[str, Optional[pd.DataFrame]] = {}
        for ticker in MARKET_TICKERS:
            history = fetch_historical_prices(ticker, lookback_start, end_date)
            self.frames[ticker] = _build_frame(history)

    def _row_for(self, ticker: str, target_date: str) -> Optional[pd.Series]:
        df = self.frames.get(ticker)
        if df is None:
            return None
        matches = df.loc[df["date"] == pd.Timestamp(target_date)]
        if matches.empty:
            return None
        return matches.iloc[0]

    @staticmethod
    def _safe(row, col):
        if row is None or col not in row or pd.isna(row[col]):
            return None
        return float(row[col])

    def get_features(self, target_date: str) -> dict:
        spy = self._row_for("SPY", target_date)
        qqq = self._row_for("QQQ", target_date)
        iwm = self._row_for("IWM", target_date)
        vix = self._row_for("^VIX", target_date)

        spy_close = self._safe(spy, "close")
        spy_ma20 = self._safe(spy, "ma20")
        spy_ma50 = self._safe(spy, "ma50")
        spy_ma200 = self._safe(spy, "ma200")

        spy_above_ma20 = (spy_close > spy_ma20) if (spy_close is not None and spy_ma20 is not None) else None
        spy_above_ma50 = (spy_close > spy_ma50) if (spy_close is not None and spy_ma50 is not None) else None
        spy_above_ma200 = (spy_close > spy_ma200) if (spy_close is not None and spy_ma200 is not None) else None

        spy_return_1d = self._safe(spy, "return_1d")
        spy_return_5d = self._safe(spy, "return_5d")

        return {
            "spy_close": spy_close,
            "spy_return_1d": spy_return_1d,
            "spy_return_5d": spy_return_5d,
            "spy_above_ma20": _to_int_or_none(spy_above_ma20),
            "spy_above_ma50": _to_int_or_none(spy_above_ma50),
            "spy_above_ma200": _to_int_or_none(spy_above_ma200),
            "spy_ma20_slope": self._safe(spy, "ma20_slope"),
            "spy_ma200_slope": self._safe(spy, "ma200_slope"),
            "qqq_return_5d": self._safe(qqq, "return_5d"),
            "iwm_return_5d": self._safe(iwm, "return_5d"),
            "vix_close": self._safe(vix, "close"),
            "vix_return_5d": self._safe(vix, "return_5d"),
            "market_regime": classify_regime(spy_above_ma200, spy_above_ma50, spy_return_5d),
        }


def _to_int_or_none(v: Optional[bool]) -> Optional[int]:
    return None if v is None else int(v)
