"""セクター相対強度の計算。

セクター分類はyfinanceのTicker.info['sector']を使う(現在の分類を過去に遡って適用する
簡易近似。過去のある時点で実際にどのセクターに分類されていたかとは異なる可能性がある
=真のpoint-in-timeではない点に注意。将来的にはFMPの企業プロフィール履歴等で改善余地あり)。

セクターETFマッピング(SPDRセクターETFに準拠、固定・明示):
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


def get_sector_etf(sector: Optional[str]) -> Optional[str]:
    if not sector:
        return None
    return SECTOR_ETF_MAP.get(sector)


class SectorContext:
    """セクターETFの日次データを事前ロードし、日付指定でリターンを返す。"""

    def __init__(self, start_date: str, end_date: str, sectors: list[str], market_data_provider=None):
        from datetime import date, timedelta

        from feature_store.providers import default_market_data_provider

        provider = market_data_provider or default_market_data_provider()
        lookback_start = (date.fromisoformat(start_date) - timedelta(days=60)).isoformat()
        self.frames: dict[str, Optional[pd.DataFrame]] = {}
        etfs_needed = {get_sector_etf(s) for s in sectors if get_sector_etf(s)}
        for etf in etfs_needed:
            history = provider.fetch_historical_prices(etf, lookback_start, end_date)
            self.frames[etf] = self._build_frame(history)

    @staticmethod
    def _build_frame(price_history: list[dict]) -> Optional[pd.DataFrame]:
        if not price_history:
            return None
        df = pd.DataFrame(price_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["return_5d"] = df["close"].pct_change(5) * 100
        df["return_20d"] = df["close"].pct_change(20) * 100
        return df

    def get_sector_return(self, sector: Optional[str], target_date: str) -> dict:
        etf = get_sector_etf(sector)
        empty = {"sector_etf": etf, "sector_return_5d": None, "sector_return_20d": None}
        if etf is None or self.frames.get(etf) is None:
            return empty
        df = self.frames[etf]
        matches = df.loc[df["date"] == pd.Timestamp(target_date)]
        if matches.empty:
            return empty
        row = matches.iloc[0]
        return {
            "sector_etf": etf,
            "sector_return_5d": float(row["return_5d"]) if pd.notna(row["return_5d"]) else None,
            "sector_return_20d": float(row["return_20d"]) if pd.notna(row["return_20d"]) else None,
        }
