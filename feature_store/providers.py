"""データソースの抽象化層(Phase 2)。

これまでyfinanceへの依存が`scripts/build_feature_store.py`や`feature_store/universe.py`に
直接書かれていたが、将来的に別のデータソース(point-in-time構成銘柄データ、上場廃止銘柄データ、
有料APIなど)へ差し替えられるように、3つのインターフェースに切り出す。

- MarketDataProvider: 銘柄の過去OHLCV取得
- FundamentalDataProvider: 銘柄の属性(セクター/業種/取引所/時価総額)取得
- UniverseProvider: 対象ユニバース(銘柄リスト)と、日次の候補判定条件の提供

現時点の実装(YFinance*)は全てyfinanceのみを使うが、呼び出し側(build_feature_store*.py)は
これらのインターフェースだけに依存させ、yfinance固有のAPIを直接呼ばないようにする。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_historical_prices(self, symbol: str, from_date: str, to_date: str) -> list[dict]:
        """[{"date","open","high","low","close","volume"}, ...] を返す。取得失敗時は空リスト。"""
        raise NotImplementedError


class FundamentalDataProvider(ABC):
    @abstractmethod
    def get_company_meta(self, symbol: str) -> dict:
        """{"exchange","sector","industry","market_cap"} を返す。取得失敗時は全てNone。"""
        raise NotImplementedError


class UniverseProvider(ABC):
    @abstractmethod
    def get_universe_symbols(self) -> list[str]:
        """このビルドで対象とする固定銘柄リストを返す。"""
        raise NotImplementedError

    @abstractmethod
    def evaluate_candidate(
        self, row: "pd.Series", market_cap: Optional[float]
    ) -> tuple[bool, str]:
        """(候補かどうか, 理由文字列) を返す。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def universe_version(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def candidate_source_label(self) -> str:
        raise NotImplementedError


class YFinanceMarketDataProvider(MarketDataProvider):
    def fetch_historical_prices(self, symbol: str, from_date: str, to_date: str) -> list[dict]:
        from api.yfinance_client import fetch_historical_prices
        return fetch_historical_prices(symbol, from_date, to_date)


class YFinanceFundamentalDataProvider(FundamentalDataProvider):
    def get_company_meta(self, symbol: str) -> dict:
        import yfinance as yf
        try:
            info = yf.Ticker(symbol).info
        except Exception:
            info = {}
        return {
            "exchange": info.get("exchange"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
        }


class FixedWatchlistUniverseProvider(UniverseProvider):
    """あらかじめ決めた固定銘柄リストの中から、一次スクリーニング条件で候補日を判定する近似ユニバース。
    真の「その日の値動き上位」ランキングではない点は feature_store/universe.py 参照。
    """

    def __init__(
        self,
        symbols: list[str],
        min_price: float,
        min_market_cap: float,
        min_avg_volume: float,
    ):
        self._symbols = list(symbols)
        self._min_price = min_price
        self._min_market_cap = min_market_cap
        self._min_avg_volume = min_avg_volume

    def get_universe_symbols(self) -> list[str]:
        return list(self._symbols)

    def evaluate_candidate(self, row: "pd.Series", market_cap: Optional[float]) -> tuple[bool, str]:
        from feature_store.universe import evaluate_candidate
        return evaluate_candidate(row, self._min_price, self._min_market_cap, self._min_avg_volume, market_cap)

    @property
    def universe_version(self) -> str:
        from feature_store.universe import UNIVERSE_VERSION
        return UNIVERSE_VERSION

    @property
    def candidate_source_label(self) -> str:
        from feature_store.universe import APPROX_SOURCE_LABEL
        return APPROX_SOURCE_LABEL
