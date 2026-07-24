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

import hashlib
import logging
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


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


def _retry(fn, max_retries: int = 3, base_delay: float = 1.5, what: str = ""):
    """指数バックオフ付きリトライ。最終的に失敗した場合は例外を投げずNone相当(呼び出し側で判定)。"""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- 外部データソースの失敗は種類を問わず継続したい
            last_exc = e
            logger.warning("取得失敗(%s, 試行%d/%d): %s", what, attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(base_delay * attempt)
    logger.error("取得を%d回試行しましたが失敗しました(%s): %s", max_retries, what, last_exc)
    return None


class CachingMarketDataProvider(MarketDataProvider):
    """他のMarketDataProviderをラップし、ディスクキャッシュ・リトライ・差分取得・
    メタデータ記録(price_cache_metaテーブル)を追加する。

    キャッシュは銘柄ごとに1ファイル(data/price_cache/{symbol}.pkl、Git管理外)。
    要求範囲[from_date, to_date]が既存キャッシュに完全に含まれていれば再取得しない。
    既存キャッシュのfetch_endより後ろだけを要求している場合は、その差分だけをfetchして
    キャッシュへ追記する(差分取得)。それ以外(要求開始日がキャッシュより前など)は全期間を
    再取得する(部分マージの複雑化を避けるための簡略化。将来的な改善余地として明記)。
    """

    CACHE_VERSION = "v1"
    ADJUSTMENT_MODE = "split_adjusted_no_dividend"  # api/yfinance_clientの仕様(auto_adjust=False)固定

    def __init__(self, inner: MarketDataProvider, cache_dir: Path, db=None, max_retries: int = 3):
        self._inner = inner
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db = db
        self._max_retries = max_retries

    def _cache_path(self, symbol: str) -> Path:
        return self._cache_dir / f"{symbol}.pkl"

    def _load_cache(self, symbol: str) -> Optional[dict]:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, symbol: str, cache: dict) -> None:
        path = self._cache_path(symbol)
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        if self._db is not None:
            try:
                data_bytes = pickle.dumps(cache["history"])
                self._db.upsert_price_cache_meta({
                    "symbol": symbol, "provider": "yfinance", "cache_version": self.CACHE_VERSION,
                    "fetch_start": cache["fetch_start"], "fetch_end": cache["fetch_end"],
                    "fetched_at": cache["fetched_at"], "row_count": len(cache["history"]),
                    "adjustment_mode": self.ADJUSTMENT_MODE,
                    "data_hash": hashlib.sha256(data_bytes).hexdigest()[:16],
                    "provider_version": self.CACHE_VERSION,
                })
            except Exception as e:  # noqa: BLE001 -- メタ記録の失敗でキャッシュ自体は失敗させない
                logger.warning("price_cache_metaの記録に失敗(%s): %s", symbol, e)

    def fetch_historical_prices(self, symbol: str, from_date: str, to_date: str) -> list[dict]:
        cache = self._load_cache(symbol)
        if cache is not None and cache["fetch_start"] <= from_date and cache["fetch_end"] >= to_date:
            return [row for row in cache["history"] if from_date <= row["date"] <= to_date]

        if cache is not None and cache["fetch_end"] < to_date and cache["fetch_start"] <= from_date:
            # 差分取得: 既存キャッシュの末尾より後ろだけを追加取得する
            next_start = cache["fetch_end"]  # 重複日はマージ時にdedupする
            new_rows = _retry(
                lambda: self._inner.fetch_historical_prices(symbol, next_start, to_date),
                max_retries=self._max_retries, what=f"{symbol}差分取得",
            )
            if new_rows is not None:
                merged = {row["date"]: row for row in cache["history"]}
                for row in new_rows:
                    merged[row["date"]] = row
                history = [merged[d] for d in sorted(merged)]
                cache = {
                    "history": history, "fetch_start": cache["fetch_start"], "fetch_end": to_date,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                }
                self._save_cache(symbol, cache)
                return [row for row in history if from_date <= row["date"] <= to_date]

        # キャッシュなし、または上記条件に当てはまらない場合は全期間を取得
        history = _retry(
            lambda: self._inner.fetch_historical_prices(symbol, from_date, to_date),
            max_retries=self._max_retries, what=f"{symbol}全期間取得",
        )
        if history is None:
            return []
        cache = {
            "history": history, "fetch_start": from_date, "fetch_end": to_date,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_cache(symbol, cache)
        return history


_default_provider_singleton: Optional["CachingMarketDataProvider"] = None


def default_market_data_provider(db=None) -> "CachingMarketDataProvider":
    """全モジュール共通のデフォルトMarketDataProvider(キャッシュ・リトライつき)を返す。
    market_regime.py/sector.py/build_feature_store_phase2_step2.py はいずれもこれ経由で
    価格データを取得し、yfinanceへ直接アクセスしない。
    """
    global _default_provider_singleton
    if _default_provider_singleton is None:
        import config
        _default_provider_singleton = CachingMarketDataProvider(
            YFinanceMarketDataProvider(), config.DATA_DIR / "price_cache", db=db,
        )
    return _default_provider_singleton


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
