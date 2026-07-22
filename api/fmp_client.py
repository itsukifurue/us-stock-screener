"""FMP (Financial Modeling Prep) APIクライアント。stable API(2025年8月以降の新エンドポイント)を使用する。

無料プランで実際に疎通確認できたエンドポイントのみをパイプラインで使用する。
実キーでの検証(2026-07-21)の結果、無料プランでは以下が **使えない**(402 Restricted Endpoint):
  - company-screener(条件指定スクリーニング)
  - stock-list(全銘柄一覧)
  - news/stock, news/stock-latest, news/press-releases-latest(ニュース全般)
一方、以下は無料プランで動作確認済み:
  - quote, profile, income-statement(いずれも1銘柄ずつ)
  - most-actives, biggest-gainers, biggest-losers(値動き上位ランキング。市場全体の中から
    「動きのある銘柄」を無料で拾える貴重な入口。候補銘柄生成にはこれらを使う)

さらに historical-price-eod/full は、レスポンス自体は200でも「銘柄ごとに」402
(Special Endpoint)になることが判明した(AAPL/NVDA等の有名株は使えるが、実際に検証した
候補56銘柄中44銘柄が拒否された)。そのため過去株価の取得はFMPではなく
api/yfinance_client.py(Yahoo Finance、無料・APIキー不要)を使う設計にしている。
historical_prices()メソッドはFMP有料プランへの切り替え時に使えるよう残してあるだけで、
現在のパイプラインからは呼び出していない。

stable APIでは quote/profile/historical-price-eod/income-statement は1銘柄ずつしか取得できない
(v3にあった複数銘柄まとめ取得は無い)ため、このクライアントも1銘柄ずつ呼び出す設計にしている。

エンドポイントのパスはこの ENDPOINTS 辞書に一元管理する。プラン変更などで挙動が変わった場合は
scripts/test_api_connection.py で再確認し、ここを修正すればよい。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

import config
from db.database import Database

logger = logging.getLogger(__name__)

ENDPOINTS = {
    "screener": "/company-screener",  # 無料プランでは402(有料プラン向けに残置)
    "quote": "/quote",
    "historical_price": "/historical-price-eod/full",
    "profile": "/profile",
    "income_statement": "/income-statement",
    "news": "/news/stock",  # 無料プランでは402(有料プラン向けに残置)
    "most_actives": "/most-actives",
    "biggest_gainers": "/biggest-gainers",
    "biggest_losers": "/biggest-losers",
}


class FMPBudgetExceeded(RuntimeError):
    """当日のAPIリクエスト予算を超過した場合に送出する。"""


class FMPRequestError(RuntimeError):
    """FMP APIがエラーを返した場合に送出する。"""


class FMPClient:
    def __init__(self, db: Database, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 max_daily_requests: Optional[int] = None):
        self.db = db
        self.api_key = api_key or config.FMP_API_KEY
        self.base_url = (base_url or config.FMP_BASE_URL).rstrip("/")
        self.max_daily_requests = max_daily_requests or config.MAX_DAILY_REQUESTS
        self.session = requests.Session()

    def _check_budget(self) -> None:
        used = self.db.get_today_usage()
        if used >= self.max_daily_requests:
            raise FMPBudgetExceeded(
                f"本日のFMPリクエスト数が上限({self.max_daily_requests})に達しました。使用済み: {used}"
            )

    def raw(self, path: str, query_params: Optional[dict] = None) -> object:
        """ENDPOINTSに無いパスを一時的に試すための汎用呼び出し(主に疎通調査用)。"""
        return self._get(path, query_params=query_params)

    def _get(self, path: str, query_params: Optional[dict] = None, max_retries: int = 5) -> object:
        self._check_budget()
        url = f"{self.base_url}{path}"
        params = {"apikey": self.api_key, **(query_params or {})}

        # 無料プランは秒間/分間のレート制限が厳しいため、連続リクエストの間隔を空けて
        # 429(Too Many Requests)をなるべく未然に防ぐ
        time.sleep(0.3)

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                self.db.record_api_usage(1)
                if resp.status_code == 429:
                    wait = 3 * attempt
                    logger.warning("FMP rate limited (429). %s秒待機してリトライします。", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    raise FMPRequestError(f"FMP API error {resp.status_code}: {resp.text[:300]} (url={url})")
                return resp.json()
            except requests.RequestException as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("FMPリクエスト失敗(%s)。%s秒待機してリトライします。 (%s)", e, wait, url)
                time.sleep(wait)
        raise FMPRequestError(f"FMPリクエストが{max_retries}回失敗しました: {url}") from last_error

    # ---------- 候補銘柄生成(無料プランで使えるランキング系エンドポイント) ----------
    def most_actives(self) -> list[dict]:
        result = self._get(ENDPOINTS["most_actives"])
        return result if isinstance(result, list) else []

    def biggest_gainers(self) -> list[dict]:
        result = self._get(ENDPOINTS["biggest_gainers"])
        return result if isinstance(result, list) else []

    def biggest_losers(self) -> list[dict]:
        result = self._get(ENDPOINTS["biggest_losers"])
        return result if isinstance(result, list) else []

    # ---------- 有料プラン向け(無料プランでは402になるため現状未使用) ----------
    def screen_stocks(self, limit: Optional[int] = None) -> list[dict]:
        params = {
            "priceMoreThan": config.SCREEN_MIN_PRICE,
            "volumeMoreThan": config.SCREEN_MIN_AVG_VOLUME,
            "marketCapMoreThan": config.SCREEN_MIN_MARKET_CAP,
            "country": config.SCREEN_COUNTRY,
            "isEtf": "false",
            "isFund": "false",
            "isActivelyTrading": "true",
            "limit": limit or config.SCREENER_CANDIDATE_LIMIT,
        }
        result = self._get(ENDPOINTS["screener"], query_params=params)
        return result if isinstance(result, list) else []

    def news(self, symbols: list[str], limit: int = 10) -> list[dict]:
        """無料プランでは402になることを確認済み(有料プラン向けに残置)。"""
        if not symbols:
            return []
        result = self._get(
            ENDPOINTS["news"], query_params={"symbols": ",".join(symbols), "limit": limit}
        )
        return result if isinstance(result, list) else []

    # ---------- 個別銘柄API(1銘柄ずつ) ----------
    def batch_quote(self, symbols: list[str]) -> list[dict]:
        """stable APIは1銘柄ずつしか取得できないため、内部で銘柄数分リクエストする
        (=呼び出しごとに len(symbols) 件分APIリクエストを消費する)。"""
        results = []
        for symbol in symbols:
            result = self._get(ENDPOINTS["quote"], query_params={"symbol": symbol})
            if isinstance(result, list) and result:
                results.append(result[0])
        return results

    def historical_prices(self, symbols: list[str], from_date: str, to_date: str) -> dict[str, list[dict]]:
        """現在パイプラインからは未使用(api/yfinance_client.pyに置き換え済み。理由はモジュール
        docstring参照)。有料プランに切り替えた場合の参考実装として残してある。
        stable APIは1銘柄ずつしか取得できないため、内部で銘柄数分リクエストする
        (=呼び出しごとに len(symbols) 件分APIリクエストを消費する)。
        戻り値: {symbol: [OHLCVレコード, ...]}"""
        out: dict[str, list[dict]] = {}
        for symbol in symbols:
            result = self._get(
                ENDPOINTS["historical_price"],
                query_params={"symbol": symbol, "from": from_date, "to": to_date},
            )
            if isinstance(result, list):
                out[symbol] = result
        return out

    def company_profile(self, symbol: str) -> Optional[dict]:
        result = self._get(ENDPOINTS["profile"], query_params={"symbol": symbol})
        if isinstance(result, list) and result:
            return result[0]
        return None

    def income_statement(self, symbol: str, limit: int = 4, period: str = "quarter") -> list[dict]:
        result = self._get(
            ENDPOINTS["income_statement"],
            query_params={"symbol": symbol, "limit": limit, "period": period},
        )
        return result if isinstance(result, list) else []
