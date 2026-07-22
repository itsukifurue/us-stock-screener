"""Yahoo Finance(yfinanceライブラリ)から過去株価を取得するクライアント。APIキー不要・無料。

FMP無料プランの historical-price-eod/full は、実キーでの検証(2026-07-21)の結果、
ごく一部の有名な大企業銘柄にしか使えない(それ以外は402 Special Endpoint)ことが判明した。
このツールの目的は小型株も含めたスイングトレード候補の発掘であるため、過去株価の取得だけを
yfinanceに切り替える。企業概要・財務・候補銘柄生成(most-actives/biggest-gainers)は
引き続きFMPを使う(これらは小型株でも問題なく動作することを確認済み)。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_historical_prices(symbol: str, from_date: str, to_date: str) -> list[dict]:
    """指定期間(from_date〜to_date、両端含む)のOHLCVを取得する。

    戻り値は db.insert_daily_prices にそのまま渡せる形式:
    [{"date": "YYYY-MM-DD", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
    取得に失敗した場合は空リストを返す(呼び出し側でその銘柄をスキップする)。
    """
    try:
        # yfinanceのendは「その日を含まない」ため、to_dateを含めるために1日後ろにずらす
        end_exclusive = (date.fromisoformat(to_date) + timedelta(days=1)).isoformat()
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=from_date, end=end_exclusive, interval="1d", auto_adjust=False)
    except Exception as e:
        logger.warning("yfinanceでの取得に失敗しました: %s (%s)", symbol, e)
        return []

    if hist is None or hist.empty:
        return []

    rows = []
    for idx, row in hist.iterrows():
        try:
            rows.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                }
            )
        except (KeyError, ValueError, TypeError):
            continue
    return rows
