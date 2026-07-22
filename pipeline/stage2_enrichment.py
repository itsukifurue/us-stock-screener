"""Stage2: 過去株価取得(差分キャッシュ)→テクニカル計算→出来高フィルタ→スコアリング→上位N銘柄選出。

過去株価はFMPではなくyfinance(api/yfinance_client.py)から取得する。理由は
api/yfinance_client.py のモジュールdocstringを参照(FMP無料プランのhistorical-price-eod/fullが
ごく一部の有名銘柄にしか使えないことが実キーでの検証で判明したため)。
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import config
from analysis.scoring import score_technical
from analysis.technical import compute_indicators
from api.yfinance_client import fetch_historical_prices
from db.database import Database

logger = logging.getLogger(__name__)


def _fetch_missing_history(db: Database, symbols: list[str]) -> None:
    """DBにキャッシュされていない期間の株価のみを取得し、daily_pricesへ追記する(差分方式)。"""
    today = date.today()
    lookback_start = (today - timedelta(days=config.HISTORICAL_LOOKBACK_DAYS)).isoformat()
    to_date = today.isoformat()

    for symbol in symbols:
        latest = db.latest_price_date(symbol)
        if latest and latest >= to_date:
            continue
        from_date = (date.fromisoformat(latest) + timedelta(days=1)).isoformat() if latest else lookback_start
        rows = fetch_historical_prices(symbol, from_date, to_date)
        if rows:
            db.insert_daily_prices(symbol, rows)
        else:
            logger.warning("過去株価が取得できませんでした: %s", symbol)


def run_stage2(db: Database, run_id: int, symbols: list[str]) -> list[dict]:
    """戻り値: technical_score降順の上位config.TOP_N_TECHNICAL件
    各要素: {symbol, technical_score, breakdown, indicators}
    """
    _fetch_missing_history(db, symbols)

    scored: list[dict] = []
    for symbol in symbols:
        history = db.get_price_history(symbol)
        indicators = compute_indicators(history)
        if indicators is None:
            continue
        # 一次スクリーニング条件のうち出来高フィルタはここで適用する
        # (stage1で使うランキングAPIのレスポンスに出来高が含まれないため)
        avg_volume20 = indicators.get("avg_volume20")
        if avg_volume20 is not None and avg_volume20 < config.SCREEN_MIN_AVG_VOLUME:
            continue
        result = score_technical(indicators)
        db.upsert_candidate(
            run_id,
            symbol,
            stage="stage2",
            technical_score=result["subtotal"],
            total_score=result["subtotal"],
        )
        scored.append(
            {
                "symbol": symbol,
                "technical_score": result["subtotal"],
                "breakdown": result["breakdown"],
                "indicators": indicators,
            }
        )

    scored.sort(key=lambda x: x["technical_score"], reverse=True)
    top = scored[: config.TOP_N_TECHNICAL]
    for rank, item in enumerate(top, start=1):
        db.upsert_candidate(
            run_id,
            item["symbol"],
            stage="stage2_top",
            technical_score=item["technical_score"],
            total_score=item["technical_score"],
            rank=rank,
        )
    logger.info("Stage2: %d件中、テクニカルスコア上位%d件を選出", len(scored), len(top))
    return top
