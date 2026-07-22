"""Stage1: 候補銘柄(100〜200銘柄程度)を抽出する。

当初はCompany Screener APIを使う設計だったが、実際のAPIキーで疎通確認したところ
無料プランでは company-screener・stock-list ともに402(Restricted Endpoint)で使えないことが
判明した(2026-07-21確認)。代わりに無料プランで動作確認済みの most-actives(出来高上位)と
biggest-gainers(値上がり上位)を候補生成の入口として使う。これらは「動きのある銘柄」を
拾うランキングAPIで、市場全体を条件フィルタする代わりに使う。

これらのレスポンスには出来高(volume)や時価総額(marketCap)が含まれないため、
・株価(price)によるフィルタだけをここで行い、
・出来高フィルタはstage2でヒストリカルデータから計算したavg_volume20を使い、
・時価総額/ETF除外/国フィルタはstage3でプロフィール取得時に行う
という形で、要件定義書の一次スクリーニング条件をパイプライン全体に分散して適用する。
"""
from __future__ import annotations

import logging

import config
from api.fmp_client import FMPClient
from db.database import Database

logger = logging.getLogger(__name__)


def run_stage1(db: Database, client: FMPClient, run_id: int) -> list[str]:
    """一次スクリーニングを実行し、候補銘柄のシンボル一覧を返す。stocks/candidatesテーブルも更新する。"""
    most_active = client.most_actives()
    gainers = client.biggest_gainers()
    logger.info("Stage1: most-actives=%d件, biggest-gainers=%d件を取得", len(most_active), len(gainers))

    merged: dict[str, dict] = {}
    for item in most_active + gainers:
        symbol = item.get("symbol")
        if not symbol:
            continue
        merged.setdefault(symbol, item)

    symbols: list[str] = []
    for symbol, item in merged.items():
        price = item.get("price")
        if price is not None and price < config.SCREEN_MIN_PRICE:
            continue  # 株価5ドル未満は一次除外(要件の株価フィルタ)

        db.upsert_stock(
            {
                "symbol": symbol,
                "name": item.get("name"),
                "exchange": item.get("exchange"),
                "sector": None,
                "industry": None,
                "country": None,
                "is_etf": False,  # ここでは判定不能。stage3のプロフィール取得時に判定する
                "market_cap": None,
            }
        )
        db.upsert_candidate(run_id, symbol, stage="stage1")
        symbols.append(symbol)

    logger.info("Stage1: 価格フィルタ後の候補は%d件", len(symbols))
    return symbols[: config.SCREENER_CANDIDATE_LIMIT]
