"""point-in-time候補ユニバースの近似生成。

【重要な制約(必ずドキュメントとして残す)】
FMPの無料プランでは「過去のある日のmost-actives/biggest-gainers」を取得できない
(実キーでの検証の結果、現在時点のランキングしか取得できないことを確認済み)。
そのため過去に遡ったバックテスト/特徴量生成では、真の意味での「その日の値動き上位」を
再現することはできない。

このモジュールで行う近似方法:
  あらかじめ用意した銘柄リスト(例: backtest.watchlist.DEFAULT_WATCHLIST、または
  Phase2でより広げたリスト)の中から、各営業日ごとに、その日時点のデータだけを使って
  一次スクリーニング条件(株価5ドル以上・時価総額1億ドル以上・平均出来高50万株以上)を
  満たす銘柄を「その日の候補」とみなす。

  これは「値動きが大きい銘柄を無料で拾う」という本番パイプラインの発想とは異なり、
  「あらかじめ決め打ちした銘柄群の中から、その日の基本条件を満たすものを選ぶ」だけの
  簡易近似である。上場廃止銘柄を含められないため、生存者バイアスも残る。

  重要: 候補判定に使うのは「その日の終値・その日までの出来高」であり、これらは
  当日の取引が終わった後(引け後)にしか確定しない。したがって、この候補情報を使った
  バックテスト上の最短エントリーは「翌営業日の始値」でなければならず、当日の寄り付きや
  日中にエントリーしてはならない(backtest.engineのentry_mode="next_open"と整合させる)。

candidate_source は "approx_universe" として記録し、本番の "most_actives"/"biggest_gainers"
とは区別できるようにする。ユニバース条件自体もバージョン管理する(UNIVERSE_VERSION)。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

APPROX_SOURCE_LABEL = "approx_universe"
UNIVERSE_VERSION = "universe_v2_1"  # 条件を変更したらこのバージョン文字列も更新すること


def evaluate_candidate(
    row: pd.Series, min_price: float, min_market_cap: float, min_avg_volume: float, market_cap: Optional[float]
) -> tuple[bool, str]:
    """(候補かどうか, 理由文字列) を返す。理由は非候補の場合に最初に引っかかった条件を示す。"""
    price = row.get("close")
    avg_volume = row.get("avg_volume20") if "avg_volume20" in row else row.get("avg_volume_20d")

    if price is None or pd.isna(price):
        return False, "price_unavailable"
    if price < min_price:
        return False, f"price<{min_price}"
    if avg_volume is None or pd.isna(avg_volume):
        return False, "avg_volume_unavailable"
    if avg_volume < min_avg_volume:
        return False, f"avg_volume<{min_avg_volume}"
    if market_cap is not None and market_cap < min_market_cap:
        return False, f"market_cap<{min_market_cap}"
    return True, "ok"


def is_candidate_on_date(row: pd.Series, min_price: float, min_market_cap: float, min_avg_volume: float, market_cap: Optional[float]) -> bool:
    """後方互換用(bool のみ返す)。理由も欲しい場合は evaluate_candidate を使う。"""
    ok, _ = evaluate_candidate(row, min_price, min_market_cap, min_avg_volume, market_cap)
    return ok
