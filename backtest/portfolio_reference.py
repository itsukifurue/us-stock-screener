"""backtest.portfolio.simulate_portfolio の独立検証用リファレンス実装。

意図的に backtest/portfolio.py とは異なるデータ構造・書き方で実装し、
同じ入力(トレード一覧・価格データ)に対して同じNAV系列が得られるかを
クロスチェックするための「別実装」である(バグの検出には同一ロジックの
コピーではなく、独立した実装との突き合わせが有効なため)。

会計モデルは backtest/portfolio.py と同一の前提(現金+保有株数×終値、
レバレッジなし、手数料は売買代金に対して1回ずつ、同日中の決済→新規エントリー
の順で処理)に合わせているが、内部実装は下記の通り異なる:
- スロット配列ではなく「保有中ポジションの辞書(symbol -> Position)」で管理
- 終値の参照は手書きの辞書ではなく pandas の pivot_table を使う
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import config


@dataclass
class Position:
    symbol: str
    shares: float
    exit_date: str
    exit_price: float


def simulate_portfolio_reference(
    trades: list[dict],
    price_frames: dict[str, pd.DataFrame],
    max_concurrent: int = 3,
    initial_capital: float = 1.0,
    commission_pct: float | None = None,
) -> dict:
    if commission_pct is None:
        commission_pct = config.BACKTEST_COMMISSION_PCT
    comm = commission_pct / 100

    if not trades:
        return {"nav_series": [initial_capital], "dates": []}

    # pandasのpivotで「日付 x 銘柄」の終値テーブルを作る(portfolio.pyの手書き辞書とは別方式)
    frames = []
    for sym, df in price_frames.items():
        tmp = df[["date", "close"]].copy()
        tmp["symbol"] = sym
        frames.append(tmp)
    long_df = pd.concat(frames, ignore_index=True)
    close_pivot = long_df.pivot_table(index="date", columns="symbol", values="close")
    close_pivot = close_pivot.sort_index()
    all_dates = [d.strftime("%Y-%m-%d") for d in close_pivot.index]

    trades_by_entry: dict[str, list[dict]] = {}
    for t in trades:
        trades_by_entry.setdefault(t["entry_date"], []).append(t)
    for lst in trades_by_entry.values():
        lst.sort(key=lambda t: -t["signal_score"])

    cash = initial_capital
    open_positions: dict[str, Position] = {}  # symbol -> Position
    nav_history: list[float] = []

    for today in all_dates:
        # ① 決済(本日がexit_dateのポジションを全てクローズ)
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            if pos.exit_date == today:
                proceeds = pos.shares * pos.exit_price
                fee = proceeds * comm
                cash += proceeds - fee
                del open_positions[sym]

        # ② 新規エントリー(空き枠がある限り、スコア順に割り当てる)
        for t in trades_by_entry.get(today, []):
            if len(open_positions) >= max_concurrent:
                continue
            if t["symbol"] in open_positions:
                continue  # 同一銘柄への重複ポジションは持たない

            current_value = 0.0
            for sym, pos in open_positions.items():
                price = close_pivot.at[pd.Timestamp(today), sym] if sym in close_pivot.columns else pos.exit_price
                if pd.isna(price):
                    price = pos.exit_price
                current_value += pos.shares * price
            nav_now = cash + current_value
            target = nav_now / max_concurrent
            invest_amount = min(cash, target)
            if invest_amount <= 0:
                continue

            entry_price = t["entry_price"]
            shares = invest_amount / entry_price
            fee = invest_amount * comm
            cash -= invest_amount
            cash -= fee
            open_positions[t["symbol"]] = Position(t["symbol"], shares, t["exit_date"], t["exit_price"])

            # entry_date == exit_date(同日決済)の場合、①のエグジット処理は既に通過済みなので
            # ここで即座に決済しないとポジションが永久に残ってしまう。
            if t["exit_date"] == today:
                pos = open_positions.pop(t["symbol"])
                proceeds = pos.shares * pos.exit_price
                fee2 = proceeds * comm
                cash += proceeds - fee2

        # ③ 本日のNAVを記録
        total_value = 0.0
        for sym, pos in open_positions.items():
            price = close_pivot.at[pd.Timestamp(today), sym] if sym in close_pivot.columns else pos.exit_price
            if pd.isna(price):
                price = pos.exit_price
            total_value += pos.shares * price
        nav_history.append(cash + total_value)

    return {"nav_series": [initial_capital] + nav_history, "dates": all_dates}
