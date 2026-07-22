"""複数銘柄のトレードを、同時保有数の上限つきでポートフォリオとしてシミュレーションする。

【会計モデル(明示)】
- 総資産(NAV) = 現金残高 + Σ(保有株数 × 当日終値)  ※日次でこの式のみから算出する
- 新規エントリー時の投資額 = min(現金残高, その時点のNAV / max_concurrent)
  → レバレッジ・信用取引は一切使わない。現金が不足する場合は現金の範囲内でのみ投資し、
    投資額がその時点のNAVを超えることはない。
- 手数料は「売買代金 × BACKTEST_COMMISSION_PCT」を、エントリー時とエグジット時にそれぞれ
  1回ずつ現金から控除する(日次の保有中は控除しない)。
- スリッページは約定価格そのものに埋め込む(backtest.engine側で買いは高く・売りは安く反映済み)。
  ポートフォリオ側では約定価格をそのまま使うだけで、スリッページを二重に反映することはない。
- 1日の処理順序: ①本日エグジットするポジションを決済して現金化 → ②本日エントリーする
  シグナルを空きスロットに割り当て → ③日次NAVを記録。
  そのため「本日決済されたスロット」は同日中に新規エントリーへ再利用され得る
  (証券会社の受渡日(T+2等)は考慮しない簡易モデル)。
- 1銘柄について同時に複数ポジションを持つことはない(backtest.engine側の制約による)。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd

import config


def simulate_portfolio(
    trades: list[dict],
    price_frames: dict[str, pd.DataFrame],
    max_concurrent: int = 3,
    initial_capital: float = 1.0,
    commission_pct: Optional[float] = None,
) -> dict:
    """最大 max_concurrent 銘柄までの同時保有を許容したポートフォリオシミュレーション。

    trades: backtest.engine.simulate_symbol の出力(entry_price/exit_priceは既にスリッページ
            込みの約定価格)を全銘柄分まとめたリスト。
    price_frames: {symbol: date/close列を持つDataFrame}(日次mark-to-market用)。
    commission_pct: Noneの場合 config.BACKTEST_COMMISSION_PCT を使う。

    戻り値: {"nav_series", "cash_series", "dates", "trades_taken", "trades_skipped",
             "num_years", "log", "per_symbol_taken"}
    """
    if commission_pct is None:
        commission_pct = config.BACKTEST_COMMISSION_PCT
    comm = commission_pct / 100

    if not trades:
        return {
            "nav_series": [initial_capital], "cash_series": [initial_capital], "dates": [],
            "trades_taken": 0, "trades_skipped": 0, "num_years": 0.0, "log": [], "per_symbol_taken": {},
        }

    trades_by_entry_date: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        trades_by_entry_date[t["entry_date"]].append(t)
    for day_trades in trades_by_entry_date.values():
        day_trades.sort(key=lambda t: -t["signal_score"])  # 枠が足りない時はスコアが高い方を優先

    all_dates = sorted({d.strftime("%Y-%m-%d") for df in price_frames.values() for d in df["date"]})

    close_lookup: dict[str, dict[str, float]] = {
        sym: dict(zip(df["date"].dt.strftime("%Y-%m-%d"), df["close"].astype(float)))
        for sym, df in price_frames.items()
    }

    cash = initial_capital
    slots: list[Optional[dict]] = [None] * max_concurrent  # 各要素: {"symbol","shares","trade","last_price"}

    taken = 0
    skipped = 0
    skipped_no_slot = 0
    skipped_cash = 0
    max_concurrent_used = 0
    per_symbol_taken: dict[str, int] = defaultdict(int)
    nav_history: list[float] = []
    cash_history: list[float] = []
    date_history: list[str] = []
    utilization_history: list[float] = []
    log: list[dict] = []

    def positions_value(today: str) -> float:
        total = 0.0
        for slot in slots:
            if slot is None:
                continue
            price = close_lookup.get(slot["symbol"], {}).get(today, slot["last_price"])
            slot["last_price"] = price
            total += slot["shares"] * price
        return total

    def close_slot(idx: int, today: str) -> None:
        nonlocal cash
        slot = slots[idx]
        exit_price = slot["trade"]["exit_price"]  # スリッページ込みの約定価格(engine側で反映済み)
        proceeds = slot["shares"] * exit_price
        fee = proceeds * comm
        cash += proceeds - fee
        log.append(
            {"date": today, "action": "exit", "symbol": slot["symbol"],
             "shares": slot["shares"], "exit_price": exit_price,
             "proceeds": proceeds, "commission_fee": fee, "cash_after": cash}
        )
        slots[idx] = None

    for today in all_dates:
        # ① 本日エグジットするポジションを決済する
        for idx, slot in enumerate(slots):
            if slot is not None and slot["trade"]["exit_date"] == today:
                close_slot(idx, today)

        # ② 本日エントリーするシグナルを空きスロットに割り当てる
        for t in trades_by_entry_date.get(today, []):
            free_slot = next((idx for idx in range(max_concurrent) if slots[idx] is None), None)
            if free_slot is None:
                skipped += 1
                skipped_no_slot += 1
                continue

            nav_now = cash + positions_value(today)
            target_size = nav_now / max_concurrent
            invest_amount = min(cash, target_size)  # レバレッジなし: 現金の範囲内でのみ投資
            if invest_amount <= 0:
                skipped += 1
                skipped_cash += 1
                continue

            entry_price = t["entry_price"]  # スリッページ込みの約定価格(engine側で反映済み)
            shares = invest_amount / entry_price
            fee = invest_amount * comm
            cash -= invest_amount
            cash -= fee

            slots[free_slot] = {"symbol": t["symbol"], "shares": shares, "trade": t, "last_price": entry_price}
            taken += 1
            per_symbol_taken[t["symbol"]] += 1
            log.append(
                {"date": today, "action": "entry", "symbol": t["symbol"],
                 "shares": shares, "entry_price": entry_price,
                 "invest_amount": invest_amount, "commission_fee": fee, "cash_after": cash}
            )

            # entry_date == exit_date(シグナル当日中に損切り/利確が確定する同日決済)の場合、
            # ①のエグジット処理は既にこの日は通過済みなので、ここで即座に決済しないと
            # スロットが永久に埋まったままになってしまう(実際に発生し重大なバグだった)。
            if t["exit_date"] == today:
                close_slot(free_slot, today)

        # ③ 本日時点のNAVを記録する(NAV = 現金 + Σ 保有株数×当日終値)
        pv = positions_value(today)
        nav_today = cash + pv
        nav_history.append(nav_today)
        cash_history.append(cash)
        date_history.append(today)
        utilization_history.append((pv / nav_today) if nav_today > 0 else 0.0)
        occupied = sum(1 for s in slots if s is not None)
        max_concurrent_used = max(max_concurrent_used, occupied)

    if date_history:
        start = datetime.strptime(date_history[0], "%Y-%m-%d").date()
        end = datetime.strptime(date_history[-1], "%Y-%m-%d").date()
        num_years = max((end - start).days / 365.25, 1 / 365.25)
    else:
        num_years = 0.0

    avg_utilization = sum(utilization_history) / len(utilization_history) if utilization_history else 0.0

    return {
        "nav_series": [initial_capital] + nav_history,
        "cash_series": [initial_capital] + cash_history,
        "dates": date_history,
        "trades_taken": taken,
        "trades_skipped": skipped,
        "trades_skipped_no_slot": skipped_no_slot,
        "trades_skipped_cash": skipped_cash,
        "max_concurrent_used": max_concurrent_used,
        "avg_capital_utilization_pct": round(avg_utilization * 100, 1),
        "num_years": num_years,
        "log": log,
        "per_symbol_taken": dict(per_symbol_taken),
    }
