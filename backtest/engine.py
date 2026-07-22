"""過去データを使ったテクニカルルールのバックテストエンジン。

本番パイプラインが使うAI分析(Claude)や、当日の「値動き上位ランキング」による銘柄選定は
過去の特定日について無料で再現できないため、テクニカル条件だけで「もしこのルールで毎日
売買していたら」を検証する簡易バックテストである。

基本ルール:
- 各銘柄・各営業日について、その日までのデータでテクニカルスコア(score_technical)を計算
- スコアが min_technical_score 以上ならシグナル発生
- entry_mode="close"(旧・現行方式): シグナル当日の終値でそのままエントリー(先読みになる簡易版)
  entry_mode="next_open"(現実的方式): 翌営業日の始値でエントリー(実際に売買可能なタイミング)
- スリッページ・手数料を指定すると、エントリー/エグジットの約定価格に不利な方向で反映する
- ATRベースの損切り/利確(config.STOP_LOSS_ATR_MULTIPLIER / TAKE_PROFIT_2_TARGET_RETURN、
  本番パイプラインのstage3と同じ式)に、保有期間中の高値・安値が触れた時点で手仕舞い
- max_holding_days以内にどちらにも触れなければ、その時点の終値で強制手仕舞い
- 1銘柄につき同時に1ポジションまで(トレード終了後から再スキャンする)
- excluded_conditions を指定すると、その条件の配点を0にしてシグナル判定する(アブレーション用)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

import config
from analysis.scoring import score_technical
from analysis.technical import compute_indicator_frame, row_to_indicators
from api.yfinance_client import fetch_historical_prices

logger = logging.getLogger(__name__)


def price_trade_at_signal(
    symbol: str,
    df: pd.DataFrame,
    signal_idx: int,
    indicators: dict,
    signal_score: float,
    max_holding_days: int,
    entry_mode: str = "close",
    slippage_pct: float = 0.0,
    commission_pct: float = 0.0,
    same_day_priority: str = "stop_first",
) -> Optional[dict]:
    """既に判定済みのシグナル(signal_idx)1件について、指定したコスト/エントリー方式で
    約定価格・損益を計算する。固定したシグナル集合を、異なるコスト条件で価格付けし直す
    (項目3の「固定トレード集合比較」)用に、simulate_symbolの内部ロジックを切り出したもの。

    シグナル自体の判定(スコアがしきい値以上か)はこの関数の外で既に行われている前提。
    """
    n = len(df)
    slip = slippage_pct / 100
    comm = commission_pct / 100
    cost_buy = 1 + slip + comm
    cost_sell = 1 - slip - comm

    if entry_mode == "next_open":
        entry_idx = signal_idx + 1
        if entry_idx >= n:
            return None  # 翌営業日のデータが無いので取引できない
        entry_price_raw = float(df.iloc[entry_idx]["open"])
        scan_start = entry_idx
        limit = min(entry_idx + max_holding_days - 1, n - 1)
    else:
        entry_idx = signal_idx
        entry_price_raw = indicators["current_close"]
        scan_start = entry_idx + 1
        limit = min(entry_idx + max_holding_days, n - 1)

    entry_price = entry_price_raw * cost_buy

    atr = indicators.get("atr14") or 0
    if atr <= 0:
        return None

    stop_loss = entry_price - config.STOP_LOSS_ATR_MULTIPLIER * atr
    take_profit = entry_price * (1 + config.TAKE_PROFIT_2_TARGET_RETURN)

    exit_idx: Optional[int] = None
    exit_price_raw: Optional[float] = None
    outcome: Optional[str] = None

    j = scan_start
    while j <= limit:
        day = df.iloc[j]
        hit_stop = day["low"] <= stop_loss
        hit_target = day["high"] >= take_profit

        if hit_stop and hit_target:
            if same_day_priority == "target_first":
                exit_idx, exit_price_raw, outcome = j, take_profit, "win"
            elif same_day_priority == "nearest_open":
                day_open = float(day["open"])
                if abs(day_open - stop_loss) <= abs(take_profit - day_open):
                    exit_idx, exit_price_raw, outcome = j, stop_loss, "loss"
                else:
                    exit_idx, exit_price_raw, outcome = j, take_profit, "win"
            else:
                exit_idx, exit_price_raw, outcome = j, stop_loss, "loss"
            break
        if hit_stop:
            exit_idx, exit_price_raw, outcome = j, stop_loss, "loss"
            break
        if hit_target:
            exit_idx, exit_price_raw, outcome = j, take_profit, "win"
            break
        j += 1

    if exit_idx is None:
        exit_idx = limit
        exit_price_raw = float(df.iloc[limit]["close"])
        outcome = "win" if exit_price_raw > entry_price else "loss"

    exit_price = exit_price_raw * cost_sell
    exit_row = df.iloc[exit_idx]
    holding_days = (exit_idx - entry_idx + 1) if entry_mode == "next_open" else (exit_idx - entry_idx)
    pnl_pct = (exit_price - entry_price) / entry_price * 100

    return {
        "symbol": symbol,
        "signal_date": indicators["date"],
        "signal_score": round(signal_score, 1),
        "signal_volume_ratio": indicators.get("volume_ratio"),
        "signal_breakout_pct": indicators.get("breakout_pct"),
        "entry_date": df.iloc[entry_idx]["date"].strftime("%Y-%m-%d"),
        "entry_price": round(entry_price, 4),
        "stop_loss": round(stop_loss, 4),
        "risk_per_share": round(entry_price - stop_loss, 4),
        "exit_date": exit_row["date"].strftime("%Y-%m-%d"),
        "exit_price": round(float(exit_price), 4),
        "pnl_pct": round(pnl_pct, 4),
        "holding_days": int(holding_days),
        "outcome": outcome,
    }


def simulate_symbol(
    symbol: str,
    price_history: list[dict],
    min_technical_score: float,
    max_holding_days: int,
    min_rows: int = 30,
    entry_mode: str = "close",
    slippage_pct: float = 0.0,
    commission_pct: float = 0.0,
    excluded_conditions: Optional[set] = None,
    same_day_priority: str = "stop_first",
) -> list[dict]:
    """1銘柄分のシミュレーションを行い、トレード記録のリストを返す。

    same_day_priority: 同じ日に損切り価格・利確価格の両方に触れた場合にどちらを採用するか。
    "stop_first"(既定, 保守的): 損切りを優先。
    "target_first": 利確を優先。
    "nearest_open": その日の始値から近い方(先に到達したと仮定)を採用。
    """
    df = compute_indicator_frame(price_history, min_rows=min_rows)
    if df is None:
        return []

    trades: list[dict] = []
    n = len(df)
    i = min_rows - 1

    while i < n:
        indicators = row_to_indicators(df, i)
        tech_result = score_technical(indicators, excluded_conditions=excluded_conditions)
        signal_score = tech_result["subtotal"]

        if signal_score < min_technical_score:
            i += 1
            continue

        trade = price_trade_at_signal(
            symbol, df, i, indicators, signal_score, max_holding_days,
            entry_mode=entry_mode, slippage_pct=slippage_pct, commission_pct=commission_pct,
            same_day_priority=same_day_priority,
        )
        if trade is None:
            if entry_mode == "next_open" and i + 1 >= n:
                break  # データ末尾に達したので打ち切り
            i += 1
            continue

        trades.append(trade)
        # entry_idxはtrade["entry_date"]から逆算せず、close方式ならi、next_openならi+1
        entry_idx = i + 1 if entry_mode == "next_open" else i
        exit_idx = entry_idx + (trade["holding_days"] - 1 if entry_mode == "next_open" else trade["holding_days"])
        i = exit_idx + 1  # トレード終了後から再スキャンする(同時保有は1銘柄1ポジションまで)

    return trades


def run_backtest(
    symbols: list[str],
    years: int,
    min_technical_score: float,
    max_holding_days: int,
    entry_mode: str = "close",
    slippage_pct: float = 0.0,
    commission_pct: float = 0.0,
    excluded_conditions: Optional[set] = None,
    same_day_priority: str = "stop_first",
) -> list[dict]:
    """全銘柄分のトレード記録をまとめて日付順に返す。"""
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=365 * years + 60)).isoformat()

    all_trades: list[dict] = []
    for symbol in symbols:
        logger.info("バックテスト中: %s", symbol)
        history = fetch_historical_prices(symbol, from_date, to_date)
        if not history:
            logger.warning("価格データが取得できませんでした: %s", symbol)
            continue
        trades = simulate_symbol(
            symbol, history, min_technical_score, max_holding_days,
            entry_mode=entry_mode, slippage_pct=slippage_pct,
            commission_pct=commission_pct, excluded_conditions=excluded_conditions,
            same_day_priority=same_day_priority,
        )
        logger.info("%s: %d件のトレード", symbol, len(trades))
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t["entry_date"])
    return all_trades
