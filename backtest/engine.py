"""過去データを使ったテクニカルルールのバックテストエンジン。

本番パイプラインが使うAI分析(Claude)や、当日の「値動き上位ランキング」による銘柄選定は
過去の特定日について無料で再現できないため、テクニカル条件だけで「もしこのルールで毎日
売買していたら」を検証する簡易バックテストである。

ルール:
- 各銘柄・各営業日について、その日までのデータでテクニカルスコア(score_technical)を計算
- スコアが min_technical_score 以上ならその日の終値でエントリー
- ATRベースの損切り/利確(config.STOP_LOSS_ATR_MULTIPLIER / TAKE_PROFIT_2_TARGET_RETURN、
  本番パイプラインのstage3と同じ式)に、翌営業日以降の高値・安値が触れた時点で手仕舞い
- max_holding_days以内にどちらにも触れなければ、その時点の終値で強制手仕舞い
- 1銘柄につき同時に1ポジションまで(トレード終了後から再スキャンする)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import config
from analysis.scoring import score_technical
from analysis.technical import compute_indicator_frame, row_to_indicators
from api.yfinance_client import fetch_historical_prices

logger = logging.getLogger(__name__)


def simulate_symbol(
    symbol: str,
    price_history: list[dict],
    min_technical_score: float,
    max_holding_days: int,
    min_rows: int = 30,
) -> list[dict]:
    """1銘柄分のシミュレーションを行い、トレード記録のリストを返す。"""
    df = compute_indicator_frame(price_history, min_rows=min_rows)
    if df is None:
        return []

    trades: list[dict] = []
    n = len(df)
    i = min_rows - 1

    while i < n:
        indicators = row_to_indicators(df, i)
        tech_result = score_technical(indicators)

        if tech_result["subtotal"] < min_technical_score:
            i += 1
            continue

        entry_price = indicators["current_close"]
        atr = indicators.get("atr14") or 0
        if atr <= 0:
            i += 1
            continue

        stop_loss = entry_price - config.STOP_LOSS_ATR_MULTIPLIER * atr
        take_profit = entry_price * (1 + config.TAKE_PROFIT_2_TARGET_RETURN)

        limit = min(i + max_holding_days, n - 1)
        exit_idx: Optional[int] = None
        exit_price: Optional[float] = None
        outcome: Optional[str] = None

        j = i + 1
        while j <= limit:
            day = df.iloc[j]
            if day["low"] <= stop_loss:
                exit_idx, exit_price, outcome = j, stop_loss, "loss"
                break
            if day["high"] >= take_profit:
                exit_idx, exit_price, outcome = j, take_profit, "win"
                break
            j += 1

        if exit_idx is None:
            # 保有期限まで損切り/利確どちらにも触れなかった場合、期限最終日の終値で強制手仕舞い
            exit_idx = limit
            exit_price = float(df.iloc[limit]["close"])
            outcome = "win" if exit_price > entry_price else "loss"

        exit_row = df.iloc[exit_idx]
        holding_days = exit_idx - i
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        trades.append(
            {
                "symbol": symbol,
                "entry_date": indicators["date"],
                "entry_price": round(entry_price, 4),
                "exit_date": exit_row["date"].strftime("%Y-%m-%d"),
                "exit_price": round(float(exit_price), 4),
                "pnl_pct": round(pnl_pct, 4),
                "holding_days": int(holding_days),
                "outcome": outcome,
            }
        )
        i = exit_idx + 1  # トレード終了後から再スキャンする(同時保有は1銘柄1ポジションまで)

    return trades


def run_backtest(
    symbols: list[str],
    years: int,
    min_technical_score: float,
    max_holding_days: int,
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
        trades = simulate_symbol(symbol, history, min_technical_score, max_holding_days)
        logger.info("%s: %d件のトレード", symbol, len(trades))
        all_trades.extend(trades)

    all_trades.sort(key=lambda t: t["entry_date"])
    return all_trades
