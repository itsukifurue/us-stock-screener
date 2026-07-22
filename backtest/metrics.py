"""バックテスト結果(トレード一覧)から統計指標を計算する。"""
from __future__ import annotations

import math
from collections import defaultdict


def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_gain_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": None,
            "max_drawdown_pct": 0.0,
            "avg_holding_days": 0.0,
            "sharpe_ratio": 0.0,
            "expected_value_pct": 0.0,
        }

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    num_trades = len(trades)
    win_rate = len(wins) / num_trades * 100
    avg_gain_pct = sum(wins) / len(wins) if wins else 0.0
    avg_loss_pct = sum(losses) / len(losses) if losses else 0.0  # 負の値(または0)

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None  # 損失ゼロで無限大になるため「計算不能」として扱う
    else:
        profit_factor = 0.0

    avg_holding_days = sum(t["holding_days"] for t in trades) / num_trades

    # 最大ドローダウン: 複数銘柄のトレードをまたいで1口座に100%ずつ複利で流し込むと、
    # 実際には同時保有(分散投資)されるはずのトレードが直列につながり非現実的な暴落を生む
    # (実際に試したところ最大ドローダウンが100%を超える異常値になった)。
    # このエンジンは1銘柄内では同時に1ポジションまでしか持たない設計になっており、
    # 銘柄内の時系列は本当に直列なので、銘柄ごとに複利の資産曲線を計算し、
    # その中で最も悪かったドローダウンを採用する(「その銘柄に集中投資していたら」の
    # 最悪ケースの目安であり、分散投資を前提にした真のポートフォリオ・ドローダウンではない)。
    by_symbol: dict[str, list[float]] = defaultdict(list)
    for t in sorted(trades, key=lambda t: t["entry_date"]):
        by_symbol[t["symbol"]].append(t["pnl_pct"])

    max_drawdown_pct = 0.0
    for symbol_pnls in by_symbol.values():
        equity = 1.0
        peak = 1.0
        for p in symbol_pnls:
            equity *= 1 + p / 100
            peak = max(peak, equity)
            drawdown = (peak - equity) / peak * 100
            max_drawdown_pct = max(max_drawdown_pct, drawdown)

    mean_pnl = sum(pnls) / num_trades
    variance = sum((p - mean_pnl) ** 2 for p in pnls) / num_trades
    std_pnl = math.sqrt(variance)
    # トレード単位の簡易Sharpe Ratio(日次リターン系列ではなくトレードリターン系列を使う近似値)
    sharpe_ratio = (mean_pnl / std_pnl * math.sqrt(num_trades)) if std_pnl > 0 else 0.0

    expected_value_pct = (win_rate / 100) * avg_gain_pct + (1 - win_rate / 100) * avg_loss_pct

    return {
        "num_trades": num_trades,
        "win_rate": round(win_rate, 2),
        "avg_gain_pct": round(avg_gain_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "avg_holding_days": round(avg_holding_days, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "expected_value_pct": round(expected_value_pct, 2),
    }
