"""バックテスト結果(トレード一覧)から統計指標を計算する。"""
from __future__ import annotations

import math
from collections import defaultdict


def compute_detailed_stats(trades: list[dict], weight: float = 1.0) -> dict:
    """トレード一覧から詳細統計を計算する。

    weight: 各トレードに割り当てる資金比率(例: 1/3集中投資なら1/3)。1.0なら
    「各トレードに全額投資した場合の単純なリターン」として扱う。
    """
    if not trades:
        return {
            "num_trades": 0, "win_rate": 0.0, "profit_factor": None,
            "arithmetic_mean_pct": 0.0, "median_pct": 0.0, "geometric_mean_pct": 0.0,
            "std_pct": 0.0, "expected_value_pct": 0.0, "avg_gain_pct": 0.0, "avg_loss_pct": 0.0,
            "max_gain_pct": 0.0, "max_loss_pct": 0.0,
        }

    returns = [(t["pnl_pct"] / 100) * weight for t in trades]  # 比率(例: 0.05 = +5%)
    pnl_pcts = [r * 100 for r in returns]

    n = len(returns)
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = len(wins) / n * 100

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    arithmetic_mean = sum(returns) / n
    sorted_r = sorted(returns)
    median = sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2

    # 幾何平均: log(1+r)の平均をexpで戻す(1+r<=0の異常値は評価対象から除く)
    log_terms = [math.log(1 + r) for r in returns if 1 + r > 0]
    geometric_mean = (math.exp(sum(log_terms) / len(log_terms)) - 1) if log_terms else float("nan")

    variance = sum((r - arithmetic_mean) ** 2 for r in returns) / n
    std = math.sqrt(variance)

    avg_gain = (sum(wins) / len(wins) * 100) if wins else 0.0
    avg_loss = (sum(losses) / len(losses) * 100) if losses else 0.0
    expected_value = (win_rate / 100) * avg_gain + (1 - win_rate / 100) * avg_loss

    return {
        "num_trades": n,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "arithmetic_mean_pct": round(arithmetic_mean * 100, 4),
        "median_pct": round(median * 100, 4),
        "geometric_mean_pct": round(geometric_mean * 100, 4) if not math.isnan(geometric_mean) else None,
        "std_pct": round(std * 100, 4),
        "expected_value_pct": round(expected_value, 4),
        "avg_gain_pct": round(avg_gain, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "max_gain_pct": round(max(pnl_pcts), 4),
        "max_loss_pct": round(min(pnl_pcts), 4),
    }


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


def compute_concentration(trades: list[dict], top_n: int = 5) -> dict:
    """利益が上位何銘柄にどれだけ集中しているかを計算する。"""
    by_symbol: dict[str, float] = defaultdict(float)
    for t in trades:
        by_symbol[t["symbol"]] += t["pnl_pct"]

    ranked = sorted(by_symbol.items(), key=lambda x: -x[1])
    positive_total = sum(v for v in by_symbol.values() if v > 0)
    top_symbols = ranked[:top_n]
    top_positive_sum = sum(v for _, v in top_symbols if v > 0)
    share_pct = (top_positive_sum / positive_total * 100) if positive_total > 0 else 0.0

    return {
        "top_n": top_n,
        "top_n_symbols": [s for s, _ in top_symbols],
        "top_n_profit_share_pct": round(share_pct, 1),
        "num_losing_symbols": sum(1 for v in by_symbol.values() if v < 0),
        "num_symbols": len(by_symbol),
    }


def compute_portfolio_metrics(nav_series: list[float], num_years: float, trading_days_per_year: int = 252) -> dict:
    """日次NAV(資産評価額)系列からCAGR・日次Sharpe・Sortino・最大ドローダウンを計算する。

    nav_series: 先頭を初期資金(通常1.0)とした日次の資産評価額のリスト。
    num_years: シミュレーション対象期間の実年数(カレンダーベース、CAGR計算に使用)。
    """
    if len(nav_series) < 2 or num_years <= 0:
        return {"cagr_pct": 0.0, "daily_sharpe": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0}

    returns = [(nav_series[k] / nav_series[k - 1] - 1) for k in range(1, len(nav_series)) if nav_series[k - 1] > 0]

    total_return = nav_series[-1] / nav_series[0]
    cagr_pct = ((total_return ** (1 / num_years)) - 1) * 100 if total_return > 0 else -100.0

    if returns:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std_r = math.sqrt(var_r)
        daily_sharpe = (mean_r / std_r * math.sqrt(trading_days_per_year)) if std_r > 0 else 0.0

        downside = [r for r in returns if r < 0]
        downside_var = sum(r ** 2 for r in downside) / len(returns) if downside else 0.0
        downside_std = math.sqrt(downside_var)
        sortino = (mean_r / downside_std * math.sqrt(trading_days_per_year)) if downside_std > 0 else 0.0
    else:
        daily_sharpe = 0.0
        sortino = 0.0

    peak = nav_series[0]
    max_dd = 0.0
    for v in nav_series:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak * 100)

    return {
        "cagr_pct": round(cagr_pct, 2),
        "daily_sharpe": round(daily_sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
    }
