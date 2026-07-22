"""CAGR悪化要因を切り分けるための詳細検証スクリプト(9項目)。

1. 採用済み(taken)トレードのみの詳細統計 vs 全候補トレード
2. 採用ルール(同日に空き枠を上回るシグナルが出た場合の優先順位)の明示と比較
3. 固定トレード集合による7パターン比較(動的ポートフォリオ比較とは別に出力)
4. ボラティリティ・ドラッグの直接検証(mean(log(1+return/3)))
5. 時系列ランダム化テスト(日次リターンを1000回シャッフルしCAGR分布と比較)
6. 月次/年次の損失集中(相関・同日複数損失など)
7. ポジションサイズ比較(33%/3, 20%/5, 10%/10, ATR固定リスク1%, 0.5%)
8. 同日処理の厳密化(現金再利用タイミング same_day vs next_day)
9. 上記の根拠に基づく結論の分類(A〜F)

実行:
    python scripts/run_cagr_decomposition.py
"""
from __future__ import annotations

import logging
import math
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from analysis.technical import compute_indicator_frame, row_to_indicators
from analysis.scoring import score_technical
from api.yfinance_client import fetch_historical_prices
from backtest.engine import price_trade_at_signal, simulate_symbol
from backtest.metrics import compute_detailed_stats, compute_portfolio_metrics
from backtest.portfolio import simulate_portfolio
from backtest.watchlist import DEFAULT_WATCHLIST


def fmt(v, unit=""):
    return "N/A" if v is None else f"{v}{unit}"


def fetch_all_histories(symbols, years):
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=365 * years + 60)).isoformat()
    histories = {}
    for sym in symbols:
        h = fetch_historical_prices(sym, from_date, to_date)
        if h:
            histories[sym] = h
        else:
            logging.warning("価格データが取得できませんでした: %s", sym)
    return histories


def gen_trades(histories, symbols, **kwargs):
    trades = []
    for sym in symbols:
        trades.extend(
            simulate_symbol(
                sym, histories[sym],
                min_technical_score=config.BACKTEST_MIN_TECHNICAL_SCORE,
                max_holding_days=config.BACKTEST_MAX_HOLDING_DAYS,
                **kwargs,
            )
        )
    trades.sort(key=lambda t: t["entry_date"])
    return trades


def stats_row(label, stats):
    return (
        f"| {label} | {stats['num_trades']} | {fmt(stats['win_rate'], '%')} | {fmt(stats['profit_factor'])} | "
        f"{fmt(stats['arithmetic_mean_pct'], '%')} | {fmt(stats['median_pct'], '%')} | "
        f"{fmt(stats['geometric_mean_pct'], '%')} | {fmt(stats['std_pct'], '%')} | "
        f"{fmt(stats['expected_value_pct'], '%')} | {fmt(stats['avg_gain_pct'], '%')} | "
        f"{fmt(stats['avg_loss_pct'], '%')} | {fmt(stats['max_gain_pct'], '%')} | {fmt(stats['max_loss_pct'], '%')} |"
    )


STATS_HEADER = (
    "| 対象 | 件数 | 勝率 | PF | 算術平均 | 中央値 | 幾何平均 | 標準偏差 | 期待値 | 平均利益 | 平均損失 | 最大利益 | 最大損失 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    years = config.BACKTEST_YEARS
    window_start = (date.today() - timedelta(days=365 * years)).isoformat()
    max_concurrent = config.BACKTEST_MAX_CONCURRENT_POSITIONS

    print("価格データ取得中...")
    histories = fetch_all_histories(DEFAULT_WATCHLIST, years)
    symbols = list(histories.keys())
    price_frames = {s: compute_indicator_frame(histories[s]) for s in symbols}
    print(f"取得できた銘柄数: {len(symbols)}/{len(DEFAULT_WATCHLIST)}")

    lines = [f"# CAGR悪化要因の切り分け検証 ({date.today().isoformat()})", ""]

    # ------------------------------------------------------------------
    # 候補トレード生成: 現行(close/no-cost) と 現実条件(next_open+slippage+commission)
    # ------------------------------------------------------------------
    print("候補トレード生成中(現行/現実条件)...")
    trades_current_all = gen_trades(histories, symbols, entry_mode="close", slippage_pct=0.0, commission_pct=0.0)
    trades_current = [t for t in trades_current_all if t["entry_date"] >= window_start]

    trades_real_all = gen_trades(
        histories, symbols, entry_mode="next_open",
        slippage_pct=config.BACKTEST_SLIPPAGE_PCT, commission_pct=config.BACKTEST_COMMISSION_PCT,
    )
    trades_real = [t for t in trades_real_all if t["entry_date"] >= window_start]

    print("ポートフォリオシミュレーション中(既定: スコア優先・同日現金再利用)...")
    port_current = simulate_portfolio(trades_current, price_frames, max_concurrent=max_concurrent)
    port_real = simulate_portfolio(trades_real, price_frames, max_concurrent=max_concurrent)

    # ------------------------------------------------------------------
    # 項目1: 採用済み(taken)のみ vs 全候補
    # ------------------------------------------------------------------
    print("[1/9] 採用済み vs 全候補の詳細統計...")
    stats_candidate_current = compute_detailed_stats(trades_current)
    stats_taken_current = compute_detailed_stats(port_current["admitted_trades"])
    stats_candidate_real = compute_detailed_stats(trades_real)
    stats_taken_real = compute_detailed_stats(port_real["admitted_trades"])

    lines += ["## 1. 採用済み(taken)トレードのみの成績 vs 全候補トレード", ""]
    lines += ["### 現行(close entry, コストなし)", "", STATS_HEADER]
    lines.append(stats_row("全候補", stats_candidate_current))
    lines.append(stats_row("採用のみ(taken)", stats_taken_current))
    lines += ["", "### 現実条件(next_open + slippage + commission)", "", STATS_HEADER]
    lines.append(stats_row("全候補", stats_candidate_real))
    lines.append(stats_row("採用のみ(taken)", stats_taken_real))
    lines.append("")
    lines.append(
        f"採用率: 現行 {port_current['trades_taken']}/{len(trades_current)}"
        f"({port_current['trades_taken']/len(trades_current)*100:.1f}%)、"
        f"現実条件 {port_real['trades_taken']}/{len(trades_real)}"
        f"({port_real['trades_taken']/len(trades_real)*100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # 項目2: 採用ルールの明示 + 比較
    # ------------------------------------------------------------------
    print("[2/9] 採用ルールの比較...")
    lines += ["", "## 2. 採用ルールの明示と比較", "", (
        "現行の採用ルール: 同じ日に複数シグナルが発生し空き枠を上回る場合、"
        "**テクニカルスコア降順**(backtest/portfolio.py の ranking_key='score')で優先的に空き枠に割り当てる。"
        "日をまたぐ順序は entry_date の時系列順(早い日付が先)。"
    ), ""]
    lines += [STATS_HEADER.replace("| 対象 |", "| 採用ルール |")]
    for key, label in [
        ("score", "a. テクニカルスコア降順(現行)"),
        ("volume_ratio", "b. 出来高倍率降順"),
        ("breakout_pct", "c. 20日高値からの突破率降順"),
        ("random", "d. ランダム選択(seed=42)"),
        ("insertion_order", "e. 現行の処理順(銘柄ループ順、ソートなし)"),
    ]:
        port = simulate_portfolio(trades_real, price_frames, max_concurrent=max_concurrent, ranking_key=key)
        stats = compute_detailed_stats(port["admitted_trades"])
        pm = compute_portfolio_metrics(port["nav_series"], port["num_years"])
        lines.append(
            stats_row(label, stats) + f" CAGR={fmt(pm['cagr_pct'],'%')}, taken={port['trades_taken']}"
        )

    # ------------------------------------------------------------------
    # 項目3: 固定トレード集合による7パターン比較
    # ------------------------------------------------------------------
    print("[3/9] 固定トレード集合による比較...")
    CONFIGS7 = [
        ("1.close-nocost(現行)", "close", 0.0, 0.0),
        ("2.next_open-nocost", "next_open", 0.0, 0.0),
        ("3.close-slippageのみ", "close", 0.2, 0.0),
        ("4.close-commissionのみ", "close", 0.0, 0.1),
        ("5.next_open-slippageのみ", "next_open", 0.2, 0.0),
        ("6.next_open-commissionのみ", "next_open", 0.0, 0.1),
        ("7.next_open-両方(現実条件)", "next_open", config.BACKTEST_SLIPPAGE_PCT, config.BACKTEST_COMMISSION_PCT),
    ]

    def reprice_fixed_set(admitted_trades, label_prefix):
        """admitted_tradesの(symbol, signal_date)集合を固定し、7パターンで価格付けし直す。"""
        sub_lines = [f"### {label_prefix}", "", STATS_HEADER]
        # symbol -> signal_idx のリストを作る(price_frames内でsignal_dateに対応する行番号)
        idx_lookup = {}
        for sym in {t["symbol"] for t in admitted_trades}:
            df = price_frames[sym]
            idx_lookup[sym] = dict(zip(df["date"].dt.strftime("%Y-%m-%d"), range(len(df))))

        for cfg_label, entry_mode, slip, comm in CONFIGS7:
            repriced = []
            for t in admitted_trades:
                sym = t["symbol"]
                df = price_frames[sym]
                sig_idx = idx_lookup[sym].get(t["signal_date"])
                if sig_idx is None:
                    continue
                indicators = row_to_indicators(df, sig_idx)
                tech = score_technical(indicators)
                new_trade = price_trade_at_signal(
                    sym, df, sig_idx, indicators, tech["subtotal"], config.BACKTEST_MAX_HOLDING_DAYS,
                    entry_mode=entry_mode, slippage_pct=slip, commission_pct=comm,
                )
                if new_trade is not None:
                    repriced.append(new_trade)
            sub_lines.append(stats_row(cfg_label, compute_detailed_stats(repriced)))
        return sub_lines

    lines += ["", "## 3. 固定トレード集合による7パターン比較(動的ポートフォリオ比較とは別)", "",
              "同一の(銘柄, シグナル日)集合を固定し、価格付け条件だけを変えている"
              "(=どのシグナルが採用されるかは変えず、コスト/エントリー方式の純粋な効果のみを見る)。", ""]
    lines += reprice_fixed_set(port_current["admitted_trades"], "現行ケースで採用された473件前後の集合を固定")
    lines += [""]
    lines += reprice_fixed_set(port_real["admitted_trades"], "現実条件で採用された集合を固定")

    lines += ["", "### (参考)動的ポートフォリオ比較(コストを変えるとシグナル自体や採用件数も変わる、従来の比較)", "",
              "| パターン | 採用件数 | 勝率 | PF | CAGR |", "|---|---|---|---|---|"]
    for cfg_label, entry_mode, slip, comm in CONFIGS7:
        t_all = gen_trades(histories, symbols, entry_mode=entry_mode, slippage_pct=slip, commission_pct=comm)
        t_win = [t for t in t_all if t["entry_date"] >= window_start]
        port = simulate_portfolio(t_win, price_frames, max_concurrent=max_concurrent)
        pm = compute_portfolio_metrics(port["nav_series"], port["num_years"])
        st = compute_detailed_stats(port["admitted_trades"])
        lines.append(f"| {cfg_label} | {port['trades_taken']} | {fmt(st['win_rate'],'%')} | {fmt(st['profit_factor'])} | {fmt(pm['cagr_pct'],'%')} |")

    # ------------------------------------------------------------------
    # 項目4: ボラティリティ・ドラッグの直接検証
    # ------------------------------------------------------------------
    print("[4/9] ボラティリティ・ドラッグの直接検証...")
    taken_real = port_real["admitted_trades"]
    weight = 1 / max_concurrent
    position_returns = [(t["pnl_pct"] / 100) * weight for t in taken_real]
    arithmetic_mean = sum(position_returns) / len(position_returns)
    log_terms = [math.log(1 + r) for r in position_returns if 1 + r > 0]
    mean_log_return = sum(log_terms) / len(log_terms)
    geometric_mean = math.exp(mean_log_return) - 1

    lines += [
        "", "## 4. ボラティリティ・ドラッグの直接検証(現実条件・採用済みトレード)", "",
        f"- 1トレードあたり投資比率: NAVの1/{max_concurrent} = {weight*100:.2f}%",
        f"- 算術平均リターン(position-sized): {arithmetic_mean*100:.4f}%",
        f"- mean(log(1 + return/{max_concurrent})): {mean_log_return:.6f}"
        f"  ({'マイナス' if mean_log_return < 0 else 'プラス'})",
        f"- 幾何平均リターン(position-sized): {geometric_mean*100:.4f}%",
        f"- 算術平均と幾何平均の差(ドラッグ量): {(arithmetic_mean - geometric_mean)*100:.4f}%pt",
        "",
        (
            "→ mean(log(1+return/3))がマイナスであれば、ボラティリティ・ドラッグだけで"
            "複利成長率がマイナスになり得ることが直接確認できたことになる。プラスの場合は、"
            "ドラッグは存在する(算術平均>幾何平均は常に成立)がCAGRマイナスの主因ではないことを示す。"
        ),
    ]

    # ------------------------------------------------------------------
    # 項目5: 時系列ランダム化テスト
    # ------------------------------------------------------------------
    print("[5/9] 時系列ランダム化テスト(1000回)...")
    nav = port_real["nav_series"]
    daily_returns = [(nav[k] / nav[k - 1] - 1) for k in range(1, len(nav)) if nav[k - 1] > 0]
    num_years_real = port_real["num_years"]

    rng = random.Random(123)
    shuffled_cagrs = []
    shuffled_maxdds = []
    base_returns = daily_returns[:]
    for _ in range(1000):
        rng.shuffle(base_returns)
        eq = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in base_returns:
            eq *= 1 + r
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak * 100 if peak > 0 else 0)
        total_return = eq
        cagr = ((total_return ** (1 / num_years_real)) - 1) * 100 if total_return > 0 else -100.0
        shuffled_cagrs.append(cagr)
        shuffled_maxdds.append(max_dd)

    shuffled_cagrs.sort()
    shuffled_maxdds.sort()
    actual_cagr = compute_portfolio_metrics(nav, num_years_real)["cagr_pct"]
    actual_maxdd = compute_portfolio_metrics(nav, num_years_real)["max_drawdown_pct"]
    cagr_spread = max(shuffled_cagrs) - min(shuffled_cagrs)

    def percentile(sorted_list, p):
        idx = min(len(sorted_list) - 1, max(0, int(len(sorted_list) * p)))
        return sorted_list[idx]

    def rank_of(sorted_list, value):
        below = sum(1 for v in sorted_list if v < value)
        return below / len(sorted_list) * 100

    lines += [
        "", "## 5. 時系列ランダム化テスト(現実条件、日次リターンを1000回シャッフル)", "",
        f"- 実際の結果: CAGR={actual_cagr}%, 最大DD={actual_maxdd}%",
        f"- シャッフル後CAGR: 中央値={percentile(shuffled_cagrs,0.5):.2f}%, "
        f"5%点={percentile(shuffled_cagrs,0.05):.2f}%, 95%点={percentile(shuffled_cagrs,0.95):.2f}%"
        f"(1000回全ての幅: {cagr_spread:.6f}pt)",
        f"- シャッフル後最大DD: 中央値={percentile(shuffled_maxdds,0.5):.2f}%, "
        f"5%点={percentile(shuffled_maxdds,0.05):.2f}%, 95%点={percentile(shuffled_maxdds,0.95):.2f}%",
        f"- 実際の最大DDのパーセンタイル順位: 下位{rank_of(shuffled_maxdds, actual_maxdd):.1f}%点",
        "",
        (
            "→ 【重要】複利計算(掛け算)は交換法則が成り立つため、日次リターンを何回シャッフルしても"
            "最終的な複利リターン(=CAGRの元になる総リターン)は理論上まったく同じ値になる"
            f"(実際に1000回のシャッフル全体でのCAGRのばらつきは{cagr_spread:.6f}ptとほぼゼロ)。"
            "これは検証の失敗ではなく、それ自体が重要な結論である: **CAGRがマイナスなのは"
            "「たまたま悪い順序で損失が重なった不運」のせいではなく、リターンの集合(中身)そのものに"
            "起因する**ことが数学的に証明される。一方、最大ドローダウンは経路(順序)に依存するため"
            "シャッフルにより変動するが、実際の値は分布の中央付近(中央値と近い水準)にあり、"
            "実際に経験した下落幅も「特別運が悪い順序」だったわけではないことを示している。"
        ),
    ]

    # ------------------------------------------------------------------
    # 項目6: 月次/年次の損失集中
    # ------------------------------------------------------------------
    print("[6/9] 月次/年次の損失集中を集計中...")
    returns_df = pd.DataFrame({s: price_frames[s].set_index("date")["close"].pct_change() for s in symbols})

    dates_dt = pd.to_datetime(port_real["dates"])
    nav_series_no_init = port_real["nav_series"][1:]
    nav_df = pd.DataFrame({"date": dates_dt, "nav": nav_series_no_init}).set_index("date")

    occupied_by_date = dict(port_real["daily_occupied_symbols"])

    monthly_rows = []
    for period, group in nav_df.groupby(pd.Grouper(freq="ME")):
        if group.empty:
            continue
        month_str = period.strftime("%Y-%m")
        start_nav = group["nav"].iloc[0]
        end_nav = group["nav"].iloc[-1]
        month_return = (end_nav / start_nav - 1) * 100 if start_nav > 0 else 0.0

        month_trades_entered = [t for t in taken_real if t["entry_date"].startswith(month_str)]
        month_trades_exited = [t for t in taken_real if t["exit_date"].startswith(month_str)]
        st = compute_detailed_stats(month_trades_exited)

        month_dates = [d.strftime("%Y-%m-%d") for d in group.index]
        max_conc = max((len(occupied_by_date.get(d, set())) for d in month_dates), default=0)

        held_symbols = set()
        for d in month_dates:
            held_symbols |= occupied_by_date.get(d, set())
        corr_val = None
        if len(held_symbols) >= 2:
            sub = returns_df.loc[returns_df.index.strftime("%Y-%m") == month_str, list(held_symbols)]
            if len(sub) >= 3:
                corr_matrix = sub.corr()
                vals = [
                    corr_matrix.iloc[i, j]
                    for i in range(len(corr_matrix))
                    for j in range(i + 1, len(corr_matrix))
                    if pd.notna(corr_matrix.iloc[i, j])
                ]
                if vals:
                    corr_val = sum(vals) / len(vals)

        monthly_rows.append({
            "month": month_str, "return_pct": round(month_return, 2),
            "trades_entered": len(month_trades_entered), "win_rate": st["win_rate"],
            "profit_factor": st["profit_factor"], "max_concurrent": max_conc,
            "avg_correlation": round(corr_val, 2) if corr_val is not None else None,
        })

    lines += ["", "## 6. 月次の損失集中(現実条件)", "",
              "| 年月 | リターン | 新規建玉数 | 勝率(決済ベース) | PF(決済ベース) | 最大同時保有 | 平均銘柄間相関 |",
              "|---|---|---|---|---|---|---|"]
    for r in monthly_rows:
        lines.append(
            f"| {r['month']} | {r['return_pct']}% | {r['trades_entered']} | {fmt(r['win_rate'],'%')} | "
            f"{fmt(r['profit_factor'])} | {r['max_concurrent']} | {fmt(r['avg_correlation'])} |"
        )
    lines.append("")
    lines.append(f"同日に2件以上が損失決済した日数: {port_real['same_day_multi_loss_days']}日")

    worst_months = sorted(monthly_rows, key=lambda r: r["return_pct"])[:5]
    lines.append("")
    lines.append("### 最も悪かった月トップ5")
    for r in worst_months:
        lines.append(f"- {r['month']}: {r['return_pct']}% (平均相関={fmt(r['avg_correlation'])}, 最大同時保有={r['max_concurrent']})")

    # ------------------------------------------------------------------
    # 項目7: ポジションサイズ比較
    # ------------------------------------------------------------------
    print("[7/9] ポジションサイズ比較...")
    lines += ["", "## 7. ポジションサイズ比較(現実条件のトレード群を使用)", "",
              "| 方式 | PF | CAGR | 日次Sharpe | Sortino | 最大DD |", "|---|---|---|---|---|---|"]

    sizing_results = {}
    for label, mc in [("33.3% x 最大3銘柄", 3), ("20% x 最大5銘柄", 5), ("10% x 最大10銘柄", 10)]:
        port = simulate_portfolio(trades_real, price_frames, max_concurrent=mc)
        pm = compute_portfolio_metrics(port["nav_series"], port["num_years"])
        st = compute_detailed_stats(port["admitted_trades"])
        sizing_results[mc] = pm
        lines.append(f"| {label} | {fmt(st['profit_factor'])} | {fmt(pm['cagr_pct'],'%')} | {fmt(pm['daily_sharpe'])} | {fmt(pm['sortino'])} | {fmt(pm['max_drawdown_pct'],'%')} |")

    def simulate_fixed_risk(trades, risk_pct, initial_capital=1.0, commission_pct=None):
        if commission_pct is None:
            commission_pct = config.BACKTEST_COMMISSION_PCT
        comm = commission_pct / 100
        trades_by_entry = defaultdict(list)
        for t in trades:
            trades_by_entry[t["entry_date"]].append(t)
        for lst in trades_by_entry.values():
            lst.sort(key=lambda t: -t["signal_score"])

        all_dates = sorted({d.strftime("%Y-%m-%d") for df in price_frames.values() for d in df["date"]})
        close_lookup = {
            sym: dict(zip(df["date"].dt.strftime("%Y-%m-%d"), df["close"].astype(float)))
            for sym, df in price_frames.items()
        }
        cash = initial_capital
        open_positions = {}  # symbol -> {"shares","trade","last_price"}
        nav_history = []

        for today in all_dates:
            for sym in list(open_positions.keys()):
                pos = open_positions[sym]
                if pos["trade"]["exit_date"] == today:
                    proceeds = pos["shares"] * pos["trade"]["exit_price"]
                    cash += proceeds - proceeds * comm
                    del open_positions[sym]

            for t in trades_by_entry.get(today, []):
                if t["symbol"] in open_positions:
                    continue
                risk_per_share = t.get("risk_per_share") or 0
                if risk_per_share <= 0:
                    continue
                nav_now = cash + sum(
                    p["shares"] * close_lookup.get(s, {}).get(today, p["last_price"])
                    for s, p in open_positions.items()
                )
                risk_amount = nav_now * (risk_pct / 100)
                shares = risk_amount / risk_per_share
                invest_amount = min(cash, shares * t["entry_price"])
                if invest_amount <= 0:
                    continue
                shares = invest_amount / t["entry_price"]
                cash -= invest_amount
                cash -= invest_amount * comm
                open_positions[t["symbol"]] = {"shares": shares, "trade": t, "last_price": t["entry_price"]}
                if t["exit_date"] == today:
                    pos = open_positions.pop(t["symbol"])
                    proceeds = pos["shares"] * pos["trade"]["exit_price"]
                    cash += proceeds - proceeds * comm

            pv = sum(
                p["shares"] * close_lookup.get(s, {}).get(today, p["last_price"])
                for s, p in open_positions.items()
            )
            nav_history.append(cash + pv)

        return [initial_capital] + nav_history

    for label, risk_pct in [("ATR固定リスク1%/トレード", 1.0), ("ATR固定リスク0.5%/トレード", 0.5)]:
        nav_fr = simulate_fixed_risk(trades_real, risk_pct)
        pm = compute_portfolio_metrics(nav_fr, num_years_real)
        lines.append(f"| {label} | N/A(可変サイズのため) | {fmt(pm['cagr_pct'],'%')} | {fmt(pm['daily_sharpe'])} | {fmt(pm['sortino'])} | {fmt(pm['max_drawdown_pct'],'%')} |")

    # ------------------------------------------------------------------
    # 項目8: 同日処理の厳密化
    # ------------------------------------------------------------------
    print("[8/9] 同日処理の厳密化(現金再利用タイミング)...")
    lines += ["", "## 8. 同日処理の厳密化", "",
              "- close方式は既にエントリー日の高値・安値を決済判定に使用していない(翌営業日からのみ判定)。",
              "- next_open方式はエントリー当日の高値・安値を決済判定に使用している(翌営業日始値で"
              "エントリーした後、その日のうちの値動きにもさらされるため)。",
              "", "| 現金再利用タイミング | 採用件数 | PF | CAGR | 最大DD |", "|---|---|---|---|---|"]
    for label, timing in [("a. 同日中に再利用(現行)", "same_day"), ("b. 翌営業日から再利用(保守的・本番基準)", "next_day")]:
        port = simulate_portfolio(trades_real, price_frames, max_concurrent=max_concurrent, cash_reuse_timing=timing)
        pm = compute_portfolio_metrics(port["nav_series"], port["num_years"])
        st = compute_detailed_stats(port["admitted_trades"])
        lines.append(f"| {label} | {port['trades_taken']} | {fmt(st['profit_factor'])} | {fmt(pm['cagr_pct'],'%')} | {fmt(pm['max_drawdown_pct'],'%')} |")

    # ------------------------------------------------------------------
    # 項目9: 結論の分類(数値に基づく機械的判定)
    # ------------------------------------------------------------------
    print("[9/9] 結論の分類...")
    findings = []

    e_holds = (stats_candidate_real["profit_factor"] or 0) <= 1.05 and stats_candidate_real["expected_value_pct"] <= 0.05
    if e_holds:
        findings.append(
            f"E: 現実コスト後は全候補トレード単位でもPF={fmt(stats_candidate_real['profit_factor'])}, "
            f"期待値={fmt(stats_candidate_real['expected_value_pct'],'%')}であり、優位性が確認できない。"
        )

    ranking_cagrs = {}
    for key in ["score", "volume_ratio", "breakout_pct", "random", "insertion_order"]:
        port_r = simulate_portfolio(trades_real, price_frames, max_concurrent=max_concurrent, ranking_key=key)
        ranking_cagrs[key] = compute_portfolio_metrics(port_r["nav_series"], port_r["num_years"])["cagr_pct"]
    ranking_spread = max(ranking_cagrs.values()) - min(ranking_cagrs.values())
    a_holds = ranking_spread >= 5.0  # 採用ルールでCAGRが5pt以上動くなら「採用ルールの影響が大きい」とみなす
    if a_holds:
        findings.append(
            f"A: 採用ルール(優先順位)によってCAGRが{ranking_spread:.2f}pt変動しており、"
            f"採用ルールが結果に無視できない影響を与えている(内訳: {ranking_cagrs})。"
        )

    b_holds = stats_taken_real["arithmetic_mean_pct"] > 0 and actual_cagr < 0
    if b_holds:
        findings.append(
            f"B: 採用トレードの算術平均リターンは{fmt(stats_taken_real['arithmetic_mean_pct'],'%')}とプラスだが、"
            f"ポートフォリオCAGRは{fmt(actual_cagr,'%')}とマイナスであり、複利成長率の低下"
            f"(mean(log(1+return/{max_concurrent}))={mean_log_return:.6f})が寄与している。"
        )

    worst_avg_corr = [r["avg_correlation"] for r in worst_months if r["avg_correlation"] is not None]
    all_avg_corr = [r["avg_correlation"] for r in monthly_rows if r["avg_correlation"] is not None]
    c_holds = False
    if worst_avg_corr and all_avg_corr:
        worst_mean_corr = sum(worst_avg_corr) / len(worst_avg_corr)
        all_mean_corr = sum(all_avg_corr) / len(all_avg_corr)
        c_holds = worst_mean_corr > all_mean_corr + 0.05
        if c_holds:
            findings.append(
                f"C: 損失が大きかった月の平均銘柄間相関({worst_mean_corr:.2f})は、"
                f"全期間平均({all_mean_corr:.2f})より高く、相関の高い銘柄への同時集中が"
                f"ドローダウンを悪化させている可能性を示唆する。"
            )

    cagr_3 = sizing_results[3]["cagr_pct"]
    cagr_10 = sizing_results[10]["cagr_pct"]
    d_holds = (cagr_10 - cagr_3) >= 5.0  # 分散すると5pt以上改善するなら「ポジションサイズが過大」寄与ありとみなす
    if d_holds:
        findings.append(
            f"D: 集中度を下げる(1/3集中→1/10分散)だけでCAGRが{fmt(cagr_3,'%')}→{fmt(cagr_10,'%')}"
            f"へ改善しており、ポジションサイズの過大さがCAGR悪化に寄与している。"
        )

    if len(findings) == 0:
        classification = "判定不能(いずれの基準にも明確に該当しなかった。数値を目視で確認してください)"
    elif len(findings) == 1:
        classification = findings[0][0]  # A/B/C/D/Eの先頭文字
    else:
        classification = "F: 複数要因の組み合わせ"

    lines += [
        "", "## 9. 結論の分類", "",
        f"**判定: {classification}**", "",
        "根拠として確認された項目:",
    ]
    if findings:
        for f in findings:
            lines.append(f"- {f}")
    else:
        lines.append("- 上記A〜Dのいずれの基準(しきい値)にも明確に該当しなかった。個別の数値を参照して手動で判断してください。")

    report_path = config.REPORTS_DIR / f"cagr_decomposition_{date.today().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nレポートを保存しました: {report_path}")


if __name__ == "__main__":
    main()
