"""バックテストの成績悪化要因を1つずつ切り分ける分解分析。

「翌日始値エントリー」「スリッページ0.2%」「手数料0.1%」を1つずつ組み合わせた7パターンを
比較し、どの要素が最も成績悪化に寄与しているかを特定する。
あわせて、翌日始値エントリーがシグナル当日終値から翌日始値への「ギャップアップ」により
エッジを失っている可能性も検証する。

実行:
    python scripts/run_backtest_decomposition.py
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from analysis.technical import compute_indicator_frame
from api.yfinance_client import fetch_historical_prices
from backtest.engine import simulate_symbol
from backtest.metrics import compute_metrics, compute_portfolio_metrics
from backtest.portfolio import simulate_portfolio
from backtest.watchlist import DEFAULT_WATCHLIST

CONFIGS = [
    ("1. 終値約定・コストなし(現行)", "close", 0.0, 0.0),
    ("2. 翌日始値・コストなし", "next_open", 0.0, 0.0),
    ("3. 終値約定・スリッページ0.2%のみ", "close", 0.2, 0.0),
    ("4. 終値約定・手数料0.1%のみ", "close", 0.0, 0.1),
    ("5. 翌日始値+スリッページのみ", "next_open", 0.2, 0.0),
    ("6. 翌日始値+手数料のみ", "next_open", 0.0, 0.1),
    ("7. 翌日始値+スリッページ+手数料(現実条件)", "next_open", config.BACKTEST_SLIPPAGE_PCT, config.BACKTEST_COMMISSION_PCT),
]


def fetch_all_histories(symbols: list[str], years: int) -> dict[str, list[dict]]:
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


def run_all_symbols(histories: dict[str, list[dict]], symbols: list[str], entry_mode: str, slippage_pct: float, commission_pct: float) -> list[dict]:
    trades = []
    for sym in symbols:
        trades.extend(
            simulate_symbol(
                sym, histories[sym],
                min_technical_score=config.BACKTEST_MIN_TECHNICAL_SCORE,
                max_holding_days=config.BACKTEST_MAX_HOLDING_DAYS,
                entry_mode=entry_mode, slippage_pct=slippage_pct, commission_pct=commission_pct,
            )
        )
    trades.sort(key=lambda t: t["entry_date"])
    return trades


def fmt(v, unit=""):
    return "計算不能" if v is None else f"{v}{unit}"


def analyze_gap(histories: dict[str, list[dict]], symbols: list[str], price_frames: dict) -> dict:
    """シグナル当日終値→翌営業日始値のギャップ(%)を、config2(翌日始値・コストなし)の
    トレード群について集計する。"""
    trades = run_all_symbols(histories, symbols, "next_open", 0.0, 0.0)
    gaps = []
    for t in trades:
        df = price_frames[t["symbol"]]
        signal_row = df.loc[df["date"] == t["signal_date"]]
        if signal_row.empty:
            # signal_dateは文字列、dfのdateはTimestampなので変換して再検索
            import pandas as pd
            signal_row = df.loc[df["date"] == pd.Timestamp(t["signal_date"])]
        if signal_row.empty:
            continue
        signal_close = float(signal_row.iloc[0]["close"])
        entry_open = t["entry_price"]  # config2はコストなしなので生値と同じ
        gap_pct = (entry_open / signal_close - 1) * 100
        gaps.append(gap_pct)

    if not gaps:
        return {"n": 0, "avg_gap_pct": 0.0, "median_gap_pct": 0.0, "pct_positive_gap": 0.0}

    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)
    median = gaps_sorted[n // 2] if n % 2 == 1 else (gaps_sorted[n // 2 - 1] + gaps_sorted[n // 2]) / 2
    positive = sum(1 for g in gaps if g > 0)
    return {
        "n": n,
        "avg_gap_pct": round(sum(gaps) / n, 3),
        "median_gap_pct": round(median, 3),
        "pct_positive_gap": round(positive / n * 100, 1),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    years = config.BACKTEST_YEARS
    window_start = (date.today() - timedelta(days=365 * years)).isoformat()

    print(f"価格データ取得中(対象{len(DEFAULT_WATCHLIST)}銘柄)...")
    histories = fetch_all_histories(DEFAULT_WATCHLIST, years)
    symbols = list(histories.keys())
    print(f"取得できた銘柄数: {len(symbols)}/{len(DEFAULT_WATCHLIST)}")

    price_frames = {sym: compute_indicator_frame(histories[sym]) for sym in symbols}

    rows = []
    for label, entry_mode, slippage_pct, commission_pct in CONFIGS:
        print(f"計算中: {label} ...")
        trades_all = run_all_symbols(histories, symbols, entry_mode, slippage_pct, commission_pct)
        trades = [t for t in trades_all if t["entry_date"] >= window_start]
        m = compute_metrics(trades)
        port = simulate_portfolio(trades, price_frames, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS)
        pm = compute_portfolio_metrics(port["nav_series"], port["num_years"])

        simple_avg_return = (sum(t["pnl_pct"] for t in trades) / len(trades)) if trades else 0.0
        equal_weight_cum_pnl = sum(t["pnl_pct"] for t in trades)  # 各トレード1単位ずつ投資した場合の合計損益(pt)
        portfolio_cum_pnl_pct = (port["nav_series"][-1] - port["nav_series"][0]) / port["nav_series"][0] * 100

        rows.append({
            "label": label, **m, **pm,
            "simple_avg_return": round(simple_avg_return, 3),
            "equal_weight_cum_pnl": round(equal_weight_cum_pnl, 1),
            "portfolio_cum_pnl_pct": round(portfolio_cum_pnl_pct, 2),
            "trades_taken": port["trades_taken"],
            "trades_skipped_no_slot": port["trades_skipped_no_slot"],
            "trades_skipped_cash": port["trades_skipped_cash"],
            "max_concurrent_used": port["max_concurrent_used"],
            "avg_capital_utilization_pct": port["avg_capital_utilization_pct"],
            "per_symbol_taken": port["per_symbol_taken"],
        })

    print("ギャップ分析中...")
    gap_stats = analyze_gap(histories, symbols, price_frames)

    lines = [
        f"# バックテスト分解分析 ({date.today().isoformat()})",
        "",
        f"対象銘柄: {len(symbols)}/{len(DEFAULT_WATCHLIST)}銘柄  期間: {window_start} 〜 {date.today().isoformat()}",
        "",
        "## 7パターン比較",
        "",
        "| パターン | トレード数 | 勝率 | PF | 期待値 | CAGR | 最大DD(ポートフォリオ) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['num_trades']} | {fmt(r['win_rate'], '%')} | {fmt(r['profit_factor'])} | "
            f"{fmt(r['expected_value_pct'], '%')} | {fmt(r['cagr_pct'], '%')} | {fmt(r['max_drawdown_pct'], '%')} |"
        )

    lines += [
        "",
        "## 整合性指標(パターンごと)",
        "",
        "| パターン | 単純平均リターン/トレード | 同額投資時の累積損益(pt) | ポートフォリオ累積損益 | CAGR | 総投資回数 | 平均資金使用率 | 最大同時保有数 | 見送り(枠不足) | 見送り(現金不足) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['simple_avg_return']}% | {r['equal_weight_cum_pnl']} | "
            f"{r['portfolio_cum_pnl_pct']}% | {fmt(r['cagr_pct'], '%')} | {r['trades_taken']} | "
            f"{r['avg_capital_utilization_pct']}% | {r['max_concurrent_used']} | "
            f"{r['trades_skipped_no_slot']} | {r['trades_skipped_cash']} |"
        )

    lines += ["", "### 銘柄別採用数(パターン7: 現実条件)", ""]
    real_row = next(r for r in rows if r["label"].startswith("7."))
    per_symbol = real_row["per_symbol_taken"]
    for sym, cnt in sorted(per_symbol.items(), key=lambda x: -x[1]):
        lines.append(f"- {sym}: {cnt}回")

    lines += [
        "",
        "## ギャップアップ分析(シグナル当日終値 → 翌営業日始値)",
        "",
        f"- サンプル数: {gap_stats['n']}件",
        f"- 平均ギャップ: {gap_stats['avg_gap_pct']}%",
        f"- 中央値ギャップ: {gap_stats['median_gap_pct']}%",
        f"- 上にギャップした割合: {gap_stats['pct_positive_gap']}%",
        "",
        "## 注記",
        "",
        "- 全パターンで最大保有10営業日・シグナル閾値45点・最大3銘柄同時保有ポートフォリオの前提は共通。",
        "- CAGR・最大ドローダウンはbacktest.portfolioによる簡易ポートフォリオシミュレーション"
        "(資金を3分割、空き枠は現金)に基づく値。",
        "- 【整合性監査で発見・修正した重大バグ】シグナル当日中に損切り/利確が確定する"
        "(entry_date == exit_dateの同日決済)トレードが発生すると、決済処理がエントリー処理より"
        "先に実行される日次ループの都合上、そのポジションのエグジットが永久に処理されずスロットが"
        "恒久的に埋まったままになるバグがあった(全トレードの約6%が該当し、2023年以降ほぼ全ての"
        "新規エントリーが停止していた)。独立実装(portfolio_reference.py)にも同じ設計上の見落としを"
        "引き写していたためクロスチェックでは検出できず、tests/tests_portfolio.pyの手計算ユニット"
        "テストで検出・修正し、修正後は両実装の結果が完全一致(誤差 4.4e-16)することを確認済み。",
        "- 同日中に損切り価格・利確価格の両方に到達した場合の優先順位(損切り優先/利確優先/始値に"
        "近い方優先)を比較したが、PF/勝率/期待値への影響は誤差の範囲(PF 0.98でほぼ不変)であり、"
        "重要な要因ではないことを確認した。",
    ]

    report_path = config.REPORTS_DIR / f"backtest_decomposition_{date.today().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nレポートを保存しました: {report_path}")


if __name__ == "__main__":
    main()
