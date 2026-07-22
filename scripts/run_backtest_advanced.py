"""バックテスト高度化スクリプト。

現行(簡易・ノーコスト)バックテストに対して、以下を追加検証する:
  1. 翌営業日始値エントリー + スリッページ0.2% + 手数料0.1%(現実的コスト)
  2. 利益上位5銘柄を除外した頑健性チェック
  3. テクニカル条件ごとのアブレーション(1条件ずつ無効化)
  4. シグナルスコア帯別の集計
  5. 最大3銘柄同時保有のポートフォリオシミュレーション(CAGR・日次Sharpe・Sortino)
  6. 現行 vs 現実的 の比較表

実行:
    python scripts/run_backtest_advanced.py
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
from backtest.metrics import compute_concentration, compute_metrics, compute_portfolio_metrics
from backtest.portfolio import simulate_portfolio
from backtest.watchlist import DEFAULT_WATCHLIST

CONDITION_LABELS = {
    "volume_surge": "出来高急増",
    "ma5_gt_ma15": "5MA>15MA",
    "ma25_breakout": "25MA突破",
    "new_20d_high": "20日高値更新",
    "rsi_50_70": "RSI50-70",
}


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


def run_all_symbols(histories: dict[str, list[dict]], symbols: list[str], **kwargs) -> list[dict]:
    trades = []
    for sym in symbols:
        if sym not in histories:
            continue
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


def filter_to_window(trades: list[dict], start_date: str) -> list[dict]:
    return [t for t in trades if t["entry_date"] >= start_date]


def fmt(v, unit=""):
    return "計算不能" if v is None else f"{v}{unit}"


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

    # ---------- 1. 現行(close entry, ノーコスト) ----------
    print("\n[1/6] 現行方式(シグナル当日終値エントリー、コストなし)を計算中...")
    trades_current_all = run_all_symbols(histories, symbols, entry_mode="close", slippage_pct=0.0, commission_pct=0.0)
    trades_current = filter_to_window(trades_current_all, window_start)
    metrics_current = compute_metrics(trades_current)
    conc_current = compute_concentration(trades_current, top_n=config.BACKTEST_TOP_PROFIT_EXCLUDE_N)
    port_current = simulate_portfolio(trades_current, price_frames, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS)
    portmetrics_current = compute_portfolio_metrics(port_current["nav_series"], port_current["num_years"])

    # ---------- 2. 現実的コスト(翌営業日始値 + スリッページ + 手数料) ----------
    print("[2/6] 現実的コスト方式(翌営業日始値エントリー + スリッページ + 手数料)を計算中...")
    trades_real_all = run_all_symbols(
        histories, symbols,
        entry_mode=config.BACKTEST_REALISTIC_ENTRY_MODE,
        slippage_pct=config.BACKTEST_SLIPPAGE_PCT,
        commission_pct=config.BACKTEST_COMMISSION_PCT,
    )
    trades_real = filter_to_window(trades_real_all, window_start)
    metrics_real = compute_metrics(trades_real)
    conc_real = compute_concentration(trades_real, top_n=config.BACKTEST_TOP_PROFIT_EXCLUDE_N)
    port_real = simulate_portfolio(trades_real, price_frames, max_concurrent=config.BACKTEST_MAX_CONCURRENT_POSITIONS)
    portmetrics_real = compute_portfolio_metrics(port_real["nav_series"], port_real["num_years"])

    # ---------- 3. 利益上位N銘柄を除外(現実的コストベース) ----------
    print("[3/6] 利益上位銘柄を除外した頑健性チェックを計算中...")
    excluded_symbols = set(conc_real["top_n_symbols"])
    remaining_symbols = [s for s in symbols if s not in excluded_symbols]
    trades_excl_all = run_all_symbols(
        histories, remaining_symbols,
        entry_mode=config.BACKTEST_REALISTIC_ENTRY_MODE,
        slippage_pct=config.BACKTEST_SLIPPAGE_PCT,
        commission_pct=config.BACKTEST_COMMISSION_PCT,
    )
    trades_excl = filter_to_window(trades_excl_all, window_start)
    metrics_excl = compute_metrics(trades_excl)

    # ---------- 4. 条件アブレーション(現実的コストベース) ----------
    print("[4/6] 条件ごとのアブレーションを計算中...")
    ablation_results = {}
    for condition in CONDITION_LABELS:
        trades_abl_all = run_all_symbols(
            histories, symbols,
            entry_mode=config.BACKTEST_REALISTIC_ENTRY_MODE,
            slippage_pct=config.BACKTEST_SLIPPAGE_PCT,
            commission_pct=config.BACKTEST_COMMISSION_PCT,
            excluded_conditions={condition},
        )
        trades_abl = filter_to_window(trades_abl_all, window_start)
        ablation_results[condition] = compute_metrics(trades_abl)

    # ---------- 5. スコア帯別集計(現実的コストベース) ----------
    print("[5/6] スコア帯別集計を計算中...")
    band_results = []
    for lo, hi in config.BACKTEST_SCORE_BANDS:
        band_trades = [t for t in trades_real if lo <= t["signal_score"] <= hi]
        band_results.append(((lo, hi), compute_metrics(band_trades)))

    # ---------- 6. 比較表の出力 ----------
    print("[6/6] レポートを作成中...")

    lines = [
        f"# バックテスト高度化レポート ({date.today().isoformat()})",
        "",
        f"対象銘柄: {len(symbols)}/{len(DEFAULT_WATCHLIST)}銘柄  期間: {window_start} 〜 {date.today().isoformat()}",
        f"シグナル条件: テクニカルスコア{config.BACKTEST_MIN_TECHNICAL_SCORE}点以上(75点満点)  "
        f"最大保有{config.BACKTEST_MAX_HOLDING_DAYS}営業日  同時保有上限{config.BACKTEST_MAX_CONCURRENT_POSITIONS}銘柄",
        "",
        "## 現行 vs 現実的コスト 比較表",
        "",
        "| 指標 | 現行(当日終値・コストなし) | 現実的(翌営業日始値+スリッページ0.2%+手数料0.1%) |",
        "|---|---|---|",
        f"| トレード数 | {metrics_current['num_trades']} | {metrics_real['num_trades']} |",
        f"| 勝率 | {fmt(metrics_current['win_rate'], '%')} | {fmt(metrics_real['win_rate'], '%')} |",
        f"| プロフィットファクター | {fmt(metrics_current['profit_factor'])} | {fmt(metrics_real['profit_factor'])} |",
        f"| 期待値(1トレードあたり) | {fmt(metrics_current['expected_value_pct'], '%')} | {fmt(metrics_real['expected_value_pct'], '%')} |",
        f"| 最大DD(1銘柄集中の最悪ケース) | {fmt(metrics_current['max_drawdown_pct'], '%')} | {fmt(metrics_real['max_drawdown_pct'], '%')} |",
        f"| 利益上位{conc_current['top_n']}銘柄の利益占有率 | {fmt(conc_current['top_n_profit_share_pct'], '%')} | {fmt(conc_real['top_n_profit_share_pct'], '%')} |",
        f"| 赤字銘柄数 | {conc_current['num_losing_symbols']}/{conc_current['num_symbols']} | {conc_real['num_losing_symbols']}/{conc_real['num_symbols']} |",
        "",
        f"### 最大{config.BACKTEST_MAX_CONCURRENT_POSITIONS}銘柄同時保有ポートフォリオ("
        f"採用{port_current['trades_taken']}/見送り{port_current['trades_skipped']}件 → "
        f"採用{port_real['trades_taken']}/見送り{port_real['trades_skipped']}件)",
        "",
        "| 指標 | 現行 | 現実的コスト |",
        "|---|---|---|",
        f"| CAGR(年率換算リターン) | {fmt(portmetrics_current['cagr_pct'], '%')} | {fmt(portmetrics_real['cagr_pct'], '%')} |",
        f"| 日次Sharpe Ratio | {fmt(portmetrics_current['daily_sharpe'])} | {fmt(portmetrics_real['daily_sharpe'])} |",
        f"| Sortino Ratio | {fmt(portmetrics_current['sortino'])} | {fmt(portmetrics_real['sortino'])} |",
        f"| 最大ドローダウン(ポートフォリオ全体) | {fmt(portmetrics_current['max_drawdown_pct'], '%')} | {fmt(portmetrics_real['max_drawdown_pct'], '%')} |",
        "",
        f"## 利益上位{conc_real['top_n']}銘柄({', '.join(conc_real['top_n_symbols'])})を除外した場合(現実的コストベース)",
        "",
        "| 指標 | 除外前(全銘柄) | 除外後 |",
        "|---|---|---|",
        f"| 対象銘柄数 | {len(symbols)} | {len(remaining_symbols)} |",
        f"| トレード数 | {metrics_real['num_trades']} | {metrics_excl['num_trades']} |",
        f"| 勝率 | {fmt(metrics_real['win_rate'], '%')} | {fmt(metrics_excl['win_rate'], '%')} |",
        f"| プロフィットファクター | {fmt(metrics_real['profit_factor'])} | {fmt(metrics_excl['profit_factor'])} |",
        f"| 期待値 | {fmt(metrics_real['expected_value_pct'], '%')} | {fmt(metrics_excl['expected_value_pct'], '%')} |",
        "",
        "## 条件ごとのアブレーション(その条件を無効化した場合、現実的コストベース)",
        "",
        "| 除外した条件 | トレード数 | 勝率 | PF | 期待値 |",
        "|---|---|---|---|---|",
        f"| (なし・全条件有効) | {metrics_real['num_trades']} | {fmt(metrics_real['win_rate'], '%')} | {fmt(metrics_real['profit_factor'])} | {fmt(metrics_real['expected_value_pct'], '%')} |",
    ]
    for condition, label in CONDITION_LABELS.items():
        m = ablation_results[condition]
        lines.append(
            f"| {label}を除外 | {m['num_trades']} | {fmt(m['win_rate'], '%')} | {fmt(m['profit_factor'])} | {fmt(m['expected_value_pct'], '%')} |"
        )

    lines += [
        "",
        "## スコア帯別集計(現実的コストベース)",
        "",
        "| スコア帯 | トレード数 | 勝率 | PF | 期待値 |",
        "|---|---|---|---|---|",
    ]
    for (lo, hi), m in band_results:
        band_label = f"{lo}点" if lo == hi else f"{lo}〜{hi}点"
        lines.append(
            f"| {band_label} | {m['num_trades']} | {fmt(m['win_rate'], '%')} | {fmt(m['profit_factor'])} | {fmt(m['expected_value_pct'], '%')} |"
        )

    lines += [
        "",
        "## 注記",
        "",
        "- 「現行」はこれまでのシンプルなバックテスト(シグナル当日終値で即エントリーでき、"
        "コストもかからないという簡易的な前提)であり、実際には成立しないタイミングでの"
        "エントリーを仮定している(先読みバイアスに近い)。",
        "- 「現実的」は翌営業日の始値でのエントリー、スリッページ0.2%、手数料0.1%(往復0.2%相当)を"
        "反映している。手数料率は証券会社によって異なるため目安の値。",
        "- ポートフォリオシミュレーションは、資金を同時保有上限の数で均等分割し、空き枠は現金"
        "(リターン0%)のままとする簡易モデル。動的なリバランスは行っていない。",
        "- 最大ドローダウン(1銘柄集中)は「その銘柄1本に集中投資していた場合の最悪ケース」の目安で、"
        "分散投資を前提にした真のポートフォリオ・ドローダウンではない(ポートフォリオ全体の"
        "ドローダウンは上記の別セクションを参照)。",
        "- AIによる判断や当日の値動き上位ランキングによる銘柄選定は過去分を無料で再現できないため、"
        "全てテクニカル条件のみのシミュレーションである。本番パイプラインの実績をそのまま表すもの"
        "ではない。",
    ]

    report_path = config.REPORTS_DIR / f"backtest_advanced_{date.today().isoformat()}.md"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nレポートを保存しました: {report_path}")


if __name__ == "__main__":
    main()
