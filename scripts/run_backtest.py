"""バックテスト実行スクリプト。

実行:
    python scripts/run_backtest.py

対象銘柄は backtest/watchlist.py の DEFAULT_WATCHLIST、期間・しきい値は config.py の
BACKTEST_* を参照する。結果は data/screener.db(backtest_results / backtest_summary)と
reports/backtest_YYYY-MM-DD.md に保存される。
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.watchlist import DEFAULT_WATCHLIST
from db.database import Database

METRIC_LABELS = [
    ("トレード数", "num_trades", ""),
    ("勝率", "win_rate", "%"),
    ("平均利益", "avg_gain_pct", "%"),
    ("平均損失", "avg_loss_pct", "%"),
    ("プロフィットファクター", "profit_factor", ""),
    ("最大ドローダウン", "max_drawdown_pct", "%"),
    ("平均保有日数", "avg_holding_days", "日"),
    ("Sharpe Ratio", "sharpe_ratio", ""),
    ("期待値", "expected_value_pct", "%"),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    run_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365 * config.BACKTEST_YEARS)).isoformat()
    end_date = date.today().isoformat()

    print(f"バックテスト開始: 対象{len(DEFAULT_WATCHLIST)}銘柄, 期間{start_date}〜{end_date}")
    trades = run_backtest(
        DEFAULT_WATCHLIST,
        years=config.BACKTEST_YEARS,
        min_technical_score=config.BACKTEST_MIN_TECHNICAL_SCORE,
        max_holding_days=config.BACKTEST_MAX_HOLDING_DAYS,
    )
    metrics = compute_metrics(trades)

    db = Database(config.DB_PATH)
    db.insert_backtest_trades(run_date, trades)
    db.insert_backtest_summary(
        {
            "run_date": run_date,
            "start_date": start_date,
            "end_date": end_date,
            "num_symbols": len(DEFAULT_WATCHLIST),
            "min_technical_score": config.BACKTEST_MIN_TECHNICAL_SCORE,
            "max_holding_days": config.BACKTEST_MAX_HOLDING_DAYS,
            **metrics,
        }
    )
    db.close()

    print("\n=== バックテスト結果 ===")
    for label, key, unit in METRIC_LABELS:
        value = metrics[key]
        print(f"{label}: {'計算不能' if value is None else f'{value}{unit}'}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / f"backtest_{run_date}.md"
    lines = [
        f"# バックテスト結果 ({run_date})",
        "",
        f"対象銘柄({len(DEFAULT_WATCHLIST)}件): {', '.join(DEFAULT_WATCHLIST)}",
        "",
        f"期間: {start_date} 〜 {end_date}",
        f"シグナル条件: テクニカルスコア {config.BACKTEST_MIN_TECHNICAL_SCORE}点以上(75点満点)",
        f"最大保有日数: {config.BACKTEST_MAX_HOLDING_DAYS}営業日",
        "",
        "| 指標 | 値 |",
        "|---|---|",
    ]
    for label, key, unit in METRIC_LABELS:
        value = metrics[key]
        lines.append(f"| {label} | {'計算不能' if value is None else f'{value}{unit}'} |")
    lines.append("")
    lines.append(
        "※ このバックテストはテクニカル条件のみのシミュレーションです。"
        "AIによる判断や当日の値動き上位ランキングによる銘柄選定は過去分を無料で再現できないため、"
        "対象外としています。本番パイプラインの成績をそのまま表すものではありません。"
    )
    lines.append(
        "※ 最大ドローダウンは、銘柄ごとに(その銘柄1本に集中投資していた場合の)複利資産曲線を計算し、"
        "その中で最も悪かった値を採用した簡易値です。分散投資を前提にした真のポートフォリオ全体の"
        "ドローダウンではなく、「この銘柄だけに張っていたら最悪どこまで沈んだか」の目安です。"
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nレポートを保存しました: {report_path}")


if __name__ == "__main__":
    main()
