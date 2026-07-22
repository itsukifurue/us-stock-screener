"""日次レポート(Markdown + JSON)の生成。"""
from __future__ import annotations

import json
from pathlib import Path

import config

DISCLAIMER = (
    "※ 勝率・期待値は現時点ではバックテストエンジン未実装のため、総合スコアから導いた暫定値です。"
    "投資判断は自己責任で行ってください。本レポートは投資助言ではありません。"
)


def _format_stock_section(rank: int, item: dict) -> str:
    return f"""### {rank}位: {item['symbol']} — {item['company_name']}

| 項目 | 値 |
|---|---|
| 現在価格 | ${item['current_price']:.2f} |
| エントリー価格 | ${item['entry_price']:.2f} |
| 損切価格 | ${item['stop_loss']:.2f} |
| 利確価格① | ${item['take_profit_1']:.2f} |
| 利確価格② | ${item['take_profit_2']:.2f} |
| 期待上昇率 | {item['expected_return_pct']:.2f}% |
| 勝率(暫定) | {item['win_rate']:.1f}% |
| 期待値(暫定) | {item['expected_value']:.2f} |
| 総合スコア | {item['total_score']:.1f} / {config.SCORE_MAX} |

**AIコメント**: {item['ai_comment']}

**ニュース要約**: {item['news_summary']}

**リスク要因**: {item['risk_factors']}
"""


def generate_markdown(run_date: str, top3: list[dict]) -> str:
    if not top3:
        return f"# 米国株AIスクリーニング結果 ({run_date})\n\n本日は条件を満たす銘柄が見つかりませんでした。\n"

    sections = "\n".join(_format_stock_section(i, item) for i, item in enumerate(top3, start=1))
    return f"""# 米国株AIスクリーニング結果 ({run_date})

本日最も期待できる{len(top3)}銘柄

{sections}
---
{DISCLAIMER}
"""


def save_report(run_date: str, top3: list[dict]) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    md_path = config.REPORTS_DIR / f"{run_date}.md"
    md_path.write_text(generate_markdown(run_date, top3), encoding="utf-8")

    json_path = config.REPORTS_DIR / f"{run_date}.json"
    json_path.write_text(
        json.dumps({"run_date": run_date, "top": top3, "disclaimer": DISCLAIMER}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return md_path
