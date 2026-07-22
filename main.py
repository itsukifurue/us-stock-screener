"""米国株AIスクリーニングツール メインエントリーポイント。

通常実行:
    python main.py
API予算を消費せずロジックのみ検証したい場合:
    python main.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from analysis.scoring import combine_scores, score_ai, score_technical
from analysis.technical import compute_indicators
from api.claude_client import ClaudeClient
from api.fmp_client import FMPBudgetExceeded, FMPClient
from db.database import Database
from pipeline.stage1_screener import run_stage1
from pipeline.stage2_enrichment import run_stage2
from pipeline.stage3_ai_ranking import run_stage3
from report.report_generator import save_report

logger = logging.getLogger(__name__)

DRY_RUN_SYMBOLS = ["DRYA", "DRYB", "DRYC"]


def setup_logging() -> None:
    # Windows環境で日本語コンソール出力が文字化けしないようにする
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / f"run_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def _seed_dry_run_data(db: Database) -> None:
    """API不使用でロジックを一気通貫確認するための合成データをdaily_pricesへ投入する。"""
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    for i, symbol in enumerate(DRY_RUN_SYMBOLS):
        if db.latest_price_date(symbol) == date.today().isoformat():
            continue
        base = 20 + i * 10 + np.cumsum(rng.normal(0.03, 0.8, n))
        close = base + rng.normal(0, 0.3, n)
        close[-25:] = close[-26] + np.cumsum(np.abs(rng.normal(0.5, 0.3, 25)))
        high = close + np.abs(rng.normal(0.3, 0.2, n))
        low = close - np.abs(rng.normal(0.3, 0.2, n))
        open_ = close + rng.normal(0, 0.2, n)
        volume = rng.integers(500_000, 1_000_000, n).astype(float)
        volume[-1] = volume[-20:-1].mean() * 2.5
        rows = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
            for d, o, h, l, c, v in zip(dates, open_, high, low, close, volume)
        ]
        db.insert_daily_prices(symbol, rows)
        db.upsert_stock({"symbol": symbol, "name": f"Dry Run Corp {symbol}", "exchange": "NASDAQ",
                          "sector": "Technology", "industry": "Software", "country": "US", "is_etf": False,
                          "market_cap": 5_000_000_000})


def run_dry_run(db: Database, run_id: int) -> list[dict]:
    _seed_dry_run_data(db)

    fake_ai_analysis = {
        "ai_score": 15,
        "news_positive": True,
        "earnings_good": True,
        "comment": "[DRY-RUN] 実際のAI分析は呼び出していません。",
        "risk_factors": "[DRY-RUN] ダミーのリスク要因",
        "news_summary": "[DRY-RUN] ダミーのニュース要約",
        "upside_reason": "[DRY-RUN]",
        "downside_risk": "[DRY-RUN]",
        "two_week_expectation": "[DRY-RUN]",
        "expected_value": "[DRY-RUN]",
    }

    results = []
    for symbol in DRY_RUN_SYMBOLS:
        indicators = compute_indicators(db.get_price_history(symbol))
        if indicators is None:
            continue
        tech_result = score_technical(indicators)
        ai_result = score_ai(fake_ai_analysis)
        combined = combine_scores(tech_result, ai_result)

        entry_price = indicators["current_close"]
        atr = indicators.get("atr14") or 0
        stop_loss = entry_price - config.STOP_LOSS_ATR_MULTIPLIER * atr
        take_profit_1 = entry_price + config.TAKE_PROFIT_1_ATR_MULTIPLIER * atr
        take_profit_2 = entry_price * (1 + config.TAKE_PROFIT_2_TARGET_RETURN)
        expected_return_pct = (take_profit_2 - entry_price) / entry_price * 100

        results.append(
            {
                "symbol": symbol,
                "company_name": f"Dry Run Corp {symbol}",
                "current_price": entry_price,
                "entry_price": entry_price,
                "stop_loss": round(stop_loss, 2),
                "take_profit_1": round(take_profit_1, 2),
                "take_profit_2": round(take_profit_2, 2),
                "expected_return_pct": round(expected_return_pct, 2),
                "win_rate": round(min(90, max(10, combined["total_score"])), 1),
                "expected_value": round(take_profit_2 - entry_price, 2),
                "ai_comment": fake_ai_analysis["comment"],
                "news_summary": fake_ai_analysis["news_summary"],
                "risk_factors": fake_ai_analysis["risk_factors"],
                "total_score": combined["total_score"],
            }
        )
        db.upsert_candidate(run_id, symbol, stage="final", total_score=combined["total_score"])

    results.sort(key=lambda x: x["total_score"], reverse=True)
    top3 = results[: config.TOP_N_FINAL]
    for item in top3:
        db.insert_analysis_result({"run_id": run_id, **item})
    return top3


def run_pipeline(db: Database, run_id: int) -> list[dict]:
    fmp = FMPClient(db)
    claude = ClaudeClient()

    symbols: list[str] = []
    top10: list[dict] = []
    top3: list[dict] = []
    try:
        symbols = run_stage1(db, fmp, run_id)
        top10 = run_stage2(db, run_id, symbols)
        top3 = run_stage3(db, fmp, claude, run_id, top10)
    except FMPBudgetExceeded as e:
        logger.warning("API予算超過によりパイプラインを打ち切りました: %s", e)

    db.finish_screening_run(run_id, candidate_count=len(symbols))
    return top3


def main() -> None:
    parser = argparse.ArgumentParser(description="米国株AIスクリーニングツール")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばずに合成データでロジックのみ検証する")
    args = parser.parse_args()

    setup_logging()
    run_date = date.today().isoformat()
    logger.info("=== 実行開始 run_date=%s dry_run=%s ===", run_date, args.dry_run)

    db = Database(config.DB_PATH)
    run_id = db.create_screening_run(run_date)

    try:
        if args.dry_run:
            top3 = run_dry_run(db, run_id)
            db.finish_screening_run(run_id, candidate_count=len(DRY_RUN_SYMBOLS))
        else:
            top3 = run_pipeline(db, run_id)
    finally:
        db.close()

    md_path = save_report(run_date, top3)
    logger.info("レポートを出力しました: %s", md_path)

    summary_lines = [f"=== 本日のTop{len(top3)}銘柄 ==="]
    for i, item in enumerate(top3, start=1):
        summary_lines.append(
            f"{i}. {item['symbol']} ({item['company_name']}) スコア={item['total_score']:.1f} "
            f"エントリー=${item['entry_price']:.2f} 利確②=${item['take_profit_2']:.2f}"
        )
    summary_lines.append(f"詳細レポート: {md_path}")
    summary_text = "\n".join(summary_lines)

    # タスクスケジューラ等コンソールが無い環境で実行された場合でもログに結果が残るようにする
    logger.info("\n%s", summary_text)
    print(f"\n{summary_text}")


if __name__ == "__main__":
    main()
