"""Stage3: Top10についてプロフィール/財務を取得しClaudeで分析→Top3を選出しレポート用データを作る。

win_rate/expected_value はバックテストエンジン(次フェーズ)実装前のMVPヒューリスティックであり、
総合スコアから導いた暫定値であることに注意(report側でも明記する)。

無料プランではニュース系エンドポイントが全て402(Restricted Endpoint)で使えないことを実キーで
確認済みのため(2026-07-21)、ニュース取得は行わずClaude分析はプロフィール・財務・テクニカル情報
のみで行う。また、stage1のランキングAPIはETF/時価総額/国でのフィルタができないため、
ここでプロフィール取得時にETF・時価総額・国の一次スクリーニング条件を適用する。
"""
from __future__ import annotations

import logging

import config
from analysis.scoring import combine_scores, score_ai
from api.claude_client import ClaudeClient
from api.fmp_client import FMPBudgetExceeded, FMPClient
from db.database import Database

logger = logging.getLogger(__name__)


def _summarize_profile(profile: dict | None) -> str:
    if not profile:
        return "企業概要データなし"
    market_cap = profile.get("mktCap", profile.get("marketCap", ""))
    return (
        f"{profile.get('companyName', '')} / セクター: {profile.get('sector', '')} / "
        f"業種: {profile.get('industry', '')} / 時価総額: {market_cap} / "
        f"概要: {(profile.get('description') or '')[:400]}"
    )


def _summarize_financials(statements: list[dict]) -> str:
    if not statements:
        return "財務データなし"
    latest = statements[0]
    return (
        f"直近期({latest.get('date', '')}): 売上高={latest.get('revenue')}, "
        f"純利益={latest.get('netIncome')}, EPS={latest.get('eps')}"
    )


def _summarize_technical(indicators: dict) -> str:
    return (
        f"終値={indicators.get('current_close')}, 5MA={indicators.get('sma5')}, 15MA={indicators.get('sma15')}, "
        f"25MA={indicators.get('sma25')}, RSI14={indicators.get('rsi14')}, ATR14={indicators.get('atr14')}, "
        f"出来高倍率={indicators.get('volume_ratio')}, 20日高値更新={indicators.get('new_20d_high')}, "
        f"52週高値との距離%={indicators.get('distance_from_52w_high_pct')}"
    )


def _fallback_ai_analysis(error: Exception) -> dict:
    return {
        "ai_score": 0,
        "news_positive": False,
        "earnings_good": False,
        "comment": f"AI分析に失敗しました: {error}",
        "risk_factors": "AI分析失敗のためリスク評価不能",
        "news_summary": "",
        "upside_reason": "",
        "downside_risk": "",
        "two_week_expectation": "",
        "expected_value": "",
    }


def run_stage3(
    db: Database,
    fmp: FMPClient,
    claude: ClaudeClient,
    run_id: int,
    technical_candidates: list[dict],
) -> list[dict]:
    """technical_candidates: stage2の出力(Top10)。戻り値: 最終Top3の詳細分析結果リスト。"""
    enriched: list[dict] = []

    for cand in technical_candidates:
        symbol = cand["symbol"]

        try:
            profile = fmp.company_profile(symbol)
        except FMPBudgetExceeded:
            logger.warning("API予算超過のためstage3をここで打ち切ります。")
            break
        except Exception as e:
            logger.warning("企業概要取得失敗 %s: %s", symbol, e)
            profile = None

        if profile:
            if profile.get("isEtf") or profile.get("isFund"):
                logger.info("除外(ETF/投資信託): %s", symbol)
                continue
            if profile.get("country") and profile.get("country") != config.SCREEN_COUNTRY:
                logger.info("除外(対象国外 country=%s): %s", profile.get("country"), symbol)
                continue
            market_cap = profile.get("marketCap")
            if market_cap is not None and market_cap < config.SCREEN_MIN_MARKET_CAP:
                logger.info("除外(時価総額不足): %s", symbol)
                continue

        try:
            financials = fmp.income_statement(symbol, limit=1, period="quarter")
        except FMPBudgetExceeded:
            logger.warning("API予算超過のためstage3をここで打ち切ります。")
            break
        except Exception as e:
            logger.warning("財務データ取得失敗 %s: %s", symbol, e)
            financials = []

        # 無料プランではニュース系エンドポイントが使えないため、ニュースは常に空とする
        news_items: list[dict] = []

        try:
            ai_analysis = claude.analyze_stock(
                symbol=symbol,
                company_name=(profile or {}).get("companyName", symbol),
                profile_summary=_summarize_profile(profile),
                news_items=news_items,
                financial_summary=_summarize_financials(financials),
                technical_summary=_summarize_technical(cand["indicators"]),
            )
        except Exception as e:
            logger.error("Claude分析失敗 %s: %s", symbol, e)
            ai_analysis = _fallback_ai_analysis(e)

        technical_score_result = {"breakdown": cand["breakdown"], "subtotal": cand["technical_score"]}
        ai_score_result = score_ai(ai_analysis)
        combined = combine_scores(technical_score_result, ai_score_result)

        entry_price = cand["indicators"]["current_close"]
        atr = cand["indicators"].get("atr14") or 0
        stop_loss = entry_price - config.STOP_LOSS_ATR_MULTIPLIER * atr
        take_profit_1 = entry_price + config.TAKE_PROFIT_1_ATR_MULTIPLIER * atr
        take_profit_2 = entry_price * (1 + config.TAKE_PROFIT_2_TARGET_RETURN)
        expected_return_pct = (take_profit_2 - entry_price) / entry_price * 100 if entry_price else 0

        # MVPヒューリスティック(バックテスト未実装のため暫定値、次フェーズで正式な統計値に置き換え予定)
        win_rate = round(min(90, max(10, combined["total_score"])), 1)
        avg_gain = take_profit_2 - entry_price
        avg_loss = entry_price - stop_loss
        expected_value = round((win_rate / 100) * avg_gain - (1 - win_rate / 100) * avg_loss, 2)

        db.upsert_candidate(
            run_id,
            symbol,
            stage="stage3",
            technical_score=cand["technical_score"],
            ai_score=ai_score_result["subtotal"],
            total_score=combined["total_score"],
        )

        enriched.append(
            {
                "symbol": symbol,
                "company_name": (profile or {}).get("companyName", symbol),
                "current_price": entry_price,
                "entry_price": entry_price,
                "stop_loss": round(stop_loss, 2),
                "take_profit_1": round(take_profit_1, 2),
                "take_profit_2": round(take_profit_2, 2),
                "expected_return_pct": round(expected_return_pct, 2),
                "win_rate": win_rate,
                "expected_value": expected_value,
                "ai_comment": ai_analysis.get("comment", ""),
                "news_summary": ai_analysis.get("news_summary", ""),
                "risk_factors": ai_analysis.get("risk_factors", ""),
                "total_score": combined["total_score"],
            }
        )

    enriched.sort(key=lambda x: x["total_score"], reverse=True)
    top_final = enriched[: config.TOP_N_FINAL]
    for rank, item in enumerate(top_final, start=1):
        db.insert_analysis_result({"run_id": run_id, **item})
        db.upsert_candidate(run_id, item["symbol"], stage="final", total_score=item["total_score"], rank=rank)

    logger.info("Stage3: %d件を分析し、最終Top%d件を選出", len(enriched), len(top_final))
    return top_final
