"""テクニカルスコア計算とAI評価を合算した総合スコアリング。

配点は config.SCORE_WEIGHTS(要件定義書の配点をそのまま採用、合計120点分の
判定項目があり、最終スコアは config.SCORE_MAX(100点)でキャップする)。
"""
from __future__ import annotations

import config

VOLUME_SURGE_THRESHOLD = 2.0  # 平均出来高の何倍で「急増」とみなすか


def score_technical(indicators: dict) -> dict:
    """stage2で利用可能な情報(価格・出来高)のみから算出できる項目をスコアリングする。

    戻り値: {"breakdown": {...}, "subtotal": float}
    """
    w = config.SCORE_WEIGHTS
    breakdown = {}

    volume_ratio = indicators.get("volume_ratio")
    breakdown["volume_surge"] = w["volume_surge"] if (volume_ratio and volume_ratio >= VOLUME_SURGE_THRESHOLD) else 0

    sma5, sma15, sma25 = indicators.get("sma5"), indicators.get("sma15"), indicators.get("sma25")
    breakdown["ma5_gt_ma15"] = w["ma5_gt_ma15"] if (sma5 and sma15 and sma5 > sma15) else 0

    current_close = indicators.get("current_close")
    breakdown["ma25_breakout"] = w["ma25_breakout"] if (sma25 and current_close and current_close > sma25) else 0

    breakdown["new_20d_high"] = w["new_20d_high"] if indicators.get("new_20d_high") else 0

    rsi14 = indicators.get("rsi14")
    breakdown["rsi_50_70"] = w["rsi_50_70"] if (rsi14 is not None and 50 <= rsi14 <= 70) else 0

    subtotal = sum(breakdown.values())
    return {"breakdown": breakdown, "subtotal": subtotal}


def score_ai(ai_analysis: dict) -> dict:
    """stage3でClaudeの分析結果(news_positive/earnings_good/ai_score)からスコアリングする。"""
    w = config.SCORE_WEIGHTS
    breakdown = {
        "news_positive": w["news_positive"] if ai_analysis.get("news_positive") else 0,
        "earnings_good": w["earnings_good"] if ai_analysis.get("earnings_good") else 0,
        "ai_evaluation": min(w["ai_evaluation"], max(0, ai_analysis.get("ai_score", 0))),
    }
    subtotal = sum(breakdown.values())
    return {"breakdown": breakdown, "subtotal": subtotal}


def combine_scores(technical_result: dict, ai_result: dict) -> dict:
    """テクニカルスコアとAIスコアを合算し、config.SCORE_MAXでキャップした最終スコアを返す。"""
    breakdown = {**technical_result["breakdown"], **ai_result["breakdown"]}
    raw_total = technical_result["subtotal"] + ai_result["subtotal"]
    total_score = min(config.SCORE_MAX, raw_total)
    return {"breakdown": breakdown, "raw_total": raw_total, "total_score": total_score}
