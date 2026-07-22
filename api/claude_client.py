"""Claude(Anthropic API)クライアント。銘柄のニュース・決算・企業概要から構造化分析を得る。"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

import config

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """あなたは米国株のスイングトレードを専門とするアナリストです。
与えられた銘柄情報(企業概要・ニュース・決算・テクニカル状況)をもとに、
2週間以内に+15%の上昇が期待できるかを評価してください。
必ず有効なJSONのみを出力し、それ以外の文章は一切含めないでください。

出力するJSONのキー:
- ai_score: 0〜20の整数(このAIとしての総合評価点)
- news_positive: 直近ニュースが株価にとって好材料と言えるか(true/false)
- earnings_good: 直近決算が良好と言えるか(材料が無ければfalse)(true/false)
- upside_reason: 上昇が期待できる理由(日本語、2〜3文)
- downside_risk: 下落リスク(日本語、2〜3文)
- two_week_expectation: 2週間以内の期待度に関するコメント(日本語、1〜2文)
- expected_value: このトレードの期待値に関する一言コメント(日本語、1文)
- news_summary: ニュースの要約(日本語、1〜2文。ニュースが無ければ「特筆すべきニュースなし」)
- risk_factors: リスク要因(日本語、箇条書き相当の短い1文にまとめる)
- comment: 総合コメント(日本語、2〜3文)
"""


class ClaudeAnalysisError(RuntimeError):
    pass


class ClaudeClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.model = model or config.ANTHROPIC_MODEL
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def ping(self) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        return self._extract_text(resp).strip()

    @staticmethod
    def _extract_text(resp) -> str:
        """resp.content はテキスト以外のブロック(ThinkingBlock等)を含むことがあるため、
        text属性を持つブロックだけを連結して取り出す(content[0]決め打ちにしない)。"""
        parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
        if not parts:
            raise ClaudeAnalysisError(f"Claude応答にテキストブロックが含まれていません: {resp.content}")
        return "".join(parts)

    def analyze_stock(
        self,
        symbol: str,
        company_name: str,
        profile_summary: str,
        news_items: list[dict],
        financial_summary: str,
        technical_summary: str,
    ) -> dict:
        news_text = "\n".join(f"- {n.get('title', '')}: {n.get('summary', '')[:200]}" for n in news_items[:5]) or "ニュースなし"

        user_prompt = f"""銘柄: {symbol} ({company_name})

【企業概要】
{profile_summary}

【最近のニュース】
{news_text}

【財務サマリー】
{financial_summary}

【テクニカル状況】
{technical_summary}

上記をもとに指定のJSON形式で分析結果を出力してください。"""

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = self._extract_text(resp)
        return self._parse_json_response(raw_text, symbol)

    @staticmethod
    def _parse_json_response(raw_text: str, symbol: str) -> dict:
        text = raw_text.strip()
        # ```json ... ``` のようなコードフェンスに包まれていた場合に備えて除去
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Claude応答のJSONパースに失敗しました(symbol=%s): %s\nraw=%s", symbol, e, raw_text)
            data = {
                "ai_score": 0,
                "news_positive": False,
                "earnings_good": False,
                "upside_reason": "",
                "downside_risk": "",
                "two_week_expectation": "",
                "expected_value": "",
                "news_summary": "",
                "risk_factors": "AI応答の解析に失敗したため評価不能",
                "comment": raw_text[:500],
            }

        data["ai_score"] = max(0, min(20, int(data.get("ai_score", 0) or 0)))
        data["news_positive"] = bool(data.get("news_positive", False))
        data["earnings_good"] = bool(data.get("earnings_good", False))
        return data
