"""Тесты Formatter: LLM-вызов, fallback по режимам, только числа из JSON."""
from __future__ import annotations

import asyncio

from agents.schemas import OrchestratorOutput
from llm.formatter import format_recommendation, format_recommendation_fallback
from llm.gigachat_client import LLMResponse, LLMUnavailableError


class FakeClient:
    def __init__(self, text="", fail=False):
        self.text = text
        self.fail = fail

    async def complete(self, system, messages, tools=None, temperature=0.0):
        if self.fail:
            raise LLMUnavailableError("down")
        return LLMResponse(text=self.text, tool_calls=[], tokens_used=1, cached=False)


def _decision(mode="recommend", payload=None, explanation_text=""):
    return OrchestratorOutput(
        decision_id="d1",
        mode=mode,
        payload=payload or {},
        trace_id="d1",
        explanation_text=explanation_text,
    )


def _recommend_payload():
    return {
        "best_variant": {
            "action": {"hydro_T5": 310.0},
            "expected": {
                "predictions": {
                    "sulfur_ppm": {"mean": 6.8, "low": 6.3, "high": 7.3, "unit": "ppm"}
                },
                "spec_risk": {"sulfur_over_10": 0.24},
                "confidence": "high",
                "warnings": [],
            },
            "metrics": {"sulfur": 6.8, "yield": 0.85, "energy": 0.5, "severity": 0.2},
            "feasible": True,
            "infeasible_reason": None,
        },
        "alternatives": [],
        "checks": {
            "combined_spec_risk": {"sulfur_over_10": 0.24},
            "severity_class": "elevated",
        },
        "weights": {"safety": 0.5, "quality": 0.3, "yield": 0.15, "energy": 0.05},
    }


def test_format_recommendation_uses_llm_text():
    decision = _decision(payload=_recommend_payload())
    client = FakeClient(text="Карточка готова.")

    result = asyncio.run(format_recommendation(decision, client))

    assert result == "Карточка готова."


def test_format_recommendation_falls_back_when_llm_unavailable():
    decision = _decision(payload=_recommend_payload())
    client = FakeClient(fail=True)

    result = asyncio.run(format_recommendation(decision, client))

    assert "310.00" in result
    assert "6.80" in result


def test_fallback_recommend_contains_only_payload_numbers():
    result = format_recommendation_fallback(_decision(payload=_recommend_payload()))

    assert "310.00" in result
    assert "6.80" in result
    assert "0.24" in result
    assert "elevated" in result


def test_fallback_silent():
    result = format_recommendation_fallback(
        _decision(mode="silent", explanation_text="Режим стабилен, вмешательство не требуется.")
    )

    assert result == "Режим стабилен, вмешательство не требуется."


def test_fallback_refuse():
    result = format_recommendation_fallback(
        _decision(mode="refuse", payload={"reason": "данные устарели"})
    )

    assert result == "Рекомендация не выдана: данные устарели."
