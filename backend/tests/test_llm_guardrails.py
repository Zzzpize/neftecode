"""Тесты guardrails LLM (раздел 9 ``spec.md``).

Покрывают пост-фильтр чисел, fallback при недоступности GigaChat и
ограничение количества tool-calls.
"""
from __future__ import annotations

import asyncio

from agents.schemas import OrchestratorOutput
from llm.formatter import format_recommendation
from llm.gigachat_client import LLMResponse, LLMUnavailableError, ToolCall
from llm.guardrails import (
    collect_numbers,
    extract_numbers,
    find_invented_numbers,
    validate_numbers,
)
from llm.qa_handler import QaHandler


# ---------- примитивы пост-фильтра ----------

def test_extract_numbers_parses_decimals_and_negatives():
    text = "изменить hydro_T5 на 310.00, риск 0,24, смещение -0.5"

    assert extract_numbers(text) == [310.0, 0.24, -0.5]


def test_collect_numbers_recursive_ignores_bools_and_strings():
    source = {
        "action": {"hydro_T5": 310.0},
        "feasible": True,
        "label": "sulfur_over_10",
        "metrics": {"sulfur": 6.8, "yield": 0.85},
        "list": [1, {"x": 2.5}],
    }

    assert collect_numbers(source) == {310.0, 6.8, 0.85, 1.0, 2.5}


def test_validate_numbers_allows_known_values():
    source = {"sulfur": 8.2, "risk": 0.24}

    assert validate_numbers("сера 8.2, риск 0,24", source) is True


def test_validate_numbers_flags_invented_number():
    source = {"sulfur": 8.2}

    assert validate_numbers("сера 5.0", source) is False
    assert find_invented_numbers("сера 5.0", source) == [5.0]


def test_validate_numbers_uses_tolerance():
    source = {"yield": 0.85}

    assert validate_numbers("выход 0.8500000001", source) is True


# ---------- Formatter: fallback / guardrail ----------

def _decision(payload):
    return OrchestratorOutput(
        decision_id="d1",
        mode="recommend",
        payload=payload,
        trace_id="d1",
        explanation_text="",
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
        "checks": {"combined_spec_risk": {"sulfur_over_10": 0.24}, "severity_class": "elevated"},
        "weights": {"safety": 0.5, "quality": 0.3, "yield": 0.15, "energy": 0.05},
    }


class _Client:
    def __init__(self, text="", fail=False):
        self.text = text
        self.fail = fail

    async def complete(self, system, messages, tools=None, temperature=0.0):
        if self.fail:
            raise LLMUnavailableError("down")
        return LLMResponse(text=self.text, tool_calls=[], tokens_used=1, cached=False)


def test_formatter_falls_back_on_hallucinated_number():
    decision = _decision(_recommend_payload())
    client = _Client(text="Снизьте серу до 5.0 мг/кг.")  # 5.0 отсутствует в JSON

    result = asyncio.run(format_recommendation(decision, client))

    assert "5.0" not in result
    assert "310.00" in result  # fallback-шаблон


def test_formatter_keeps_valid_llm_text():
    decision = _decision(_recommend_payload())
    client = _Client(text="Прогноз серы 6.8 ppm, риск 0.24.")

    result = asyncio.run(format_recommendation(decision, client))

    assert result == "Прогноз серы 6.8 ppm, риск 0.24."


def test_formatter_fallback_when_llm_unavailable():
    decision = _decision(_recommend_payload())
    client = _Client(fail=True)

    result = asyncio.run(format_recommendation(decision, client))

    assert "310.00" in result


# ---------- Q&A: лимит tool-calls ----------

class _ToolResponse:
    text = ""
    tool_calls = [ToolCall(name="get_history", arguments={"tag": "hydro_T5"})]
    tokens_used = 1
    cached = False


class _FinalResponse:
    text = "Готово."
    tool_calls = []
    tokens_used = 1
    cached = False


class _ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, system, messages, tools=None, temperature=0.0):
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


class _MCP:
    def __init__(self):
        self.calls = 0

    def list_tools(self):
        return [{"name": "get_history", "description": "", "parameters": {}}]

    async def call_tool(self, name, arguments):
        self.calls += 1
        return {"ok": True}


class _Tracer:
    def get_trace(self, decision_id):
        return []


def test_qa_respects_hard_tool_call_limit():
    mcp = _MCP()
    client = _ScriptedClient([_ToolResponse(), _ToolResponse(), _FinalResponse()])
    handler = QaHandler(client=client, mcp=mcp, tracer=_Tracer())

    result = asyncio.run(handler.answer("d1", "Что с серой?", max_tool_calls=2))

    assert result.tool_calls_used == 2
    assert mcp.calls == 2
    assert result.answer == "Готово."
