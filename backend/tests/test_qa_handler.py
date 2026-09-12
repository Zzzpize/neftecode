"""Тесты Q&A: цикл tool-call'ов, жёсткий лимит, контекст из трейсера."""
from __future__ import annotations

import asyncio
import json

import pytest

from llm.gigachat_client import LLMResponse, ToolCall
from llm.qa_handler import DecisionNotFoundError, QaHandler, _summarize_tool_result, answer


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    async def complete(self, system, messages, tools=None, temperature=0.0, use_cache=True):
        self.messages.append(messages)
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


class FakeMCP:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [{"name": "get_history", "description": "", "parameters": {}}]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"tag": arguments.get("tag"), "ok": True}


class FakeEntry:
    def __init__(self, agent, summary):
        self.agent = agent
        self.output_summary = summary


class FakeTracer:
    def get_trace(self, decision_id):
        return [FakeEntry("orchestrator", '{"mode": "recommend"}')]


def _tool_response():
    return LLMResponse(
        text="",
        tool_calls=[
            ToolCall(
                name="get_history",
                arguments={"tag": "hydro_T5", "ts_from": "t1", "ts_to": "t2"},
            )
        ],
        tokens_used=1,
        cached=False,
    )


def _final_response():
    return LLMResponse(text="Сера растёт.", tool_calls=[], tokens_used=1, cached=False)


def test_answer_without_tool_calls():
    client = ScriptedClient([_final_response()])
    handler = QaHandler(client=client, mcp=FakeMCP(), tracer=FakeTracer())

    result = asyncio.run(handler.answer("d1", "Что с серой?"))

    assert result.answer == "Сера растёт."
    assert result.tool_calls_used == 0


def test_answer_executes_tool_then_answers():
    mcp = FakeMCP()
    client = ScriptedClient([_tool_response(), _final_response()])
    handler = QaHandler(client=client, mcp=mcp, tracer=FakeTracer())

    result = asyncio.run(handler.answer("d1", "Что с серой?"))

    assert result.answer == "Сера растёт."
    assert result.tool_calls_used == 1
    assert mcp.calls[0][0] == "get_history"
    assert mcp.calls[0][1]["tag"] == "hydro_T5"


def test_answer_respects_max_tool_calls():
    mcp = FakeMCP()
    client = ScriptedClient([_tool_response(), _tool_response(), _final_response()])
    handler = QaHandler(client=client, mcp=mcp, tracer=FakeTracer())

    result = asyncio.run(handler.answer("d1", "Что с серой?", max_tool_calls=2))

    assert result.tool_calls_used == 2
    assert result.answer == "Сера растёт."
    assert client.calls == 3


def test_module_level_answer():
    client = ScriptedClient([_final_response()])
    result = asyncio.run(
        answer("d1", "Что с серой?", client, mcp=FakeMCP(), tracer=FakeTracer())
    )

    assert result.answer == "Сера растёт."


class EmptyTracer:
    def get_trace(self, decision_id):
        return []


class TimestampTracer:
    def get_trace(self, decision_id):
        return [
            FakeEntry("data", '{"timestamp": "2025-08-01T12:00:00", "is_stale": false}'),
            FakeEntry("orchestrator", '{"mode": "recommend"}'),
        ]


def test_answer_raises_on_missing_decision():
    client = ScriptedClient([_final_response()])
    handler = QaHandler(client=client, mcp=FakeMCP(), tracer=EmptyTracer())

    with pytest.raises(DecisionNotFoundError):
        asyncio.run(handler.answer("unknown", "Что с серой?"))


def test_context_includes_recommendation_timestamp():
    client = ScriptedClient([_final_response()])
    handler = QaHandler(client=client, mcp=FakeMCP(), tracer=TimestampTracer())

    asyncio.run(handler.answer("d1", "Что с серой?"))

    user_content = client.messages[0][0]["content"]
    assert "Время рекомендации (timestamp): 2025-08-01T12:00:00" in user_content


def test_summarize_tool_result_trims_history_points():
    result = {"tag": "hydro_T5", "points": [{"t": i} for i in range(300)]}

    data = json.loads(_summarize_tool_result(result, "get_history"))

    assert data["omitted_points"] == 200
    assert len(data["points"]) == 100


def test_summarize_tool_result_truncates_long_result():
    big = {"value": "x" * 5000}

    text = _summarize_tool_result(big)

    assert text.endswith("…(обрезано)")
    assert len(text) <= 2000 + len("…(обрезано)")
