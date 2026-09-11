"""Тесты Q&A: цикл tool-call'ов, жёсткий лимит, контекст из трейсера."""
from __future__ import annotations

import asyncio

from llm.gigachat_client import LLMResponse, ToolCall
from llm.qa_handler import QaHandler, answer


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, system, messages, tools=None, temperature=0.0):
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
