"""Тесты GigaChatClient: кеш, rate limit, ошибки, разбор tool_calls."""
from __future__ import annotations

import asyncio

import pytest

from gigachat.models import (
    ChatCompletion,
    Choices,
    FunctionCall,
    Messages,
    MessagesRole,
    Usage,
)

import llm.gigachat_client as gcc


def _ok_response():
    return ChatCompletion(
        choices=[
            Choices(
                message=Messages(
                    role=MessagesRole.ASSISTANT,
                    content="ответ",
                    function_call=None,
                ),
                index=0,
                finish_reason="stop",
            )
        ],
        created=0,
        model="GigaChat-Max",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        object="chat.completion",
    )


class FakeSDK:
    def __init__(self, responder=None, fail=False):
        self.responder = responder
        self.fail = fail
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def achat(self, chat):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            if self.fail:
                raise RuntimeError("boom")
            return self.responder(chat) if self.responder else _ok_response()
        finally:
            self.active -= 1

    async def aget_models(self):
        return []


def _client(monkeypatch, fake):
    monkeypatch.setattr(gcc, "GigaChat", lambda **kw: fake)
    return gcc.GigaChatClient(credentials="test-cred")


def test_cache_returns_cached_second_time(monkeypatch):
    fake = FakeSDK()
    client = _client(monkeypatch, fake)

    first = asyncio.run(client.complete("sys", [{"role": "user", "content": "hi"}]))
    second = asyncio.run(client.complete("sys", [{"role": "user", "content": "hi"}]))

    assert first.cached is False
    assert second.cached is True
    assert second.text == "ответ"
    assert fake.calls == 1


def test_rate_limit_keeps_five_concurrent(monkeypatch):
    fake = FakeSDK()
    client = _client(monkeypatch, fake)

    async def run_many():
        await asyncio.gather(*[
            client.complete("sys", [{"role": "user", "content": f"m{i}"}])
            for i in range(20)
        ])

    asyncio.run(run_many())

    assert fake.calls == 20
    assert fake.max_active <= 5


def test_parses_tool_call(monkeypatch):
    def responder(chat):
        return ChatCompletion(
            choices=[
                Choices(
                    message=Messages(
                        role=MessagesRole.ASSISTANT,
                        content="",
                        function_call=FunctionCall(
                            name="get_history",
                            arguments={"tag": "hydro_T5"},
                        ),
                    ),
                    index=0,
                    finish_reason="stop",
                )
            ],
            created=0,
            model="GigaChat-Max",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            object="chat.completion",
        )

    client = _client(monkeypatch, FakeSDK(responder=responder))

    result = asyncio.run(client.complete("sys", [{"role": "user", "content": "?"}]))

    assert result.text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_history"
    assert result.tool_calls[0].arguments == {"tag": "hydro_T5"}
    assert result.tokens_used == 2


def test_unavailable_without_credentials():
    client = gcc.GigaChatClient(credentials="")

    with pytest.raises(gcc.LLMUnavailableError):
        asyncio.run(client.complete("sys", [{"role": "user", "content": "hi"}]))


def test_sdk_error_raises_unavailable(monkeypatch):
    client = _client(monkeypatch, FakeSDK(fail=True))

    with pytest.raises(gcc.LLMUnavailableError):
        asyncio.run(client.complete("sys", [{"role": "user", "content": "hi"}]))


def test_health_ok_and_fail(monkeypatch):
    ok_client = _client(monkeypatch, FakeSDK())
    assert asyncio.run(ok_client.health()) is True

    class DownSDK:
        async def aget_models(self):
            raise RuntimeError("down")

    monkeypatch.setattr(gcc, "GigaChat", lambda **kw: DownSDK())
    down_client = gcc.GigaChatClient(credentials="x")
    assert asyncio.run(down_client.health()) is False
