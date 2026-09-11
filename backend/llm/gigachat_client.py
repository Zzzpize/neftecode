"""Клиент GigaChat: обёртка над SDK + кеш + rate limiting.

- LRU-кеш по хешу ``(system, messages, tools, temperature)`` (размер 256).
- Не более 5 одновременных запросов (``asyncio.Semaphore``).
- При недоступности API ``complete`` бросает ``LLMUnavailableError``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from gigachat import GigaChat
from gigachat.models import Chat, Function, FunctionCall, Messages, MessagesRole


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    tokens_used: int
    cached: bool


class LLMUnavailableError(RuntimeError):
    """Бросается, когда GigaChat недоступен или запрос завершился ошибкой."""


class GigaChatClient:
    def __init__(
        self,
        credentials: str,
        model: str = "GigaChat-Max",
        verify_ssl: bool = False,
        cache_size: int = 256,
    ):
        self.credentials = credentials
        self.model = model
        self.verify_ssl = verify_ssl
        self.cache_size = cache_size

        self._cache: OrderedDict[str, LLMResponse] = OrderedDict()
        self._semaphore = asyncio.Semaphore(5)
        self._client: GigaChat | None = None

    # ---------- внутреннее ----------

    def _get_client(self) -> GigaChat:
        if self._client is None:
            if not self.credentials:
                raise LLMUnavailableError("GigaChat credentials are not configured")
            self._client = GigaChat(
                credentials=self.credentials,
                model=self.model,
                verify_ssl_certs=self.verify_ssl,
            )
        return self._client

    @staticmethod
    def _cache_key(
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> str:
        payload = json.dumps(
            {
                "system": system,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_arguments(raw: Any) -> dict:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except (json.JSONDecodeError, TypeError):
            return {"_raw": raw}

    @staticmethod
    def _copy_response(response: LLMResponse, cached: bool) -> LLMResponse:
        return LLMResponse(
            text=response.text,
            tool_calls=[
                ToolCall(name=tc.name, arguments=dict(tc.arguments))
                for tc in response.tool_calls
            ],
            tokens_used=response.tokens_used,
            cached=cached,
        )

    def _build_messages(self, system: str, messages: list[dict]) -> list[Messages]:
        result = [Messages(role=MessagesRole.SYSTEM, content=system)]

        for item in messages:
            role = str(item.get("role", "user"))
            content = item.get("content") or ""

            if role in ("function", "tool"):
                result.append(
                    Messages(
                        role=MessagesRole.FUNCTION,
                        content=str(content),
                        name=item.get("name"),
                    )
                )
                continue

            function_call = None
            raw_calls = item.get("tool_calls") or item.get("function_call")
            if raw_calls:
                first = raw_calls[0] if isinstance(raw_calls, list) else raw_calls
                if isinstance(first, dict):
                    function_call = FunctionCall(
                        name=str(first.get("name", "")),
                        arguments=dict(first.get("arguments", {}) or {}),
                    )

            result.append(
                Messages(
                    role=MessagesRole(role),
                    content=str(content),
                    function_call=function_call,
                )
            )

        return result


    @staticmethod
    def _build_tools(tools: list[dict] | None) -> list[Function] | None:
        if not tools:
            return None
        return [
            Function(
                name=str(tool["name"]),
                description=str(tool.get("description", "")),
                parameters=tool.get("parameters") or {},
            )
            for tool in tools
        ]

    async def _request(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> LLMResponse:
        client = self._get_client()

        chat = Chat(
            model=self.model,
            messages=self._build_messages(system, messages),
            functions=self._build_tools(tools),
            temperature=temperature,
        )

        response = await client.achat(chat)

        if not response.choices:
            raise LLMUnavailableError("GigaChat returned no choices")

        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        if message.function_call is not None:
            tool_calls.append(
                ToolCall(
                    name=message.function_call.name,
                    arguments=self._parse_arguments(message.function_call.arguments),
                )
            )

        tokens = 0
        if response.usage is not None:
            tokens = int(response.usage.total_tokens or 0)

        return LLMResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            tokens_used=tokens,
            cached=False,
        )

    # ---------- публичный API ----------

    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        key = self._cache_key(system, messages, tools, temperature)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._copy_response(self._cache[key], cached=True)

        async with self._semaphore:
            try:
                response = await self._request(system, messages, tools, temperature)
            except LLMUnavailableError:
                raise
            except Exception as exc:
                raise LLMUnavailableError(f"GigaChat request failed: {exc}") from exc

        self._cache[key] = self._copy_response(response, cached=False)
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return response

    async def health(self) -> bool:
        try:
            client = self._get_client()
            await client.aget_models()
            return True
        except Exception:
            return False

