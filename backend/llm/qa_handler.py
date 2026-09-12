"""Q&A по последней рекомендации.

LLM получает контекст рекомендации из трейсера и доступ к read-only
MCP-инструментам. Лимит tool-call'ов жёсткий: ``max_tool_calls=3``.
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.schemas import AskOutput
from llm.gigachat_client import GigaChatClient

PROMPT_PATH = Path(__file__).parent / "prompts" / "qa_system.md"

# Результаты MCP-инструментов бывают очень длинными (история тегов, весь срез
# состояния). Слишком длинный контекст GigaChat иногда игнорирует, поэтому
# результат компактируем до разумного бюджета символов.
MAX_TOOL_RESULT_CHARS = 2000


class DecisionNotFoundError(LookupError):
    """Рекомендация с таким ``decision_id`` не найдена в трейсере."""


def _load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Ты инженер НПЗ. Отвечай на русском, не выдумывай числа."


def _summarize_tool_result(result, tool_name: str | None = None) -> str:
    """Компактная JSON-сериализация результата инструмента."""
    if tool_name == "get_history" and isinstance(result, dict):
        points = result.get("points")
        if isinstance(points, list) and len(points) > 100:
            omitted = len(points) - 100
            result = {**result, "points": points[:50] + points[-50:], "omitted_points": omitted}

    text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "…(обрезано)"
    return text


class QaHandler:
    def __init__(self, client: GigaChatClient, mcp, tracer):
        self.client = client
        self.mcp = mcp
        self.tracer = tracer

    @staticmethod
    def _parse_summary(summary: str) -> dict:
        try:
            data = json.loads(summary)
        except (json.JSONDecodeError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _extract_timestamp(entries) -> str | None:
        """Достаёт timestamp рекомендации из трейса.

        Надёжный источник — ``output_summary`` шага ``data`` (компактный JSON с
        полем ``timestamp``). Запасной вариант — ``payload.timestamp`` оркестратора.
        """
        for entry in entries:
            if entry.agent == "data":
                ts = QaHandler._parse_summary(entry.output_summary).get("timestamp")
                if ts:
                    return str(ts)
        for entry in entries:
            if entry.agent == "orchestrator":
                payload = QaHandler._parse_summary(entry.output_summary).get("payload")
                if isinstance(payload, dict) and payload.get("timestamp"):
                    return str(payload["timestamp"])
        return None

    def _load_context(self, decision_id: str) -> str:
        if self.tracer is None:
            return "нет данных о рекомендации"
        entries = self.tracer.get_trace(decision_id)
        if not entries:
            raise DecisionNotFoundError(decision_id)

        orchestrator = next((e for e in entries if e.agent == "orchestrator"), None)
        timestamp = self._extract_timestamp(entries)
        prefix = f"Время рекомендации (timestamp): {timestamp}\n" if timestamp else ""

        if orchestrator is not None:
            return prefix + orchestrator.output_summary
        return prefix + json.dumps(
            [{"agent": e.agent, "output_summary": e.output_summary} for e in entries],
            ensure_ascii=False,
        )

    async def answer(
        self,
        decision_id: str,
        question: str,
        max_tool_calls: int = 3,
    ) -> AskOutput:
        system = _load_prompt()
        context = self._load_context(decision_id)
        messages = [
            {
                "role": "user",
                "content": f"Контекст рекомендации:\n{context}\n\nВопрос: {question}",
            }
        ]

        tools = self.mcp.list_tools() if self.mcp is not None else None

        tool_calls_used = 0
        while tool_calls_used < max_tool_calls:
            response = await self.client.complete(
                system, messages, tools=tools, temperature=0.0, use_cache=False
            )

            if not response.tool_calls:
                return AskOutput(
                    answer=response.text.strip(),
                    tool_calls_used=tool_calls_used,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                if tool_calls_used >= max_tool_calls:
                    break
                result = await self.mcp.call_tool(tc.name, tc.arguments)
                tool_calls_used += 1
                messages.append(
                    {
                        "role": "function",
                        "name": tc.name,
                        "content": _summarize_tool_result(result, tc.name),
                    }
                )

        # Бюджет tool-call'ов исчерпан — финальный ответ без инструментов.
        final = await self.client.complete(system, messages, temperature=0.0, use_cache=False)
        answer_text = final.text.strip() or "Не удалось сформулировать ответ."
        return AskOutput(answer=answer_text, tool_calls_used=tool_calls_used)


async def answer(
    decision_id: str,
    question: str,
    client: GigaChatClient,
    max_tool_calls: int = 3,
    mcp=None,
    tracer=None,
) -> AskOutput:
    """Модульная функция (контракт из ``spec.md``); оборачивает ``QaHandler``."""
    return await QaHandler(client=client, mcp=mcp, tracer=tracer).answer(
        decision_id, question, max_tool_calls
    )

