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


def _load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Ты инженер НПЗ. Отвечай на русском, не выдумывай числа."


class QaHandler:
    def __init__(self, client: GigaChatClient, mcp, tracer):
        self.client = client
        self.mcp = mcp
        self.tracer = tracer

    def _load_context(self, decision_id: str) -> str:
        entries = self.tracer.get_trace(decision_id)
        orchestrator = next((e for e in entries if e.agent == "orchestrator"), None)
        if orchestrator is not None:
            return orchestrator.output_summary
        if entries:
            return json.dumps(
                [{"agent": e.agent, "output_summary": e.output_summary} for e in entries],
                ensure_ascii=False,
            )
        return "нет данных о рекомендации"

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
                system, messages, tools=tools, temperature=0.0
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
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        # Бюджет tool-call'ов исчерпан — финальный ответ без инструментов.
        final = await self.client.complete(system, messages, temperature=0.0)
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

