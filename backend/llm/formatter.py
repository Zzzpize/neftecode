"""JSON оркестратора превращает в текст карточки.

Все числа в тексте должны присутствовать в исходном JSON — поэтому при
недоступности LLM используется ``format_recommendation_fallback``, который
берёт числа строго из ``OrchestratorOutput``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agents.schemas import OrchestratorOutput
from llm.gigachat_client import GigaChatClient, LLMUnavailableError
from llm.guardrails import validate_numbers

PROMPT_PATH = Path(__file__).parent / "prompts" / "formatter_system.md"


def _to_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(v) for v in value]
    if is_dataclass(value):
        return {k: _to_dict(v) for k, v in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return _to_dict(value.model_dump())
    return value


def _load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "Ты инженер НПЗ. Отвечай на русском, не выдумывай числа."


async def format_recommendation(
    decision: OrchestratorOutput,
    client: GigaChatClient,
) -> str:
    """Форматирует рекомендацию через GigaChat; при сбое — fallback."""
    source = _to_dict(decision)
    try:
        response = await client.complete(
            system=_load_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(source, ensure_ascii=False, default=str),
                }
            ],
            temperature=0.0,
        )
        text = response.text.strip()
        # Guardrail #1: каждое число в ответе обязано быть в исходном JSON.
        if text and validate_numbers(text, source):
            return text
        return format_recommendation_fallback(decision)
    except LLMUnavailableError:
        return format_recommendation_fallback(decision)


def format_recommendation_fallback(decision: OrchestratorOutput) -> str:
    """Шаблонный текст без LLM; числа берутся только из ``decision.payload``."""
    mode = decision.mode

    if mode == "silent":
        return decision.explanation_text or "Режим стабилен, вмешательство не требуется."

    if mode == "refuse":
        reason = decision.payload.get("reason") or "данные непригодны для рекомендации"
        return f"Рекомендация не выдана: {reason}."

    # mode == "recommend"
    best = decision.payload.get("best_variant") or {}
    action = best.get("action") or {}
    expected = best.get("expected") or {}
    predictions = expected.get("predictions") or {}
    checks = decision.payload.get("checks") or {}
    combined_risk = checks.get("combined_spec_risk") or {}

    parts: list[str] = []

    if action:
        changes = ", ".join(
            f"{tag} {value:.2f}" for tag, value in list(action.items())[:5]
        )
        parts.append(f"Рекомендуемые изменения: {changes}.")

    sulfur = predictions.get("sulfur_ppm") or {}
    sulfur_mean = sulfur.get("mean")
    if sulfur_mean is not None:
        unit = sulfur.get("unit", "ppm")
        parts.append(f"Прогноз серы после воздействия: {sulfur_mean:.2f} {unit}.")

    risk_over_10 = combined_risk.get("sulfur_over_10")
    if risk_over_10 is not None:
        parts.append(f"Риск превышения 10 ppm: {risk_over_10:.2f}.")

    severity = checks.get("severity_class")
    if severity:
        parts.append(f"Класс тяжести: {severity}.")

    if not parts:
        return "Рекомендация сформирована."
    return " ".join(parts)


class Formatter:
    """Обёртка для инъекции в ``Orchestrator`` (``formatter``-атрибут)."""

    def __init__(self, client: GigaChatClient):
        self.client = client

    async def format_recommendation(self, decision: OrchestratorOutput) -> str:
        return await format_recommendation(decision, self.client)

