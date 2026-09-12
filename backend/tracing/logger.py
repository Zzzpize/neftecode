"""AgentTracer — пишет шаги обмена агентов в SQLite.

Один цикл принятия решения = один ``decision_id``, все сообщения агентов
связаны им.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracing import store

SUMMARY_LIMIT = 2000
TRUNCATION_MARKER = "…(обрезано)"


@dataclass
class TraceEntry:
    decision_id: str
    step_no: int
    agent: str
    input_hash: str
    output_summary: str
    duration_ms: float
    timestamp: datetime


def _dump(value: Any) -> str:
    """Детерминированная JSON-сериализация с fallback на ``str``."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _fits(value: Any, budget: int) -> bool:
    return len(_dump(value)) <= budget


def _truncate_json(value: Any, budget: int):
    """Рекурсивно укорачивает данные так, чтобы их JSON укладывался в budget.

    Возвращает значение, чья сериализация не длиннее ``budget`` символов и
    остаётся валидным JSON: строки режутся с маркером ``TRUNCATION_MARKER``,
    списки и словари сохраняют столько элементов, сколько помещается.
    """
    if _fits(value, budget):
        return value

    if isinstance(value, str):
        # Грубо режем по числу символов, затем уточняем до точного бюджета.
        cut = value[:budget]
        while cut and not _fits(cut + TRUNCATION_MARKER, budget):
            excess = len(_dump(cut + TRUNCATION_MARKER)) - budget
            cut = cut[: max(0, len(cut) - max(1, excess))]
        if not _fits(TRUNCATION_MARKER, budget):
            return ""  # не влезает даже маркер — отдаём пустую строку
        return cut + TRUNCATION_MARKER

    if isinstance(value, list):
        out: list = []
        for item in value:
            if _fits(out + [item], budget):
                out.append(item)
                continue
            room = budget - len(_dump(out))
            if room > len(_dump("")):
                truncated = _truncate_json(item, room)
                if _fits(out + [truncated], budget):
                    out.append(truncated)
            break
        return out

    if isinstance(value, dict):
        out: dict = {}
        for key, item in value.items():
            if _fits({**out, key: item}, budget):
                out[key] = item
                continue
            room = budget - len(_dump({**out, key: None}))
            if room > 0:
                truncated = _truncate_json(item, room)
                if _fits({**out, key: truncated}, budget):
                    out[key] = truncated
            break
        return out

    return value


def _dump_summary(output: Any) -> str:
    """Валидный JSON для ``output_summary``, уложенный в ``SUMMARY_LIMIT``."""
    text = _dump(output)
    if len(text) <= SUMMARY_LIMIT:
        return text
    return _dump(_truncate_json(output, SUMMARY_LIMIT))


def _hash(value: Any) -> str:
    return hashlib.sha256(_dump(value).encode("utf-8")).hexdigest()


def _to_entry(row) -> TraceEntry:
    return TraceEntry(
        decision_id=row["decision_id"],
        step_no=row["step_no"],
        agent=row["agent"],
        input_hash=row["input_hash"],
        output_summary=row["output_summary"],
        duration_ms=row["duration_ms"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )


class AgentTracer:
    """Пишет и читает трейсы обмена агентов в SQLite."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        store.init_db(self.db_path)

    def start_decision(self) -> str:
        """Возвращает новый уникальный ``decision_id``."""
        return uuid.uuid4().hex

    def record(
        self,
        decision_id: str,
        agent: str,
        input_data: dict,
        output: dict,
        duration_ms: float,
    ) -> None:
        """Логирует один шаг агента."""
        now = datetime.now(timezone.utc)
        store.insert_entry(
            self.db_path,
            decision_id=decision_id,
            agent=agent,
            input_hash=_hash(input_data),
            output_summary=_dump_summary(output),
            duration_ms=float(duration_ms),
            timestamp=now.isoformat(),
        )

    def get_trace(self, decision_id: str) -> list[TraceEntry]:
        """Возвращает полную последовательность шагов по ``decision_id``."""
        return [_to_entry(row) for row in store.fetch_trace(self.db_path, decision_id)]

    def list_recent(self, limit: int = 50) -> list[str]:
        """Возвращает последние ``decision_id`` (свежие первыми)."""
        return store.fetch_recent(self.db_path, limit)
