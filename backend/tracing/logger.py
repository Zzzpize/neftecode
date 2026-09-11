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
            output_summary=_dump(output)[:SUMMARY_LIMIT],
            duration_ms=float(duration_ms),
            timestamp=now.isoformat(),
        )

    def get_trace(self, decision_id: str) -> list[TraceEntry]:
        """Возвращает полную последовательность шагов по ``decision_id``."""
        return [_to_entry(row) for row in store.fetch_trace(self.db_path, decision_id)]

    def list_recent(self, limit: int = 50) -> list[str]:
        """Возвращает последние ``decision_id`` (свежие первыми)."""
        return store.fetch_recent(self.db_path, limit)
