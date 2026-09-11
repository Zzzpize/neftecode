"""SQLite-хранилище трейсов.

Схема создаётся автоматически при первом обращении. Любое обращение к базе
идёт через контекст-менеджер :func:`connect`, который гарантированно закрывает
соединение даже при ошибке (риск #6 из ``spec.md``).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    decision_id    TEXT    NOT NULL,
    step_no        INTEGER NOT NULL,
    agent          TEXT    NOT NULL,
    input_hash     TEXT    NOT NULL,
    output_summary TEXT    NOT NULL,
    duration_ms    REAL    NOT NULL,
    timestamp      TEXT    NOT NULL,
    PRIMARY KEY (decision_id, step_no)
);

CREATE INDEX IF NOT EXISTS idx_traces_decision ON traces (decision_id);
"""


@contextmanager
def connect(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Открывает SQLite-соединение и гарантированно закрывает его в ``finally``."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | str) -> None:
    """Создаёт схему трейсов, если её ещё нет."""
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def insert_entry(
    path: Path | str,
    *,
    decision_id: str,
    agent: str,
    input_hash: str,
    output_summary: str,
    duration_ms: float,
    timestamp: str,
) -> int:
    """Вставляет один шаг трейса и возвращает его ``step_no``."""
    with connect(path) as conn:
        step_no = conn.execute(
            "SELECT COALESCE(MAX(step_no), 0) + 1 AS next_no "
            "FROM traces WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()["next_no"]

        conn.execute(
            """
            INSERT INTO traces
                (decision_id, step_no, agent, input_hash, output_summary, duration_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (decision_id, step_no, agent, input_hash, output_summary, duration_ms, timestamp),
        )
        return step_no


def fetch_trace(path: Path | str, decision_id: str) -> list[sqlite3.Row]:
    """Возвращает строки трейса по ``decision_id`` в порядке ``step_no``."""
    with connect(path) as conn:
        return conn.execute(
            "SELECT * FROM traces WHERE decision_id = ? ORDER BY step_no",
            (decision_id,),
        ).fetchall()


def fetch_recent(path: Path | str, limit: int) -> list[str]:
    """Возвращает последние ``decision_id`` (свежие первыми)."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT decision_id FROM traces "
            "GROUP BY decision_id ORDER BY MAX(rowid) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [row["decision_id"] for row in rows]
