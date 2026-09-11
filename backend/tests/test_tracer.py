"""Тесты трейсера (Фаза 2)."""
from __future__ import annotations

import sqlite3

import pytest

from tracing.logger import AgentTracer, TraceEntry
from tracing.store import connect, init_db


@pytest.fixture
def tracer(tmp_path):
    return AgentTracer(tmp_path / "tracing.db")


def test_init_db_creates_schema(tmp_path):
    db = tmp_path / "t.db"
    init_db(db)
    with connect(db) as conn:
        names = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "traces" in names


def test_start_decision_returns_unique_ids(tracer):
    ids = {tracer.start_decision() for _ in range(10)}
    assert len(ids) == 10


def test_record_and_get_trace_returns_ordered_sequence(tracer):
    decision_id = tracer.start_decision()
    tracer.record(decision_id, "data", {"ts": 1}, {"ok": True}, 12.5)
    tracer.record(decision_id, "quality", {}, {"risk": 0.3}, 30.0)

    trace = tracer.get_trace(decision_id)

    assert [e.agent for e in trace] == ["data", "quality"]
    assert [e.step_no for e in trace] == [1, 2]
    assert all(isinstance(e, TraceEntry) for e in trace)
    assert trace[0].duration_ms == 12.5


def test_output_summary_truncated_to_500_chars(tracer):
    decision_id = tracer.start_decision()
    tracer.record(decision_id, "agent", {}, {"text": "x" * 2000}, 0.0)

    entry = tracer.get_trace(decision_id)[0]

    assert len(entry.output_summary) == 500


def test_input_hash_is_deterministic(tracer):
    decision_id = tracer.start_decision()
    tracer.record(decision_id, "a", {"x": 1, "y": [2, 3]}, {}, 0.0)
    tracer.record(decision_id, "a", {"y": [2, 3], "x": 1}, {}, 0.0)

    trace = tracer.get_trace(decision_id)

    assert trace[0].input_hash == trace[1].input_hash


def test_list_recent_returns_decision_ids_fresh_first(tracer):
    ids = []
    for _ in range(3):
        did = tracer.start_decision()
        ids.append(did)
        tracer.record(did, "orchestrator", {}, {}, 0.0)

    recent = tracer.list_recent(limit=2)

    assert recent == ids[::-1][:2]
    assert len(recent) == 2


def test_connection_is_closed_after_context_manager(tmp_path):
    db = tmp_path / "c.db"
    with connect(db) as conn:
        captured = conn

    with pytest.raises(sqlite3.ProgrammingError):
        captured.execute("SELECT 1")
