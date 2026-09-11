"""Тесты MCPServer: 4 read-only инструмента и диспетчеризация."""
from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd
import pytest

from llm.mcp_server import MCPServer


class FakeSimulator:
    def get_state(self, ts):
        return pd.DataFrame([
            {"date": pd.Timestamp(ts), "hydro_T5": 300.0, "pak_sulfur_ppm": 8.0}
        ])

    def get_history(self, tag, ts_from, ts_to):
        return pd.DataFrame([
            {"date": pd.Timestamp(ts_from), tag: 1.0},
            {"date": pd.Timestamp(ts_to), tag: 2.0},
        ])


class FakeQualityAgent:
    async def forecast(self, state):
        return {"sulfur_over_10": 0.4}

    async def forecast_after_action(self, state, action):
        return {"sulfur_over_10": 0.1, "applied_action": action}


class FakeEntry:
    def __init__(self, agent):
        self.agent = agent
        self.output_summary = "{}"
        self.duration_ms = 1.0


class FakeTracer:
    def get_trace(self, decision_id):
        return [FakeEntry("data"), FakeEntry("orchestrator")]


def _server():
    return MCPServer(
        simulator=FakeSimulator(),
        quality_agent=FakeQualityAgent(),
        tracer=FakeTracer(),
    )


def test_list_tools_has_four_readonly_tools():
    tools = _server().list_tools()

    assert {t["name"] for t in tools} == {
        "get_current_state",
        "get_forecast",
        "explain_recommendation",
        "get_history",
    }


def test_get_current_state():
    result = _server().get_current_state("2025-08-01T12:00:00")

    assert result["state"][0]["hydro_T5"] == 300.0


def test_get_forecast_without_action():
    result = asyncio.run(_server().get_forecast("2025-08-01T12:00:00"))

    assert result["forecast"]["sulfur_over_10"] == 0.4


def test_get_forecast_with_action():
    result = asyncio.run(
        _server().get_forecast("2025-08-01T12:00:00", {"hydro_T5": 310.0})
    )

    assert result["forecast"]["applied_action"] == {"hydro_T5": 310.0}


def test_get_history():
    result = _server().get_history(
        "hydro_T5",
        "2025-08-01T10:00:00",
        "2025-08-01T12:00:00",
    )

    assert result["tag"] == "hydro_T5"
    assert len(result["points"]) == 2


def test_explain_recommendation():
    result = _server().explain_recommendation("abc")

    assert result["decision_id"] == "abc"
    assert [s["agent"] for s in result["steps"]] == ["data", "orchestrator"]


def test_call_tool_dispatches():
    result = asyncio.run(
        _server().call_tool("get_current_state", {"timestamp": "2025-08-01T12:00:00"})
    )

    assert result["state"][0]["pak_sulfur_ppm"] == 8.0


def test_call_tool_unknown():
    with pytest.raises(KeyError):
        asyncio.run(_server().call_tool("mutate_state", {}))
