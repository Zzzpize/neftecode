"""MCP-сервер с read-only инструментами для GigaChat.

Инструменты: ``get_current_state``, ``get_forecast``, ``explain_recommendation``,
``get_history``. Все операции только читают данные и не меняют состояние.

``run()`` поднимает минимальный HTTP-сервер (``GET /tools``, ``POST /call``).
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pandas as pd


def _to_dict(value: Any) -> Any:
    """Рекурсивно приводит dataclass/pydantic/DataFrame к plain dict."""
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(v) for v in value]
    if is_dataclass(value):
        return {k: _to_dict(v) for k, v in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return _to_dict(value.model_dump())
    if isinstance(value, pd.DataFrame):
        return _to_dict(value.to_dict(orient="records"))
    return value


def _timestamp(value: str) -> datetime:
    if value in ("now", "current", "latest", ""):
        return datetime.now()
    return datetime.fromisoformat(value)


class MCPServer:
    def __init__(self, simulator, quality_agent, tracer):
        self.simulator = simulator
        self.quality_agent = quality_agent
        self.tracer = tracer

    # ---------- дескрипторы инструментов ----------

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": "get_current_state",
                "description": "Срез процесса на момент времени (одна строка телеметрии).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string", "description": "ISO 8601 время"},
                    },
                    "required": ["timestamp"],
                },
            },
            {
                "name": "get_forecast",
                "description": "Прогноз качества (АВТ + гидроочистка) сейчас или после действия.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string", "description": "ISO 8601 время"},
                        "action": {
                            "type": "object",
                            "description": "Опциональные уставки управляемых тегов",
                            "properties": {},
                        },
                    },
                    "required": ["timestamp"],
                },
            },
            {
                "name": "explain_recommendation",
                "description": "Объяснение ранее принятой рекомендации (шаги агентов).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision_id": {"type": "string", "description": "ID решения"},
                    },
                    "required": ["decision_id"],
                },
            },
            {
                "name": "get_history",
                "description": "Временной ряд одного тега за интервал.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string", "description": "Название тега"},
                        "ts_from": {"type": "string", "description": "ISO 8601 начало"},
                        "ts_to": {"type": "string", "description": "ISO 8601 конец"},
                    },
                    "required": ["tag", "ts_from", "ts_to"],
                },
            },
        ]


    # ---------- реализация инструментов ----------

    def get_current_state(self, timestamp: str) -> dict:
        state = self.simulator.get_state(_timestamp(timestamp))
        cols = [
            c for c in state.columns
            if not c.startswith("anomaly_") and not c.endswith("_age_h")
        ]
        return {"timestamp": timestamp, "state": _to_dict(state[cols])}

    async def get_forecast(self, timestamp: str, action: dict | None = None) -> dict:
        state = self.simulator.get_state(_timestamp(timestamp))
        if action:
            forecast = await self.quality_agent.forecast_after_action(state, action)
        else:
            forecast = await self.quality_agent.forecast(state)
        return {"timestamp": timestamp, "action": action, "forecast": _to_dict(forecast)}

    def explain_recommendation(self, decision_id: str) -> dict:
        entries = self.tracer.get_trace(decision_id)
        return {
            "decision_id": decision_id,
            "steps": [
                {
                    "agent": e.agent,
                    "output_summary": e.output_summary,
                    "duration_ms": e.duration_ms,
                }
                for e in entries
            ],
        }

    def get_history(self, tag: str, ts_from: str, ts_to: str) -> dict:
        frame = self.simulator.get_history(
            tag,
            _timestamp(ts_from),
            _timestamp(ts_to),
        )
        return {"tag": tag, "points": _to_dict(frame)}

    async def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        args = arguments or {}
        if name not in ("get_current_state", "get_forecast", "explain_recommendation", "get_history"):
            raise KeyError(f"Unknown tool: {name}")
        try:
            if name == "get_current_state":
                return self.get_current_state(args["timestamp"])
            if name == "get_forecast":
                return await self.get_forecast(args["timestamp"], args.get("action"))
            if name == "explain_recommendation":
                return self.explain_recommendation(args["decision_id"])
            return self.get_history(args["tag"], args["ts_from"], args["ts_to"])
        except Exception as exc:
            return {"error": f"{name} failed: {exc}"}

    # ---------- автономный HTTP-сервер ----------

    def run(self, port: int = 8765) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/tools":
                    self._send(200, {"tools": server.list_tools()})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/call":
                    self._send(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    request = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    self._send(400, {"error": f"invalid JSON: {exc}"})
                    return
                try:
                    result = asyncio.run(
                        server.call_tool(
                            request.get("name", ""),
                            request.get("arguments") or {},
                        )
                    )
                    self._send(200, result)
                except Exception as exc:
                    self._send(400, {"error": str(exc)})

            def log_message(self, *args) -> None:  # noqa: N802
                pass

        httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        httpd.serve_forever()

