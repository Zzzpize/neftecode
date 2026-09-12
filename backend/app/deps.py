"""DI-контейнер: ленивая инициализация shared-объектов.

Сборка агентного пайплайна AI-слоя: при наличии ML-артефактов оркестратор
использует реальные модели, иначе — mock-режим агентов (state при этом берётся
из реального симулятора, а Formatter деградирует до fallback-текста).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException

from agents.data_agent import DataAgent
from agents.optimization_agent import OptimizationAgent
from agents.orchestrator import Orchestrator
from agents.quality_agent import QualityAgent
from agents.reliability_agent import ReliabilityAgent
from app.config import settings
from app.ml_adapter import REGISTRY_PATH, _load_bundle, artifacts_ready
from data_layer.simulator import Simulator, get_simulator
from llm.formatter import Formatter
from llm.gigachat_client import GigaChatClient
from tracing.logger import AgentTracer

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _simulator() -> Simulator:
    features = settings.data_dir / "ml_features.parquet"
    if not features.exists():
        raise RuntimeError(
            f"ml_features.parquet не найден в {settings.data_dir}. "
            "Запустите 'make train' для сборки causal-признаков."
        )
    return get_simulator(settings.data_dir)


def simulator_dep() -> Simulator:
    try:
        return _simulator()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@lru_cache(maxsize=1)
def _tracer() -> AgentTracer:
    return AgentTracer(settings.tracing_db_path)


def tracer_dep() -> AgentTracer:
    return _tracer()


@lru_cache(maxsize=1)
def _gigachat_client() -> GigaChatClient:
    return GigaChatClient(
        credentials=settings.gigachat_credentials,
        model=settings.gigachat_model,
        verify_ssl=settings.gigachat_verify_ssl,
        cache_ttl_seconds=300,
    )


def gigachat_client_dep() -> GigaChatClient:
    return _gigachat_client()


@lru_cache(maxsize=1)
def _models():
    """Возвращает (avt, hydro, anomaly, blending, optimizer) или None (mock-режим)."""
    if not artifacts_ready():
        return None
    try:
        return _load_bundle()
    except Exception:
        log.exception("failed to load ML artifacts; falling back to mock agents")
        return None


@lru_cache(maxsize=1)
def _data_agent() -> DataAgent:
    models = _models()
    if models is not None:
        _avt, _hydro, anomaly, _blending, _optimizer = models
        return DataAgent(anomaly=anomaly, simulator=_simulator())
    return DataAgent(simulator=_simulator())


@lru_cache(maxsize=1)
def _quality_agent() -> QualityAgent:
    models = _models()
    if models is not None:
        avt, hydro, _anomaly, blending, _optimizer = models
        return QualityAgent(avt=avt, hydro=hydro, blending=blending)
    return QualityAgent()


def quality_agent_dep() -> QualityAgent:
    return _quality_agent()


@lru_cache(maxsize=1)
def _reliability_agent() -> ReliabilityAgent:
    return ReliabilityAgent(registry_path=REGISTRY_PATH)


@lru_cache(maxsize=1)
def _optimization_agent() -> OptimizationAgent:
    models = _models()
    if models is not None:
        _avt, _hydro, _anomaly, _blending, optimizer = models
        return OptimizationAgent(optimizer=optimizer)
    return OptimizationAgent()


@lru_cache(maxsize=1)
def _orchestrator() -> Orchestrator:
    formatter = Formatter(client=_gigachat_client())
    return Orchestrator(
        _data_agent(),
        _quality_agent(),
        _reliability_agent(),
        _optimization_agent(),
        formatter=formatter,
        tracer=_tracer(),
    )


def orchestrator_dep() -> Orchestrator:
    try:
        return _orchestrator()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


SimulatorDep = Annotated[Simulator, Depends(simulator_dep)]
TracerDep = Annotated[AgentTracer, Depends(tracer_dep)]
QualityAgentDep = Annotated[QualityAgent, Depends(quality_agent_dep)]
GigaChatClientDep = Annotated[GigaChatClient, Depends(gigachat_client_dep)]
OrchestratorDep = Annotated[Orchestrator, Depends(orchestrator_dep)]
