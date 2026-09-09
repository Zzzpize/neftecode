"""DI-контейнер: ленивая инициализация shared-объектов."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException

from app.config import settings
from data_layer.simulator import Simulator, get_simulator

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _simulator() -> Simulator:
    master = settings.data_dir / "master.parquet"
    if not master.exists():
        raise RuntimeError(
            f"master.parquet не найден в {settings.data_dir}. "
            "Запустите 'make ingest' для сборки кеша."
        )
    return get_simulator(settings.data_dir)


def simulator_dep() -> Simulator:
    try:
        return _simulator()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


SimulatorDep = Annotated[Simulator, Depends(simulator_dep)]
