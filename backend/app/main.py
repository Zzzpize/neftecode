import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import ask, health, history, recommend, scenarios, state, trace

log = logging.getLogger("neftecode")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.deps import _orchestrator, _simulator

    async def warmup():
        try:
            _simulator()
            _orchestrator()
            log.info("warmup complete: simulator + orchestrator ready")
        except Exception:
            log.exception("warmup failed; endpoints will lazy-load on first hit")

    asyncio.create_task(warmup())
    yield


app = FastAPI(title="Neftecode API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(ask.router)
app.include_router(state.router)
app.include_router(trace.router)
app.include_router(history.router)
app.include_router(scenarios.router)
