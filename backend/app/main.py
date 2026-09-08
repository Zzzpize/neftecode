from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import ask, health, history, recommend, scenarios, state, trace

app = FastAPI(title="Neftecode API", version="0.1.0")

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
