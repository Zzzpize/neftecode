from fastapi import APIRouter

from app.ai_adapter import trace_to_steps
from app.deps import TracerDep
from app.schemas import TraceStep

router = APIRouter(tags=["tracing"])


@router.get("/trace/{decision_id}", response_model=list[TraceStep])
async def get_trace(decision_id: str, tracer: TracerDep) -> list[TraceStep]:
    return trace_to_steps(tracer, decision_id)
