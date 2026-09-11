from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.deps import (
    GigaChatClientDep,
    QualityAgentDep,
    SimulatorDep,
    TracerDep,
)
from llm.gigachat_client import LLMUnavailableError
from llm.mcp_server import MCPServer
from llm.qa_handler import QaHandler

router = APIRouter(tags=["llm"])


class AskRequest(BaseModel):
    decision_id: str
    question: str


class AskResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    sim: SimulatorDep,
    tracer: TracerDep,
    quality_agent: QualityAgentDep,
    client: GigaChatClientDep,
) -> AskResponse:
    mcp = MCPServer(simulator=sim, quality_agent=quality_agent, tracer=tracer)
    handler = QaHandler(client=client, mcp=mcp, tracer=tracer)

    try:
        result = await handler.answer(req.decision_id, req.question)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"GigaChat недоступен: {exc}") from exc

    return AskResponse(answer=result.answer)
