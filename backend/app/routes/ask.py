from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["llm"])


class AskRequest(BaseModel):
    decision_id: str
    question: str


class AskResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    return AskResponse(answer="LLM Q&A not wired yet.")
