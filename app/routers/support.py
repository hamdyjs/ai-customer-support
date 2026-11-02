from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.ai import (
    AnalysisError,
    LLMServiceDep,
    QuestionAnalysis,
    ReplyError,
    SupportReply,
)
from app.auth import CurrentCustomerDep
from app.db.database import SessionDep

router = APIRouter(prefix="/support", tags=["support"])


class AIRequest(BaseModel):
    question: str


@router.post("/analyze")
async def analyze(request: AIRequest, llm: LLMServiceDep) -> QuestionAnalysis:
    try:
        return await llm.analyze_question(request.question)
    except AnalysisError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error


@router.post("/messages")
async def messages(
    request: AIRequest,
    customer: CurrentCustomerDep,
    session: SessionDep,
    llm: LLMServiceDep,
) -> SupportReply:
    try:
        return await llm.generate_reply(request.question, session, customer.id)
    except ReplyError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
