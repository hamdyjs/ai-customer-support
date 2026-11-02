from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.ai import AnalysisError, LLMServiceDep, QuestionAnalysis
from app.auth import CurrentCustomerDep

router = APIRouter(prefix="/support", tags=["support"])


class AnalyzeRequest(BaseModel):
    question: str


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, llm: LLMServiceDep) -> QuestionAnalysis:
    try:
        return await llm.analyze_question(request.question)
    except AnalysisError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error


@router.post("/messages")
async def messages(
    request: AnalyzeRequest, customer: CurrentCustomerDep, llm: LLMServiceDep
):
    return {"message": "You are " + customer.name}
