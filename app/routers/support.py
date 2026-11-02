from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.ai import AnalysisError, LLMServiceDep, QuestionAnalysis

router = APIRouter(prefix="/support", tags=["support"])


class AnalyzeRequest(BaseModel):
    question: str


@router.post("/analyze")
async def analyze(request: AnalyzeRequest, llm: LLMServiceDep) -> QuestionAnalysis:
    try:
        return await llm.analyze_question(request.question)
    except AnalysisError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
