from fastapi import APIRouter

router = APIRouter(prefix="/support", tags=["support"])


@router.post("/analyze")
async def message(question: str):
    pass
