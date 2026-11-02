from fastapi import APIRouter, Depends, FastAPI

from app.auth import get_current_customer
from app.routers import support

app = FastAPI()

v1_router = APIRouter(prefix="/v1", dependencies=[Depends(get_current_customer)])
v1_router.include_router(support.router)

app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
