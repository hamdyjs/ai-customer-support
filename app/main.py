from fastapi import APIRouter, FastAPI

from app.routers import support

app = FastAPI()

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(support.router)

app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
