from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from sqlmodel import SQLModel

from app.auth import get_current_customer
from app.db.database import engine
from app.routers import support


# This only adds missing tables and never changes or removes existing ones.
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    SQLModel.metadata.create_all(engine)
    yield


app = FastAPI(lifespan=lifespan)

v1_router = APIRouter(prefix="/v1", dependencies=[Depends(get_current_customer)])
v1_router.include_router(support.router)

app.include_router(v1_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
