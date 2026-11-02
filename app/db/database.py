from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import event
from sqlmodel import Session, create_engine

engine = create_engine("sqlite:///app.db", connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, _connection_record):
    dbapi_connection.execute("PRAGMA foreign_keys=ON")


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
