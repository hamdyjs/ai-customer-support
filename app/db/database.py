import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy import event
from sqlmodel import Session, create_engine

DEFAULT_DATABASE_URL = f"sqlite:///{Path(__file__).resolve().parents[2] / 'app.db'}"

engine = create_engine(
    os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, _connection_record):
    dbapi_connection.execute("PRAGMA foreign_keys=ON")


SessionFactory = Callable[[], Session]


# Keep database work short: open a session, do the work, and close it before any
# Claude call. Closing is what returns the connection to the pool, so a session must
# never stay open across an await on Claude.
def get_session_factory() -> SessionFactory:
    return lambda: Session(engine)


SessionFactoryDep = Annotated[SessionFactory, Depends(get_session_factory)]


# For routes that only touch the database and never await Claude.
def get_session(new_session: SessionFactoryDep) -> Iterator[Session]:
    with new_session() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
