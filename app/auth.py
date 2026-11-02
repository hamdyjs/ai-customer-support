from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db.database import SessionDep
from app.db.models import Customer

bearer_scheme = HTTPBearer()


def get_current_customer(
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
) -> Customer:
    customer = session.get(Customer, credentials.credentials)
    if customer is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid customer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return customer


CurrentCustomerDep = Annotated[Customer, Depends(get_current_customer)]
