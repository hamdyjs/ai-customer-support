from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.db.database import SessionFactoryDep
from app.db.models import Customer

bearer_scheme = HTTPBearer()


# The session closes before this returns, so the customer is detached: its columns
# can be read, but relationships like customer.invoices can't be loaded from it.
def get_current_customer(
    new_session: SessionFactoryDep,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
) -> Customer:
    with new_session() as session:
        customer = session.get(Customer, credentials.credentials)
    if customer is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Invalid customer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return customer


CurrentCustomerDep = Annotated[Customer, Depends(get_current_customer)]
