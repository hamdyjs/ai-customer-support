from datetime import date

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.auth import get_current_customer
from app.db.database import get_session_factory
from app.db.models import AccountStatus, Customer
from app.main import app

URL = "/v1/support/analyze"
BODY = {"question": "I was charged twice"}


def make_customer(customer_id: str, email: str) -> Customer:
    return Customer(
        id=customer_id,
        name="Alex Morgan",
        email=email,
        company_name="Acme Studio",
        account_status=AccountStatus.active,
        created_at=date(2026, 1, 15),
    )


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(make_customer("cust_001", "alex@example.com"))
        session.add(make_customer("cust_008", "alex.morgan@example.com"))
        session.commit()
        yield session


@pytest.fixture
def client(session, fake_llm):
    app.dependency_overrides[get_session_factory] = lambda: (
        lambda: Session(session.get_bind())
    )
    return TestClient(app)


# A known customer ID in the Authorization header gets through to the endpoint.
def test_valid_customer_can_call_the_endpoint(client, fake_llm):
    response = client.post(
        URL, json=BODY, headers={"Authorization": "Bearer cust_001"}
    )

    assert response.status_code == 200
    assert fake_llm.questions == ["I was charged twice"]


# Requests without a valid "Bearer <known customer ID>" header get a 401, and
# never reach the LLM service.
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer cust_999"},
        {"Authorization": "Basic cust_001"},
        {"Authorization": "cust_001"},
        {"Authorization": "Bearer "},
    ],
    ids=["missing", "unknown_customer", "wrong_scheme", "no_scheme", "empty_id"],
)
def test_invalid_authorization_returns_401(client, fake_llm, headers):
    response = client.post(URL, json=BODY, headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert fake_llm.questions == []


# Two customers can share a name, so the customer must be found by ID alone.
@pytest.mark.parametrize("customer_id", ["cust_001", "cust_008"])
def test_returns_the_customer_with_that_id(session, customer_id):
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=customer_id
    )

    customer = get_current_customer(lambda: Session(session.get_bind()), credentials)

    assert customer.id == customer_id


# Every /v1 endpoint requires authentication, including ones added in the future,
# because this test finds them automatically.
def test_every_v1_endpoint_requires_authentication(client):
    paths = {
        path: operations
        for path, operations in app.openapi()["paths"].items()
        if path.startswith("/v1/")
    }

    assert paths
    for path, operations in paths.items():
        for method in operations:
            response = client.request(method, path)
            assert response.status_code == 401, f"{method} {path} is public"


# Health checks stay public because they sit outside /v1.
def test_health_does_not_require_authentication(client):
    assert client.get("/health").status_code == 200
