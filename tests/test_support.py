from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.ai import AnalysisError, QuestionCategory, ReplyError, get_llm_service
from app.auth import get_current_customer
from app.db.database import get_session_factory
from app.db.models import AccountStatus, Customer, TicketPriority
from app.main import app
from app.routers.support import MAX_QUESTION_LENGTH

URL = "/v1/support/analyze"
MESSAGES_URL = "/v1/support/messages"
CUSTOMER = Customer(
    id="cust_001",
    name="Alex Morgan",
    email="alex@example.com",
    company_name="Acme Studio",
    account_status=AccountStatus.active,
    created_at=date(2026, 1, 15),
)


@pytest.fixture
def client(fake_llm, engine):
    with Session(engine) as session:
        session.add(Customer(**CUSTOMER.model_dump()))
        session.commit()
    app.dependency_overrides[get_current_customer] = lambda: CUSTOMER
    app.dependency_overrides[get_session_factory] = lambda: lambda: Session(engine)
    return TestClient(app)


# Empty, whitespace-only and oversized questions are rejected on both AI endpoints
# before the LLM service is called, so they cost nothing and are never stored.
@pytest.mark.parametrize("url", [URL, MESSAGES_URL], ids=["analyze", "messages"])
@pytest.mark.parametrize(
    "question",
    ["", "   \n\t", "x" * (MAX_QUESTION_LENGTH + 1)],
    ids=["empty", "whitespace", "too_long"],
)
def test_invalid_question_returns_422(client, fake_llm, url, question):
    response = client.post(url, json={"question": question})

    assert response.status_code == 422
    assert fake_llm.questions == []
    assert fake_llm.reply_calls == []


# A question at the length limit is accepted, and surrounding whitespace is removed
# before it reaches the LLM service.
def test_question_limit_is_inclusive_and_whitespace_is_stripped(client, fake_llm):
    longest = "x" * MAX_QUESTION_LENGTH

    assert client.post(URL, json={"question": longest}).status_code == 200
    assert client.post(URL, json={"question": "  Hi  "}).status_code == 200
    assert fake_llm.questions == [longest, "Hi"]


# A temporary Claude failure returns a 503 with Retry-After on both endpoints, so
# clients know a retry can help.
@pytest.mark.parametrize(
    ("url", "error_class"),
    [(URL, AnalysisError), (MESSAGES_URL, ReplyError)],
    ids=["analyze", "messages"],
)
def test_temporary_failure_returns_503_with_retry_after(
    client, fake_llm, url, error_class
):
    fake_llm.error = error_class("Temporarily unavailable", retry_after="12")

    response = client.post(url, json={"question": "Hi"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "12"
    assert response.json() == {"detail": "Temporarily unavailable"}


# A permanent Claude failure returns a 502 without Retry-After on both endpoints.
@pytest.mark.parametrize(
    ("url", "error_class"),
    [(URL, AnalysisError), (MESSAGES_URL, ReplyError)],
    ids=["analyze", "messages"],
)
def test_permanent_failure_returns_502(client, fake_llm, url, error_class):
    fake_llm.error = error_class("The Claude API request failed")

    response = client.post(url, json={"question": "Hi"})

    assert response.status_code == 502
    assert "Retry-After" not in response.headers


# Without an API key, both endpoints return a clear 503 that says how to fix it,
# instead of crashing with a 500 when Claude is called.
@pytest.mark.parametrize("url", [URL, MESSAGES_URL], ids=["analyze", "messages"])
def test_missing_api_key_returns_a_clear_error(client, monkeypatch, url):
    app.dependency_overrides.pop(get_llm_service)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.post(url, json={"question": "Hi"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "ANTHROPIC_API_KEY isn't set. Add it to your .env file."
    }


# A valid message returns 200 with the conversation ID, the reply for the customer
# and the analysis, using exactly these fields.
def test_messages_returns_the_reply_and_analysis(client):
    response = client.post(MESSAGES_URL, json={"question": "What did I pay?"})

    body = response.json()
    assert response.status_code == 200
    assert body.pop("conversation_id").startswith("conv_")
    assert body == {
        "reply": "You were charged $49.00 on November 1.",
        "analysis": {
            "category": "billing",
            "priority": "high",
            "summary": "The customer was charged twice.",
            "suggested_response": "We're looking into this for you.",
            "human_review_recommended": True,
        },
    }


# The route declares MessageResponse as its response model, so FastAPI validates the
# response and documents its shape.
def test_messages_declares_the_response_model(client):
    operation = client.get("/openapi.json").json()["paths"][MESSAGES_URL]["post"]
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]

    assert schema == {"$ref": "#/components/schemas/MessageResponse"}


# The route gives the LLM service the authenticated customer's ID, which is what
# limits the tools to that customer's data.
def test_messages_uses_the_authenticated_customer(client, fake_llm):
    client.post(MESSAGES_URL, json={"question": "What did I pay?"})

    assert fake_llm.reply_calls == [("What did I pay?", "cust_001")]


# A customer ID in the request body is ignored, so nobody can ask about another
# customer by sending their ID.
def test_messages_ignores_a_customer_id_in_the_body(client, fake_llm):
    client.post(
        MESSAGES_URL, json={"question": "What did I pay?", "customer_id": "cust_008"}
    )

    assert fake_llm.reply_calls == [("What did I pay?", "cust_001")]


# When the LLM service fails, the route returns a 502 with the error message.
def test_messages_reply_error_returns_502(client, fake_llm):
    fake_llm.error = ReplyError("The Claude API request failed")

    response = client.post(MESSAGES_URL, json={"question": "What did I pay?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The Claude API request failed"}


# An invalid body is rejected before the LLM service is called.
def test_messages_invalid_body_returns_422(client, fake_llm):
    response = client.post(MESSAGES_URL, json={})

    assert response.status_code == 422
    assert fake_llm.reply_calls == []


# A valid request returns 200 with exactly the five fields in the API response.
def test_returns_the_analysis(client):
    response = client.post(URL, json={"question": "I was charged twice"})

    assert response.status_code == 200
    assert response.json() == {
        "category": "billing",
        "priority": "high",
        "summary": "The customer was charged twice.",
        "suggested_response": "We're looking into this for you.",
        "human_review_recommended": True,
    }


# The route passes the customer's question to the LLM service unchanged.
def test_passes_the_question_to_the_llm_service(client, fake_llm):
    client.post(URL, json={"question": "I was charged twice"})

    assert fake_llm.questions == ["I was charged twice"]


# Each category reaches the customer as its plain string value, such as "billing".
@pytest.mark.parametrize("category", list(QuestionCategory))
def test_returns_every_category(client, fake_llm, category):
    fake_llm.analysis = fake_llm.analysis.model_copy(update={"category": category})

    response = client.post(URL, json={"question": "A question"})

    assert response.json()["category"] == category.value


# Each priority reaches the customer as its plain string value, such as "high".
@pytest.mark.parametrize("priority", list(TicketPriority))
def test_returns_every_priority(client, fake_llm, priority):
    fake_llm.analysis = fake_llm.analysis.model_copy(update={"priority": priority})

    response = client.post(URL, json={"question": "A question"})

    assert response.json()["priority"] == priority.value


# When the LLM service fails, the route returns a 502 with the error message
# instead of crashing with a 500.
def test_analysis_error_returns_502(client, fake_llm):
    fake_llm.error = AnalysisError("The Claude API request failed")

    response = client.post(URL, json={"question": "I was charged twice"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The Claude API request failed"}


# Bad requests are rejected with a 422 before the LLM service is called, so they
# never cost an API call. This includes the old query parameter format.
@pytest.mark.parametrize(
    "request_args",
    [
        {"json": {}},
        {"json": {"question": 123}},
        {"params": {"question": "Sent as a query parameter"}},
    ],
    ids=["missing_question", "wrong_type", "query_parameter"],
)
def test_invalid_request_returns_422_without_calling_the_llm(
    client, fake_llm, request_args
):
    response = client.post(URL, **request_args)

    assert response.status_code == 422
    assert fake_llm.questions == []
