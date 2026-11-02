import pytest
from fastapi.testclient import TestClient

from app.ai import AnalysisError, QuestionCategory, ReplyError
from app.auth import get_current_customer
from app.db.database import get_session
from app.db.models import Customer, TicketPriority
from app.main import app

URL = "/v1/support/analyze"
MESSAGES_URL = "/v1/support/messages"


@pytest.fixture
def client(fake_llm):
    app.dependency_overrides[get_current_customer] = lambda: Customer(id="cust_001")
    app.dependency_overrides[get_session] = lambda: None
    return TestClient(app)


# A valid message returns 200 with the LLM service's reply.
def test_messages_returns_the_reply(client):
    response = client.post(MESSAGES_URL, json={"question": "What did I pay?"})

    assert response.status_code == 200
    assert response.json() == {"message": "You were charged $49.00 on November 1."}


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
