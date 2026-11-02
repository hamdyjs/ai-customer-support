import pytest
from fastapi.testclient import TestClient

from app.ai import AnalysisError, QuestionCategory
from app.auth import get_current_customer
from app.db.models import Customer, TicketPriority
from app.main import app

URL = "/v1/support/analyze"


@pytest.fixture
def client(fake_llm):
    app.dependency_overrides[get_current_customer] = lambda: Customer(id="cust_001")
    return TestClient(app)


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
