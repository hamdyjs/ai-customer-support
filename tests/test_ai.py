import asyncio
import json

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from app.ai import MODEL, AnalysisError, LLMService, QuestionAnalysis, QuestionCategory
from app.db.models import TicketPriority

VALID_ANALYSIS = {
    "category": "billing",
    "priority": "high",
    "summary": "The customer was charged twice.",
    "suggested_response": "We're looking into this for you.",
    "human_review_recommended": True,
}


def text_block(text):
    return {"type": "text", "text": text}


def message_response(content, stop_reason="end_turn"):
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def analyze(handler):
    transport = httpx2.MockTransport(handler)
    client = AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=transport),
    )
    return asyncio.run(LLMService(client).analyze_question("I was charged twice"))


# The enums contain exactly the values the brief allows, so adding or removing
# one by accident fails here.
def test_allowed_categories_and_priorities():
    assert set(QuestionCategory) == {"billing", "technical", "account", "general"}
    assert set(TicketPriority) == {"low", "medium", "high"}


# A valid JSON reply from Claude becomes a QuestionAnalysis object with real
# Python types, not a plain dictionary or string.
def test_returns_the_parsed_analysis():
    analysis = analyze(
        lambda request: message_response([text_block(json.dumps(VALID_ANALYSIS))])
    )

    assert isinstance(analysis, QuestionAnalysis)
    assert analysis.category == QuestionCategory.billing
    assert analysis.human_review_recommended is True


# The request sent to Claude uses the right model and question, and includes a
# strict schema with all five fields. This is what enforces the JSON output.
def test_requests_structured_output():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return message_response([text_block(json.dumps(VALID_ANALYSIS))])

    analyze(handler)

    body = requests[0]
    schema = body["output_config"]["format"]["schema"]
    assert body["model"] == MODEL
    assert body["messages"] == [{"role": "user", "content": "I was charged twice"}]
    assert set(schema["properties"]) == set(VALID_ANALYSIS)
    assert schema["additionalProperties"] is False


# When Claude declines to answer, the service raises AnalysisError instead of
# returning an empty result.
def test_refusal_raises_analysis_error():
    with pytest.raises(AnalysisError, match="refusal"):
        analyze(lambda request: message_response([], stop_reason="refusal"))


# Unusable replies from Claude raise AnalysisError instead of an unhandled
# validation error: cut-off or non-JSON text, missing fields and invalid values.
@pytest.mark.parametrize(
    "text",
    [
        '{"category": "bil',
        "Not JSON",
        json.dumps({"category": "billing"}),
        json.dumps({**VALID_ANALYSIS, "category": "other"}),
        json.dumps({**VALID_ANALYSIS, "priority": "urgent"}),
    ],
    ids=["cut_off", "not_json", "missing_fields", "bad_category", "bad_priority"],
)
def test_invalid_output_raises_analysis_error(text):
    with pytest.raises(AnalysisError, match="invalid analysis"):
        analyze(lambda request: message_response([text_block(text)]))


# Error responses from the Claude API raise AnalysisError: a bad key (401), a
# rate limit (429) and a server error (500).
@pytest.mark.parametrize("status_code", [401, 429, 500])
def test_api_status_error_raises_analysis_error(status_code):
    error = {"type": "error", "error": {"type": "api_error", "message": "Failed"}}

    with pytest.raises(AnalysisError, match="API request failed"):
        analyze(lambda request: httpx2.Response(status_code, json=error))


# Failing to reach the Claude API at all also raises AnalysisError.
def test_connection_error_raises_analysis_error():
    def handler(request):
        raise httpx2.ConnectError("Network down")

    with pytest.raises(AnalysisError, match="API request failed"):
        analyze(handler)
