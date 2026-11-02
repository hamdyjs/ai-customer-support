import asyncio
import json
import time

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from app.ai import (
    CLAUDE_MAX_RETRIES,
    CLAUDE_TIMEOUT_SECONDS,
    DEFAULT_RETRY_AFTER_SECONDS,
    MODEL,
    AnalysisError,
    LLMService,
    QuestionAnalysis,
    QuestionCategory,
    RequestUsage,
    create_llm_service,
    get_llm_service,
)
from app.db.models import TicketPriority

API_ERROR = {"type": "error", "error": {"type": "api_error", "message": "Failed"}}
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


# A cut-off answer is reported as cut off, rather than as invalid output.
def test_cut_off_answer_reports_the_stop_reason():
    with pytest.raises(AnalysisError, match="No analysis returned: max_tokens"):
        analyze(
            lambda request: message_response(
                [text_block('{"category": "bil')], stop_reason="max_tokens"
            )
        )


# Errors a retry can't fix, such as a bad key, are permanent: no retry hint.
@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_permanent_api_error_has_no_retry_hint(status_code):
    with pytest.raises(AnalysisError, match="API request failed") as error:
        analyze(lambda request: httpx2.Response(status_code, json=API_ERROR))

    assert error.value.retry_after is None


# Rate limits, server errors and overload are temporary, and carry a retry hint.
@pytest.mark.parametrize("status_code", [429, 500, 503, 529])
def test_temporary_api_error_has_a_default_retry_hint(status_code):
    with pytest.raises(AnalysisError, match="temporarily unavailable") as error:
        analyze(lambda request: httpx2.Response(status_code, json=API_ERROR))

    assert error.value.retry_after == DEFAULT_RETRY_AFTER_SECONDS


# When the API says how long to wait, that value is passed on to the client.
def test_retry_hint_uses_the_api_retry_after_header():
    with pytest.raises(AnalysisError) as error:
        analyze(
            lambda request: httpx2.Response(
                429, json=API_ERROR, headers={"retry-after": "12"}
            )
        )

    assert error.value.retry_after == "12"


# Failing to reach the Claude API, or a single attempt timing out, is temporary.
@pytest.mark.parametrize(
    "failure",
    [httpx2.ConnectError("Network down"), httpx2.ReadTimeout("Too slow")],
    ids=["network_down", "attempt_timeout"],
)
def test_connection_failure_is_temporary(failure):
    def handler(request):
        raise failure

    with pytest.raises(AnalysisError, match="temporarily unavailable") as error:
        analyze(handler)

    assert error.value.retry_after == DEFAULT_RETRY_AFTER_SECONDS


# The overall request limit stops a slow call even if the attempt timeout hasn't
# been reached, so one request can't hang for minutes or hours.
def test_request_limit_stops_a_slow_call(monkeypatch):
    monkeypatch.setattr("app.ai.REQUEST_TIMEOUT_SECONDS", 0.01)

    async def handler(request):
        await asyncio.sleep(5)

    started = time.monotonic()
    with pytest.raises(AnalysisError, match="temporarily unavailable") as error:
        analyze(handler)

    assert time.monotonic() - started < 2
    assert error.value.retry_after == DEFAULT_RETRY_AFTER_SECONDS


# The real client is created with the short timeout and retry settings, instead of
# the SDK's 10-minute default.
def test_client_uses_the_configured_timeout(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    create_llm_service.cache_clear()

    client = get_llm_service().client

    create_llm_service.cache_clear()
    assert client.timeout == CLAUDE_TIMEOUT_SECONDS
    assert client.max_retries == CLAUDE_MAX_RETRIES


# A rejected API key can be diagnosed from the logs: the status code is recorded,
# even though clients only see a generic message.
def test_api_failure_is_logged_with_its_status(caplog):
    with pytest.raises(AnalysisError):
        analyze(lambda request: httpx2.Response(401, json=API_ERROR))

    assert "Claude API request failed: status=401" in caplog.text


# A refusal is logged with its stop reason and refusal category.
def test_refusal_is_logged_with_its_category(caplog):
    def handler(request):
        response = message_response([], stop_reason="refusal")
        body = json.loads(response.content)
        body["stop_details"] = {
            "type": "refusal",
            "category": "cyber",
            "explanation": None,
        }
        return httpx2.Response(200, json=body)

    with pytest.raises(AnalysisError):
        analyze(handler)

    assert "stop_reason=refusal refusal_category=cyber" in caplog.text


# Invalid output from Claude is logged, not only returned as a generic error.
def test_invalid_output_is_logged(caplog):
    with pytest.raises(AnalysisError):
        analyze(lambda request: message_response([text_block("Not JSON")]))

    assert "Claude returned an invalid analysis" in caplog.text


# /analyze prints its token usage and estimated cost. A million input tokens at
# $0.10 plus a million output tokens at $0.50 comes to $0.60.
def test_analyze_prints_usage_and_estimated_cost(capsys):
    def handler(request):
        response = message_response([text_block(json.dumps(VALID_ANALYSIS))])
        body = json.loads(response.content)
        body["usage"] = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        return httpx2.Response(200, json=body)

    analyze(handler)

    output = capsys.readouterr().out
    assert "[LLM call 1] stop_reason=end_turn input_tokens=1000000" in output
    assert "[Request total] llm_calls=1" in output
    assert "estimated_cost=$0.600000" in output


# Cache reads and writes are included in the estimate at their own prices.
def test_estimated_cost_includes_cache_tokens():
    usage = RequestUsage(cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)

    assert usage.estimated_cost() == pytest.approx(0.135)
