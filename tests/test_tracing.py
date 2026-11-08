import asyncio
import json
from datetime import date
from functools import partial
from uuid import uuid4

import httpx2
import pytest
from anthropic.types import Usage
from fastapi.testclient import TestClient
from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan, LangfuseTool
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from qdrant_client import QdrantClient
from sqlmodel import Session

from app import retrieval, tracing
from app.ai import MODEL, ReplyError, get_llm_service, to_usage_details
from app.auth import get_current_customer
from app.db.database import get_session_factory
from app.db.models import AccountStatus, Customer
from app.main import app
from tests.conftest import PASSAGES
from tests.fake_claude import (
    ANALYSIS,
    FakeClaude,
    message_response,
    reply_response,
    text_block,
    tool_call,
    tool_response,
)

MESSAGES_URL = "/v1/support/messages"
QUESTION = "Was I charged twice in October?"
CUSTOMER = Customer(
    id="cust_001",
    name="Alex Morgan",
    email="alex@example.com",
    company_name="Acme Studio",
    account_status=AccountStatus.active,
    created_at=date(2026, 1, 15),
)
# The only attributes a successful trace may have: the observation type, Langfuse's
# own root marker and the request ID. Its name and duration are part of the span.
TRACE_ATTRIBUTES = {
    "langfuse.observation.type",
    "langfuse.internal.is_app_root",
    "langfuse.observation.metadata.request_id",
}
# The only attributes a Claude call may have: its type, the model and token counts.
CLAUDE_CALL_ATTRIBUTES = {
    "langfuse.observation.type",
    "langfuse.observation.model.name",
    "langfuse.observation.usage_details",
}
# A search adds how many chunks it returned and their IDs to the tool's type.
SEARCH_ATTRIBUTES = {
    "langfuse.observation.type",
    "langfuse.observation.metadata.chunk_count",
    "langfuse.observation.metadata.chunk_ids",
}
LEVEL = "langfuse.observation.level"
SEARCH_QUERY = "refund on cancel"


@pytest.fixture
def client(fake_llm, engine):
    with Session(engine) as session:
        session.add(Customer(**CUSTOMER.model_dump()))
        session.commit()
    app.dependency_overrides[get_current_customer] = lambda: CUSTOMER
    app.dependency_overrides[get_session_factory] = lambda: lambda: Session(engine)
    return TestClient(app)


@pytest.fixture
def tracer_provider():
    return TracerProvider()


# Turns tracing on with made-up keys, and keeps exported spans in memory instead of
# sending them. Each test gets its own public key, because the SDK reuses one client
# per key, and its own tracer provider, so the global one is left alone.
@pytest.fixture
def exporter(monkeypatch, tracer_provider):
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"pk-lf-test-{uuid4().hex}")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(
        tracing,
        "Langfuse",
        partial(Langfuse, span_exporter=exporter, tracer_provider=tracer_provider),
    )
    yield exporter
    langfuse = tracing.get_langfuse()
    if langfuse is not None:
        langfuse.shutdown()


def exported_spans(exporter):
    langfuse = tracing.get_langfuse()
    if langfuse is not None:
        langfuse.flush()
    return exporter.get_finished_spans()


def fail(*_args, **_kwargs):
    raise RuntimeError("Langfuse is down")


# Answers the route's Claude calls with these responses, through the real LLM
# service, tool loop and tools, instead of the fake LLM service.
def use_claude(*responses):
    claude = FakeClaude(*responses)
    service = claude.service()
    app.dependency_overrides[get_llm_service] = lambda: service
    return claude


# The spans directly under the root, in the order they started.
def children_of(root, spans):
    return sorted(
        (
            span
            for span in spans
            if span.parent is not None
            and span.parent.span_id == root.context.span_id
            and span.context.trace_id == root.context.trace_id
        ),
        key=lambda span: span.start_time,
    )


def usage(span):
    return json.loads(span.attributes["langfuse.observation.usage_details"])


def metadata(span, key):
    return json.loads(span.attributes[f"langfuse.observation.metadata.{key}"])


# A Claude that waits before each answer, so requests running at the same time
# interleave. It asks for invoices first, then replies once the results come back.
class SlowClaude(FakeClaude):
    async def __call__(self, request):
        await asyncio.sleep(0.01)
        messages = json.loads(request.content)["messages"]
        if isinstance(messages[-1]["content"], list):
            return reply_response()
        return tool_response(tool_call("get_invoices"))


# Without Langfuse keys, no Langfuse client is created and messages work as before.
def test_tracing_is_off_without_keys(client, fake_llm):
    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["reply"] == fake_llm.reply.reply
    assert tracing.get_langfuse() is None


# LANGFUSE_TRACING_ENABLED=false turns tracing off even when the keys are set, so
# nothing is exported, and messages still work.
def test_tracing_can_be_turned_off_with_keys_set(client, exporter, monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    assert exported_spans(exporter) == ()


# A message is traced as one root span named after the operation, with its duration
# and a random request ID, and no question, reply, customer or conversation details.
def test_a_message_is_traced_without_its_content(client, exporter):
    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    [span] = exported_spans(exporter)
    assert span.name == "support-message"
    assert span.parent is None
    assert span.end_time > span.start_time
    assert set(span.attributes) == TRACE_ATTRIBUTES
    request_id = json.loads(span.attributes["langfuse.observation.metadata.request_id"])
    assert request_id.startswith("req_")
    assert request_id != response.json()["conversation_id"]
    assert span.events == ()


# A failed message is marked as an error, without the error's message or stack trace,
# and the customer still gets the same error response as without tracing.
def test_a_failed_message_is_traced_as_an_error(client, exporter, fake_llm):
    fake_llm.error = ReplyError("The Claude API request failed")

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 502
    assert response.json() == {"detail": "The Claude API request failed"}
    [span] = exported_spans(exporter)
    assert span.attributes["langfuse.observation.level"] == "ERROR"
    assert set(span.attributes) == TRACE_ATTRIBUTES | {"langfuse.observation.level"}
    assert span.status.description is None
    assert span.events == ()


# When Langfuse fails to start, to begin the trace or to end it, the customer still
# gets the normal reply.
@pytest.mark.parametrize(
    ("target", "name"),
    [(tracing, "Langfuse"), (Langfuse, "start_observation"), (LangfuseSpan, "end")],
    ids=["client", "start", "end"],
)
def test_langfuse_errors_dont_change_the_response(
    client, exporter, fake_llm, monkeypatch, target, name
):
    monkeypatch.setattr(target, name, fail)

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["reply"] == fake_llm.reply.reply


# Only spans made through Langfuse are sent. Spans from other libraries aren't, such
# as the one the Anthropic SDK makes for each Claude call, with request details.
def test_spans_from_other_libraries_are_not_sent(exporter, tracer_provider):
    with tracing.trace_request("support-message"):
        tracer = tracer_provider.get_tracer("com.anthropic.sdk.python")
        tracer.start_span(
            "chat", attributes={"gen_ai.request.model": "claude-haiku-5-5"}
        ).end()

    assert [span.name for span in exported_spans(exporter)] == ["support-message"]


# Every Claude call in the tool loop is its own generation under the request's trace,
# with the model and that call's token counts, and none of the prompt or response.
def test_each_claude_call_is_a_generation_with_its_token_counts(client, exporter):
    use_claude(
        tool_response(
            tool_call("get_invoices"), usage={"input_tokens": 120, "output_tokens": 7}
        ),
        tool_response(
            tool_call("get_payments"), usage={"input_tokens": 180, "output_tokens": 9}
        ),
        reply_response(usage={"input_tokens": 300, "output_tokens": 40}),
    )

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    spans = exported_spans(exporter)
    [root] = [span for span in spans if span.parent is None]
    children = children_of(root, spans)
    assert len(children) == len(spans) - 1
    calls = [span for span in children if span.name == "claude-call"]
    assert [usage(call) for call in calls] == [
        {"input": 120, "output": 7},
        {"input": 180, "output": 9},
        {"input": 300, "output": 40},
    ]
    for call in calls:
        assert set(call.attributes) == CLAUDE_CALL_ATTRIBUTES
        assert call.attributes["langfuse.observation.model.name"] == MODEL
        assert call.end_time > call.start_time
        assert call.events == ()


# A Claude call that fails is marked as an error, without the error's message, and
# the customer gets the same 503 as without tracing.
def test_a_failed_claude_call_is_marked_as_an_error(client, exporter):
    use_claude(httpx2.ConnectError("Network down"))

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 503
    [root, call] = sorted(exported_spans(exporter), key=lambda span: span.start_time)
    assert children_of(root, [call]) == [call]
    assert call.attributes[LEVEL] == "ERROR"
    assert set(call.attributes) == {
        "langfuse.observation.type",
        "langfuse.observation.model.name",
        LEVEL,
    }
    assert call.events == ()
    assert root.attributes[LEVEL] == "ERROR"


# Each tool run is a tool observation between the Claude calls, named after the tool.
# A search also records how many chunks it returned and their IDs, but not its query
# or the chunks' text, and an account tool records nothing of the customer's data.
def test_each_tool_run_is_recorded_without_its_input_or_results(
    client, exporter, knowledge_base
):
    use_claude(
        tool_response(
            tool_call("get_invoices"),
            tool_call("search_knowledge_base", "toolu_2", {"query": SEARCH_QUERY}),
        ),
        reply_response(),
    )

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    spans = exported_spans(exporter)
    [root] = [span for span in spans if span.parent is None]
    children = children_of(root, spans)
    assert [span.name for span in children] == [
        "claude-call",
        "get_invoices",
        "search_knowledge_base",
        "claude-call",
    ]
    invoices, search = children[1], children[2]
    assert dict(invoices.attributes) == {"langfuse.observation.type": "tool"}
    assert set(search.attributes) == SEARCH_ATTRIBUTES
    assert metadata(search, "chunk_count") == len(PASSAGES)
    assert metadata(search, "chunk_ids") == [passage.id for passage in PASSAGES]
    assert knowledge_base.queries == [SEARCH_QUERY]
    exported_text = json.dumps([dict(span.attributes) for span in spans])
    assert SEARCH_QUERY not in exported_text
    assert QUESTION not in exported_text


# A search that fails is marked as an error without any chunks, while the request
# still succeeds, because Claude answers around the failed tool as before.
def test_a_failed_tool_is_marked_as_an_error(
    client, exporter, knowledge_base, monkeypatch
):
    # A Qdrant without the knowledge_base collection, as before the first ingest.
    monkeypatch.setattr(
        retrieval, "get_qdrant_client", lambda: QdrantClient(":memory:")
    )
    use_claude(
        tool_response(
            tool_call("search_knowledge_base", tool_input={"query": SEARCH_QUERY})
        ),
        reply_response("Sorry, I can't check that right now."),
    )

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["reply"] == "Sorry, I can't check that right now."
    spans = exported_spans(exporter)
    [search] = [span for span in spans if span.name == "search_knowledge_base"]
    assert dict(search.attributes) == {
        "langfuse.observation.type": "tool",
        LEVEL: "ERROR",
    }
    assert search.events == ()
    [root] = [span for span in spans if span.parent is None]
    assert LEVEL not in root.attributes


# Two requests running at the same time each get only their own Claude calls and tool
# runs, because the request an observation belongs to follows it through its awaits.
def test_concurrent_requests_keep_their_own_observations(exporter, engine):
    with Session(engine) as session:
        session.add(Customer(**CUSTOMER.model_dump()))
        session.commit()
    service = SlowClaude().service()

    async def traced_request():
        with tracing.trace_request("support-message"):
            await service.generate_reply(
                QUESTION, [], lambda: Session(engine), CUSTOMER.id
            )

    async def run_both():
        await asyncio.gather(traced_request(), traced_request())

    asyncio.run(run_both())

    spans = exported_spans(exporter)
    roots = sorted(
        (span for span in spans if span.parent is None),
        key=lambda span: span.start_time,
    )
    assert len(roots) == 2
    # The second request started before the first ended, so they really overlapped.
    assert roots[1].start_time < roots[0].end_time
    for root in roots:
        assert [span.name for span in children_of(root, spans)] == [
            "claude-call",
            "get_invoices",
            "claude-call",
        ]
    assert len(spans) == 2 * 4


# With tracing off, Claude calls and tools run exactly as before and nothing is
# recorded, because there's no Langfuse client to record them with.
def test_tracing_off_leaves_claude_calls_and_tools_unchanged(client):
    claude = use_claude(tool_response(tool_call("get_invoices")), reply_response())

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["reply"] == "You paid $49.00."
    assert claude.requests[1]["messages"][-1]["content"][0]["content"] == "[]"
    assert tracing.get_langfuse() is None


# A Claude call outside a traced request, such as classifying a question, isn't
# recorded, rather than starting a trace of its own.
def test_claude_calls_outside_a_traced_request_are_not_recorded(client, exporter):
    use_claude(message_response([text_block(json.dumps(ANALYSIS))]))

    response = client.post("/v1/support/analyze", json={"question": QUESTION})

    assert response.status_code == 200
    assert exported_spans(exporter) == ()


# When Langfuse fails to start a nested observation, record its token counts or end
# a tool run, the customer still gets the normal reply.
@pytest.mark.parametrize(
    ("target", "name"),
    [
        (LangfuseSpan, "start_observation"),
        (LangfuseGeneration, "update"),
        (LangfuseTool, "end"),
    ],
    ids=["start", "update", "end"],
)
def test_nested_observation_errors_dont_change_the_response(
    client, exporter, monkeypatch, target, name
):
    use_claude(tool_response(tool_call("get_invoices")), reply_response())
    monkeypatch.setattr(target, name, fail)

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["reply"] == "You paid $49.00."


# Anthropic's cache counts are sent under the keys Langfuse prices them by, and are
# left out when the API doesn't send them. A count of zero is still a count.
@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            Usage(
                input_tokens=50,
                output_tokens=20,
                cache_creation_input_tokens=1000,
                cache_read_input_tokens=3000,
            ),
            {
                "input": 50,
                "output": 20,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 3000,
            },
        ),
        (
            Usage(input_tokens=50, output_tokens=20),
            {"input": 50, "output": 20},
        ),
        (
            Usage(
                input_tokens=50,
                output_tokens=20,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            {
                "input": 50,
                "output": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        ),
    ],
    ids=["cached", "cache_counts_missing", "cache_counts_zero"],
)
def test_usage_details_match_what_anthropic_reports(usage, expected):
    assert to_usage_details(usage) == expected


# A cached call records each kind of token once, so the total Langfuse adds up equals
# every token the call used. The thinking breakdown isn't added, because it's
# already part of the output tokens.
def test_a_cached_call_counts_each_token_once(client, exporter):
    use_claude(
        reply_response(
            usage={
                "input_tokens": 50,
                "output_tokens": 20,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 3000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 1000,
                    "ephemeral_1h_input_tokens": 0,
                },
                "output_tokens_details": {"thinking_tokens": 15},
            }
        )
    )

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    [call] = [span for span in exported_spans(exporter) if span.name == "claude-call"]
    assert usage(call) == {
        "input": 50,
        "output": 20,
        "cache_creation_input_tokens": 1000,
        "cache_read_input_tokens": 3000,
    }
    assert sum(usage(call).values()) == 50 + 20 + 1000 + 3000


# A generation records the model the API says it used, which is the one Langfuse
# prices, even when it differs from the model the app asked for.
def test_a_generation_records_the_model_the_api_used(client, exporter):
    use_claude(
        message_response(
            [
                text_block(
                    json.dumps({"reply": "Hi", "citations": [], "analysis": ANALYSIS})
                )
            ],
            model="claude-haiku-5-5-20261101",
        )
    )

    response = client.post(MESSAGES_URL, json={"question": QUESTION})

    assert response.status_code == 200
    [call] = [span for span in exported_spans(exporter) if span.name == "claude-call"]
    assert call.attributes["langfuse.observation.model.name"] == (
        "claude-haiku-5-5-20261101"
    )
