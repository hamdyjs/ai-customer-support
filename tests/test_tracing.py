import json
from datetime import date
from functools import partial
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse, LangfuseSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from sqlmodel import Session

from app import tracing
from app.ai import ReplyError
from app.auth import get_current_customer
from app.db.database import get_session_factory
from app.db.models import AccountStatus, Customer
from app.main import app

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
