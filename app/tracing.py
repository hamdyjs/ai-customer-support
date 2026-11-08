"""Optional request tracing with Langfuse.

Tracing is on when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Each
request is one trace, with one observation per Claude call and per tool run. A trace
records only names, durations, failures, a random request ID, the model, token
counts and the IDs of the help-article chunks a search returned, never questions,
prompts, replies, tool inputs or results, or customer details.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langfuse import Langfuse, LangfuseGeneration, LangfuseSpan, LangfuseTool
from langfuse.span_filter import is_langfuse_span

load_dotenv()

logger = logging.getLogger(__name__)

# A request's root span, a Claude call or a tool run.
Span = LangfuseSpan | LangfuseGeneration | LangfuseTool

# The current request's root span, which its Claude calls and tool runs are nested
# under. A context variable follows a request through its awaits and keeps requests
# that run at the same time apart. It's None outside a traced request.
request_span: ContextVar[Span | None] = ContextVar("request_span", default=None)


@lru_cache
def get_langfuse() -> Langfuse | None:
    if not (
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    ):
        return None
    try:
        # Only spans made through Langfuse are sent. The Anthropic SDK makes its own
        # spans, with request details, which Langfuse would otherwise send too.
        return Langfuse(should_export_span=is_langfuse_span)
    except Exception:
        logger.warning("Langfuse didn't start, so nothing is traced", exc_info=True)
        return None


class Observation:
    """An observation being recorded. Without a span, its methods do nothing."""

    def __init__(self, span: Span | None):
        self.span = span

    def update(self, **fields: Any) -> None:
        """Add fields such as token counts. Never pass content or customer details."""
        if self.span is None:
            return
        try:
            self.span.update(**fields)
        except Exception:
            logger.warning("Couldn't update the Langfuse observation", exc_info=True)


@contextmanager
def observe(parent: Langfuse | Span | None, **fields: Any) -> Iterator[Observation]:
    """Record the block as an observation under the parent, without affecting it.

    The block runs outside any Langfuse context, and every Langfuse call is guarded,
    so a tracing error never reaches the request. An exception from the block only
    marks the observation as an error, because its message could hold customer data.
    """
    observation = Observation(start_observation(parent, fields))
    failed = True
    try:
        yield observation
        failed = False
    finally:
        end_observation(observation.span, failed)


@contextmanager
def trace_request(name: str) -> Iterator[None]:
    """Record the block as one trace, with a random request ID."""
    request_id = f"req_{uuid4().hex}"
    with observe(
        get_langfuse(), name=name, metadata={"request_id": request_id}
    ) as root:
        token = request_span.set(root.span)
        try:
            yield
        finally:
            request_span.reset(token)


def trace_claude_call(model: str) -> AbstractContextManager[Observation]:
    return observe(
        request_span.get(), name="claude-call", as_type="generation", model=model
    )


def trace_tool(name: str) -> AbstractContextManager[Observation]:
    return observe(request_span.get(), name=name, as_type="tool")


def start_observation(
    parent: Langfuse | Span | None, fields: dict[str, Any]
) -> Span | None:
    if parent is None:
        return None
    try:
        return parent.start_observation(**fields)
    except Exception:
        logger.warning("Couldn't start the Langfuse observation", exc_info=True)
        return None


def end_observation(span: Span | None, failed: bool) -> None:
    if span is None:
        return
    try:
        if failed:
            span.update(level="ERROR")
        span.end()
    except Exception:
        logger.warning("Couldn't end the Langfuse observation", exc_info=True)
