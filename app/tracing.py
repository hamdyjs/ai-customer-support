"""Optional request-level tracing with Langfuse.

Tracing is on when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. A trace
records only the operation's name, its duration, whether it failed, and a random
request ID, never questions, prompts, replies, tool results or customer details.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from uuid import uuid4

from dotenv import load_dotenv
from langfuse import Langfuse, LangfuseSpan
from langfuse.span_filter import is_langfuse_span

load_dotenv()

logger = logging.getLogger(__name__)


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


@contextmanager
def trace_request(name: str) -> Iterator[None]:
    """Record the block as one trace, without letting tracing affect it.

    The block runs outside any Langfuse context, and every Langfuse call is guarded,
    so a tracing error never reaches the request. An exception from the block only
    marks the trace as an error, because its message could hold customer data.
    """
    span = start_span(name)
    failed = True
    try:
        yield
        failed = False
    finally:
        end_span(span, failed)


def start_span(name: str) -> LangfuseSpan | None:
    langfuse = get_langfuse()
    if langfuse is None:
        return None
    try:
        return langfuse.start_observation(
            name=name, metadata={"request_id": f"req_{uuid4().hex}"}
        )
    except Exception:
        logger.warning("Couldn't start the Langfuse trace", exc_info=True)
        return None


def end_span(span: LangfuseSpan | None, failed: bool) -> None:
    if span is None:
        return
    try:
        if failed:
            span.update(level="ERROR")
        span.end()
    except Exception:
        logger.warning("Couldn't end the Langfuse trace", exc_info=True)
