"""Run the support assistant end to end on the cases in eval/assistant_cases.json.

Each question goes through the real API route, authentication, tool loop, tools,
help-article search and storage, with real Claude calls. That costs a few cents, so
it isn't part of the test suite. It runs on a temporary, freshly seeded database and
leaves app.db alone. Qdrant must be running with the help articles stored.

Run it from the repo root and save the report:

    uv run python -m scripts.evaluate_assistant > assistant_report.md

Progress and the app's own console output go to stderr, so the report stays clean.
Add --case with a case ID, once or more, to run only those cases.

The report puts each answer, the tools used, what the searches returned and what
was cited next to the case's expected behavior. It doesn't score the answers:
judging them is left to the reader.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

from anthropic.types import ToolResultBlockParam, ToolUseBlock
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from qdrant_client.http.exceptions import ApiException
from sqlmodel import Session

from app.retrieval import (
    COLLECTION_NAME,
    QDRANT_URL,
    RetrievalError,
    get_qdrant_client,
)
from app.tools import TOOL_DEFINITIONS

CASES_FILE = Path(__file__).resolve().parents[1] / "eval" / "assistant_cases.json"
MESSAGES_URL = "/v1/support/messages"
TOOL_NAMES = {definition["name"] for definition in TOOL_DEFINITIONS}
CATEGORY_LABELS = {
    "knowledge_base": "Knowledge base",
    "account": "Account",
    "knowledge_base_and_account": "Knowledge base and account",
    "insufficient_or_ambiguous": "Insufficient evidence or ambiguous",
    "multi_part": "Multi-part",
}
REVIEW_CHECKLIST = (
    "- [ ] Answer is correct and answers every part\n"
    "- [ ] Every claim is grounded in the tool results\n"
    "- [ ] Each citation supports a claim in the reply\n"
    "- [ ] Missing or unclear information is handled well\n"
    "- [ ] The right tools were used"
)


class AssistantCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Literal[
        "knowledge_base",
        "account",
        "knowledge_base_and_account",
        "insufficient_or_ambiguous",
        "multi_part",
    ]
    customer_id: str  # one of the seeded customers, who asks the question
    question: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)
    expected_tools: list[str]
    expected_articles: list[str]  # article file names in knowledge_base/, no .md


class EvaluationError(Exception):
    pass


@dataclass
class ToolCall:
    name: str
    input: dict[str, object]
    result: ToolResultBlockParam  # exactly what was sent back to Claude


@dataclass
class Search:
    query: str
    passages: list[dict[str, Any]]
    error: str | None = None


@dataclass
class CaseResult:
    case: AssistantCase
    status_code: int
    body: dict[str, Any]  # the API's JSON response
    tool_calls: list[ToolCall]
    elapsed_seconds: float

    def searches(self) -> list[Search]:
        searches = []
        for call in self.tool_calls:
            if call.name != "search_knowledge_base":
                continue
            query = str(call.input.get("query", ""))
            content = call.result.get("content")
            if call.result.get("is_error") or not isinstance(content, str):
                searches.append(Search(query, [], str(content)))
            else:
                searches.append(Search(query, json.loads(content)))
        return searches


class ToolRecorder:
    """Wraps the app's run_tool_call to note each call and its result.

    It runs the real function and returns its result unchanged, so the tool loop
    behaves exactly as it does for customers.
    """

    def __init__(
        self,
        run_tool_call: Callable[[ToolUseBlock, Session, str], ToolResultBlockParam],
    ):
        self.run_tool_call = run_tool_call
        self.calls: list[ToolCall] = []

    def __call__(
        self, tool_call: ToolUseBlock, session: Session, customer_id: str
    ) -> ToolResultBlockParam:
        result = self.run_tool_call(tool_call, session, customer_id)
        self.calls.append(ToolCall(tool_call.name, tool_call.input, result))
        return result


def load_cases(path: Path = CASES_FILE) -> list[AssistantCase]:
    try:
        cases = TypeAdapter(list[AssistantCase]).validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise EvaluationError(f"Cannot load {path}: {error}") from error

    ids = [case.id for case in cases]
    problems = [
        f"duplicate case ID {id!r}" for id in sorted(set(ids)) if ids.count(id) > 1
    ]
    for case in cases:
        unknown = sorted(set(case.expected_tools) - TOOL_NAMES)
        if unknown:
            problems.append(f"{case.id} expects unknown tools {unknown}")
    if not cases:
        problems.append("there are no cases")
    if problems:
        raise EvaluationError(f"Invalid cases in {path}: " + "; ".join(problems))
    return cases


def select_cases(
    cases: list[AssistantCase], ids: list[str] | None
) -> list[AssistantCase]:
    if not ids:
        return cases
    unknown = sorted(set(ids) - {case.id for case in cases})
    if unknown:
        raise EvaluationError(f"Unknown case IDs: {', '.join(unknown)}")
    return [case for case in cases if case.id in ids]


def knowledge_base_problem() -> str | None:
    """Say why the help articles can't be searched, before any money is spent."""
    try:
        exists = get_qdrant_client().collection_exists(COLLECTION_NAME)
    except (ApiException, RetrievalError) as error:
        return f"Cannot reach Qdrant at {QDRANT_URL}: {error}"
    if not exists:
        return (
            f"The {COLLECTION_NAME} collection doesn't exist. Create it with: "
            "uv run python -m scripts.ingest"
        )
    return None


def response_body(response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {"detail": response.text}
    return body if isinstance(body, dict) else {"detail": body}


def run_cases(cases: list[AssistantCase]) -> list[CaseResult]:
    """Send each question as its customer through the real API route."""
    # Imported here, because the app picks its database from DATABASE_URL on import.
    from fastapi.testclient import TestClient

    from app import ai
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    recorder = ToolRecorder(ai.run_tool_call)
    results = []
    with patch.object(ai, "run_tool_call", recorder):
        for number, case in enumerate(cases, start=1):
            print(f"[{number}/{len(cases)}] {case.id}", file=sys.stderr)
            recorder.calls = []
            started = time.perf_counter()
            # The app prints each step of the request, which belongs with the
            # progress on stderr rather than in the report.
            with contextlib.redirect_stdout(sys.stderr):
                response = client.post(
                    MESSAGES_URL,
                    json={"question": case.question},
                    headers={"Authorization": f"Bearer {case.customer_id}"},
                )
            elapsed = time.perf_counter() - started
            results.append(
                CaseResult(
                    case,
                    response.status_code,
                    response_body(response),
                    recorder.calls,
                    elapsed,
                )
            )
    return results


def quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def describe_passage(passage: dict[str, Any]) -> str:
    heading = passage.get("section_heading") or "(introduction)"
    return f"{passage['title']} › {heading}"


def tools_line(case: AssistantCase, calls: list[ToolCall]) -> str:
    names = [call.name for call in calls]
    used = list(dict.fromkeys(names))
    parts = [
        f"{name} ({names.count(name)})" if names.count(name) > 1 else name
        for name in used
    ]
    line = ", ".join(parts) or "none"
    missing = [name for name in case.expected_tools if name not in used]
    extra = [name for name in used if name not in case.expected_tools]
    if missing:
        line += f" · not used: {', '.join(missing)}"
    if extra:
        line += f" · not expected: {', '.join(extra)}"
    return line


def format_case(result: CaseResult) -> str:
    case = result.case
    searches = result.searches()
    passages = {
        passage["id"]: passage for search in searches for passage in search.passages
    }
    lines = [
        f"## {case.id}",
        "",
        f"- **Category:** {CATEGORY_LABELS[case.category]}",
        f"- **Customer:** {case.customer_id}",
        f"- **Time:** {result.elapsed_seconds:.1f} s",
        f"- **Expected tools:** {', '.join(case.expected_tools) or 'none'}",
        f"- **Tools used:** {tools_line(case, result.tool_calls)}",
        f"- **Expected articles:** {', '.join(case.expected_articles) or 'none'}",
        "",
        f"**Question:** {case.question}",
        "",
        f"**Expected behavior:** {case.expected_behavior}",
        "",
        "**Searches:**",
        "",
    ]
    for number, search in enumerate(searches, start=1):
        lines.append(f"{number}. `{search.query}`")
        if search.error is not None:
            lines.append(f"   - Failed: {search.error}")
        for passage in search.passages:
            lines.append(f"   - `{passage['id']}` {describe_passage(passage)}")
    if not searches:
        lines.append("None")
    lines.append("")

    if result.status_code != 200:
        detail = result.body.get("detail", result.body)
        lines += [f"**Request failed:** HTTP {result.status_code}: {detail}", ""]
    else:
        analysis = result.body["analysis"]
        review = "yes" if analysis["human_review_recommended"] else "no"
        lines += [
            "**Reply:**",
            "",
            quote(result.body["reply"]),
            "",
            "**Citations:**",
            "",
        ]
        for chunk_id in result.body["citations"]:
            cited = passages.get(chunk_id)
            if cited is None:
                lines.append(f"- `{chunk_id}` (not in this case's search results)")
                continue
            # The text starts with the title and heading, which are shown above it.
            content = cited["text"].split("\n\n", 1)[-1]
            lines += [
                f"- `{chunk_id}` {describe_passage(cited)}",
                "",
                quote(content),
                "",
            ]
        if not result.body["citations"]:
            lines += ["None", ""]
        lines += [
            (
                f"**Analysis:** {analysis['category']}, "
                f"{analysis['priority']} priority, human review: {review}. "
                f"{analysis['summary']}"
            ),
            "",
        ]
    lines += ["**Review:**", "", REVIEW_CHECKLIST, ""]
    return "\n".join(lines)


def format_report(results: list[CaseResult], started_at: datetime, model: str) -> str:
    failed = sum(result.status_code != 200 for result in results)
    total_seconds = sum(result.elapsed_seconds for result in results)
    lines = [
        "# Support assistant evaluation",
        "",
        (
            f"Run at {started_at:%Y-%m-%d %H:%M} UTC with {model}: "
            f"{len(results)} cases, {failed} failed requests, "
            f"{total_seconds:.1f} s in total."
        ),
        "",
        (
            "Nothing here is scored. For each case, compare the reply with the "
            "expected behavior and tick the review list."
        ),
        "",
        "| Case | Category | Customer | HTTP | Tools used | Citations | Time |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        names = list(dict.fromkeys(call.name for call in result.tool_calls))
        citations = len(result.body.get("citations", []))
        lines.append(
            f"| {result.case.id} | {CATEGORY_LABELS[result.case.category]} "
            f"| {result.case.customer_id} | {result.status_code} "
            f"| {', '.join(names) or 'none'} | {citations} "
            f"| {result.elapsed_seconds:.1f} s |"
        )
    lines.append("")
    return "\n".join(lines) + "\n" + "\n".join(format_case(r) for r in results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the support assistant on the cases in eval/ with real Claude."
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        metavar="ID",
        help="only run this case, and repeat it to run more",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        cases = select_cases(load_cases(), args.case_ids)
    except EvaluationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY isn't set. Add it to your .env file.", file=sys.stderr)
        return 1
    problem = knowledge_base_problem()
    if problem is not None:
        print(f"Error: {problem}", file=sys.stderr)
        return 1

    started_at = datetime.now(UTC)
    with tempfile.TemporaryDirectory() as folder:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(folder) / 'evaluation.db'}"
        subprocess.run(
            [sys.executable, "-m", "app.db.seed"], check=True, stdout=sys.stderr
        )
        results = run_cases(cases)

    from app.ai import MODEL

    print(format_report(results, started_at, MODEL), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
