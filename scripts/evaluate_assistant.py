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
judging them is left to the reader. Add --judge to also have Claude judge whether
each reply is faithful to its tool results (see scripts/judge_faithfulness.py). That
is one more Claude call per case on a pricier model, so it only runs when asked.
Add --correctness to have the same model check each reply against the golden
dataset's expectations (see scripts/judge_correctness.py), with its own verdict and
another call per case.

Evaluation requests are never traced in Langfuse, so they don't mix with real
customers' traces.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import patch

from anthropic import Anthropic
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
from scripts.judge_correctness import CorrectnessResult, Finding, judge_correctness
from scripts.judge_faithfulness import (
    JUDGE_MODEL,
    ClaimVerdict,
    FaithfulnessResult,
    Verdict,
    judge_faithfulness,
)

# Only for type hints: app.ai picks its database on import, so run_cases imports it.
if TYPE_CHECKING:
    from app.ai import RequestUsage
    from scripts.validate_golden_dataset import GoldenCase

JudgeResult = FaithfulnessResult | CorrectnessResult
PASS_RULES = {
    "Faithfulness": "A reply passes when every claim in it is backed by its tool results.",
    "Correctness": (
        "A reply passes when it gets every required fact across, or asks a clarifying "
        "question where the case accepts one, and makes no prohibited, unestablished "
        "or incorrect claim."
    ),
}

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
    usage: "RequestUsage | None" = None  # the assistant's Claude calls for the case
    faithfulness: FaithfulnessResult | None = None  # set only when --judge is used
    correctness: CorrectnessResult | None = None  # set only with --correctness

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

    from app import ai, tracing
    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    recorder = ToolRecorder(ai.run_tool_call)
    add_usage = ai.RequestUsage.add
    case_usage = ai.RequestUsage()

    # Each Claude call still counts toward its request as usual, and also toward the
    # case's own totals for the cost report.
    def add_to_case(usage: "RequestUsage", response: Any) -> None:
        add_usage(usage, response)
        add_usage(case_usage, response)

    results = []
    # Without Langfuse, evaluation requests are never traced, even when .env has its
    # keys, so they can't mix with real customers' traces.
    with (
        patch.object(ai, "run_tool_call", recorder),
        patch.object(ai.RequestUsage, "add", add_to_case),
        patch.object(tracing, "get_langfuse", return_value=None),
    ):
        for number, case in enumerate(cases, start=1):
            print(f"[{number}/{len(cases)}] {case.id}", file=sys.stderr)
            recorder.calls = []
            case_usage = ai.RequestUsage()
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
                    case_usage,
                )
            )
    return results


def judge_results(results: list[CaseResult], client: Anthropic) -> None:
    """Have Claude judge each reply against the tool results behind it.

    The judge only gets the question, the reply and the tool results, never the
    case's expected behavior or any golden data. Failed requests have no reply, so
    they aren't judged.
    """
    answered = [result for result in results if result.status_code == 200]
    for number, result in enumerate(answered, start=1):
        print(f"[judge {number}/{len(answered)}] {result.case.id}", file=sys.stderr)
        result.faithfulness = judge_faithfulness(
            client, result.case.question, result.body["reply"], result.tool_calls
        )
    print_judge_usage("faithfulness", [result.faithfulness for result in answered])


def judge_correctness_results(
    results: list[CaseResult], golden: dict[str, "GoldenCase"], client: Anthropic
) -> None:
    """Have Claude check each reply against its case's golden expectations.

    This judge gets the question, the reply and the golden expectations, never the
    tool results or the faithfulness verdict. Failed requests aren't judged.
    """
    answered = [result for result in results if result.status_code == 200]
    for number, result in enumerate(answered, start=1):
        print(
            f"[correctness {number}/{len(answered)}] {result.case.id}", file=sys.stderr
        )
        result.correctness = judge_correctness(
            client, golden[result.case.id], result.body["reply"]
        )
    print_judge_usage("correctness", [result.correctness for result in answered])


def print_judge_usage(name: str, judgments: Sequence[JudgeResult | None]) -> None:
    judged = [judgment for judgment in judgments if judgment]
    print(
        f"The {name} judge used {sum(j.input_tokens for j in judged)} input and "
        f"{sum(j.output_tokens for j in judged)} output tokens, about "
        f"${sum(j.estimated_cost() for j in judged):.4f}.",
        file=sys.stderr,
    )


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
        if result.faithfulness is not None:
            lines += format_faithfulness(result.faithfulness)
        if result.correctness is not None:
            lines += format_correctness(result.correctness)
    lines += ["**Review:**", "", REVIEW_CHECKLIST, ""]
    return "\n".join(lines)


# The verdict with a count of each label, then the claims that need a look: the ones
# that failed, and the ones where the reply said it couldn't confirm something.
def format_faithfulness(faithfulness: FaithfulnessResult) -> list[str]:
    if faithfulness.verdict == Verdict.error:
        return [f"**Faithfulness:** error. {faithfulness.summary}", ""]
    lines = [
        (
            f"**Faithfulness:** {faithfulness.verdict} "
            f"({claim_breakdown(faithfulness)}). {faithfulness.summary}"
        ),
        "",
    ]
    flagged = [
        claim
        for claim in faithfulness.claims
        if claim.verdict != ClaimVerdict.supported
    ]
    for claim in flagged:
        label = claim.verdict.capitalize()
        lines.append(f"- **{label}:** {claim.claim} ({claim.explanation})")
    if flagged:
        lines.append("")
    return lines


def claim_breakdown(faithfulness: FaithfulnessResult) -> str:
    counts = Counter(claim.verdict for claim in faithfulness.claims)
    breakdown = ", ".join(
        f"{counts[label]} {label}" for label in ClaimVerdict if counts[label]
    )
    return breakdown or "no factual claims"


def faithfulness_cell(result: CaseResult) -> str:
    if result.faithfulness is None:
        return "not judged"
    return result.faithfulness.verdict


def faithfulness_column(result: CaseResult) -> str:
    faithfulness = result.faithfulness
    if faithfulness is None or faithfulness.verdict == Verdict.error:
        return faithfulness_cell(result)
    return f"{faithfulness.verdict} ({claim_breakdown(faithfulness)})"


# The golden expectations a reply missed or broke, labelled by kind.
def correctness_findings(correctness: CorrectnessResult) -> list[tuple[str, Finding]]:
    groups = [
        ("Missing", correctness.missing_facts),
        ("Prohibited claim", correctness.prohibited_claims),
        ("Answered what can't be established", correctness.unestablished),
        ("Incorrect", correctness.incorrect_claims),
    ]
    return [(label, finding) for label, findings in groups for finding in findings]


def format_correctness(correctness: CorrectnessResult) -> list[str]:
    if correctness.verdict == Verdict.error:
        return [f"**Correctness:** error. {correctness.summary}", ""]
    clarified = " with an accepted clarifying question" if correctness.clarified else ""
    lines = [
        f"**Correctness:** {correctness.verdict}{clarified}. {correctness.summary}",
        "",
    ]
    findings = correctness_findings(correctness)
    for label, finding in findings:
        lines.append(f"- **{label}:** {finding.expectation} ({finding.explanation})")
    if findings:
        lines.append("")
    return lines


def correctness_column(result: CaseResult) -> str:
    correctness = result.correctness
    if correctness is None:
        return "not judged"
    if correctness.verdict == Verdict.error:
        return correctness.verdict
    counts = Counter(label for label, _ in correctness_findings(correctness))
    details = [f"{count} {label.lower()}" for label, count in counts.items()]
    if correctness.clarified:
        details.insert(0, "clarifying question")
    return f"{correctness.verdict} ({', '.join(details)})" if details else "pass"


def money(dollars: float) -> str:
    return f"${dollars:.4f}"


# Each judge that ran, with its result for every case, in report order.
def judges_used(
    results: list[CaseResult],
) -> list[tuple[str, list[JudgeResult | None]]]:
    judges: list[tuple[str, list[JudgeResult | None]]] = [
        ("Faithfulness", [result.faithfulness for result in results]),
        ("Correctness", [result.correctness for result in results]),
    ]
    return [(name, judgments) for name, judgments in judges if any(judgments)]


# Errors and costs are kept apart from the verdicts: an error means a reply or a
# judgment is missing, not that a reply was unfaithful or wrong.
def format_errors_and_costs(results: list[CaseResult], model: str) -> list[str]:
    judges = judges_used(results)
    errors = []
    for index, result in enumerate(results):
        if result.status_code != 200:
            detail = result.body.get("detail", result.body)
            errors.append(
                f"- {result.case.id}: the assistant request failed with HTTP "
                f"{result.status_code}: {detail}"
            )
        for name, judgments in judges:
            judgment = judgments[index]
            if judgment and judgment.verdict == Verdict.error:
                errors.append(
                    f"- {result.case.id}: {name.lower()} judge: {judgment.summary}"
                )
    lines = [
        "## API errors and costs",
        "",
        (
            "Errors are counted apart from faithfulness and correctness failures, "
            "and the judges don't count toward the assistant's cost."
        ),
        "",
        "**API errors:**",
        "",
        *(errors or ["None"]),
        "",
    ]

    usages = [result.usage for result in results if result.usage]
    lines.append(
        f"**Assistant ({model}):** {sum(u.llm_calls for u in usages)} Claude calls, "
        f"{sum(u.input_tokens for u in usages):,} input and "
        f"{sum(u.output_tokens for u in usages):,} output tokens, "
        f"{sum(u.cache_read_tokens for u in usages):,} cache reads and "
        f"{sum(u.cache_write_tokens for u in usages):,} cache writes: about "
        f"{money(sum(u.estimated_cost() for u in usages))}."
    )
    header = "| Case | Assistant calls | Assistant cost |"
    divider = "| --- | --- | --- |"
    for name, judgments in judges:
        done = [judgment for judgment in judgments if judgment]
        lines.append(
            f"\n**{name} judge ({JUDGE_MODEL}):** {len(done)} judgments, "
            f"{sum(j.input_tokens for j in done):,} input and "
            f"{sum(j.output_tokens for j in done):,} output tokens: about "
            f"{money(sum(j.estimated_cost() for j in done))}."
        )
        header += f" {name} judge cost |"
        divider += " --- |"
    lines += ["", header, divider]
    for index, result in enumerate(results):
        usage = result.usage
        row = (
            f"| {result.case.id} | {usage.llm_calls if usage else '-'} "
            f"| {money(usage.estimated_cost()) if usage else '-'} |"
        )
        for _, judgments in judges:
            judgment = judgments[index]
            row += f" {money(judgment.estimated_cost()) if judgment else '-'} |"
        lines.append(row)
    lines += [
        "",
        (
            "Costs are estimates from the list prices in app/ai.py and "
            "scripts/judge_faithfulness.py."
        ),
        "",
    ]
    return lines


def format_report(results: list[CaseResult], started_at: datetime, model: str) -> str:
    failed = sum(result.status_code != 200 for result in results)
    total_seconds = sum(result.elapsed_seconds for result in results)
    judges = judges_used(results)
    lines = [
        "# Support assistant evaluation",
        "",
        (
            f"Run at {started_at:%Y-%m-%d %H:%M} UTC with {model}: "
            f"{len(results)} cases, {failed} failed requests, "
            f"{total_seconds:.1f} s in total."
        ),
        "",
    ]
    for name, judgments in judges:
        verdicts = Counter(j.verdict if j else "not judged" for j in judgments)
        lines += [
            (
                f"{name} judged by {JUDGE_MODEL}: {verdicts[Verdict.passed]} "
                f"passed, {verdicts[Verdict.failed]} failed, "
                f"{verdicts[Verdict.error]} judge errors, "
                f"{verdicts['not judged']} not judged. {PASS_RULES[name]}"
            ),
            "",
        ]
    if judges:
        scored = " and ".join(name.lower() for name, _ in judges)
        verb = "are" if len(judges) > 1 else "is"
        lines.append(
            f"Only {scored} {verb} scored. For everything else, compare the reply "
            "with the expected behavior and tick the review list."
        )
    else:
        lines.append(
            "Nothing here is scored. For each case, compare the reply with the "
            "expected behavior and tick the review list."
        )
    header = "| Case | Category | Customer | HTTP | Tools used | Citations | Time |"
    divider = "| --- | --- | --- | --- | --- | --- | --- |"
    for name, _ in judges:
        header += f" {name} |"
        divider += " --- |"
    lines += ["", header, divider]
    for result in results:
        names = list(dict.fromkeys(call.name for call in result.tool_calls))
        citations = len(result.body.get("citations", []))
        row = (
            f"| {result.case.id} | {CATEGORY_LABELS[result.case.category]} "
            f"| {result.case.customer_id} | {result.status_code} "
            f"| {', '.join(names) or 'none'} | {citations} "
            f"| {result.elapsed_seconds:.1f} s |"
        )
        columns = {
            "Faithfulness": faithfulness_column,
            "Correctness": correctness_column,
        }
        for name, _ in judges:
            row += f" {columns[name](result)} |"
        lines.append(row)
    lines += ["", *format_errors_and_costs(results, model)]
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
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            f"also have {JUDGE_MODEL} judge whether each reply is faithful to its "
            "tool results, which costs one more Claude call per case"
        ),
    )
    parser.add_argument(
        "--correctness",
        action="store_true",
        help=(
            f"also have {JUDGE_MODEL} check each reply against the golden dataset, "
            "which costs one more Claude call per case"
        ),
    )
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        cases = select_cases(load_cases(), args.case_ids)
    except EvaluationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    golden: dict[str, GoldenCase] = {}
    if args.correctness:
        # Imported here, because it imports this module. Its EvaluationError is this
        # module's, even when this one runs as __main__.
        from scripts import validate_golden_dataset

        try:
            golden = {
                case.id: case for case in validate_golden_dataset.load_golden_cases()
            }
        except validate_golden_dataset.EvaluationError as error:
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
    if args.judge or args.correctness:
        client = Anthropic()
        if args.judge:
            judge_results(results, client)
        if args.correctness:
            judge_correctness_results(results, golden, client)

    from app.ai import MODEL

    print(format_report(results, started_at, MODEL), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
