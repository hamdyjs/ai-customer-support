"""Check eval/golden_dataset.json, the reference answers for the assistant cases.

Each golden case repeats its case from eval/assistant_cases.json unchanged and adds
what a correct answer contains. Facts are judged on meaning, not wording. Details in
parentheses, such as record IDs, amounts, dates and workings, identify or back up a
fact and don't have to appear in the reply.

- expected_facts: every one must come across for the answer to be complete.
- optional_facts: correct, helpful extras. Credit them, but don't require them.
- prohibited_claims: any one of these makes the answer wrong, however it's worded.
- cannot_establish: what the records and help articles can't settle. The answer
  should say so or leave it alone, never guess.
- clarifying_question_acceptable: allows a second kind of reply, without making the
  expected facts optional. A reply that asks a clarifying question instead of
  answering doesn't need the expected facts. A reply that answers needs all of them.
  Either way, it must make no prohibited claim.

Each fact lists its sources: a tool name for the customer's own records, or an article
ID for a help article. A tool source includes the reading rules in the tool's
description in app/tools.py, such as a failed payment attempt not charging the
customer. Required facts may only rely on the case's expected tools and articles, so
the two can't contradict each other.

It makes no API calls. Run it from the repo root:

    uv run python -m scripts.validate_golden_dataset
"""

import sys
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from scripts.evaluate_assistant import (
    CASES_FILE,
    TOOL_NAMES,
    AssistantCase,
    EvaluationError,
    load_cases,
)
from scripts.ingest import KNOWLEDGE_BASE_DIR

GOLDEN_FILE = CASES_FILE.with_name("golden_dataset.json")
ARTICLES = {path.stem for path in KNOWLEDGE_BASE_DIR.glob("*.md")}

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Text
    sources: list[str] = Field(min_length=1)  # tool names or article IDs


class GoldenCase(AssistantCase):
    expected_facts: list[Fact] = Field(min_length=1)
    optional_facts: list[Fact]
    prohibited_claims: list[Text] = Field(min_length=1)
    cannot_establish: list[Text]
    clarifying_question_acceptable: bool = False


def load_golden_cases(
    path: Path = GOLDEN_FILE, cases_path: Path = CASES_FILE
) -> list[GoldenCase]:
    try:
        cases = TypeAdapter(list[GoldenCase]).validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise EvaluationError(f"Cannot load {path}: {error}") from error
    originals = {case.id: case for case in load_cases(cases_path)}

    ids = [case.id for case in cases]
    problems = [
        f"duplicate case ID {id!r}" for id in sorted(set(ids)) if ids.count(id) > 1
    ]
    problems += [
        f"{id} from {cases_path.name} is missing" for id in originals if id not in ids
    ]
    for case in cases:
        original = originals.get(case.id)
        if original is None:
            problems.append(f"{case.id} isn't in {cases_path.name}")
            continue
        for field in AssistantCase.model_fields:
            if getattr(case, field) != getattr(original, field):
                problems.append(f"{case.id}: {field} differs from {cases_path.name}")
        for fact in case.expected_facts + case.optional_facts:
            unknown = sorted(set(fact.sources) - TOOL_NAMES - ARTICLES)
            if unknown:
                problems.append(f"{case.id}: unknown sources {unknown}")
        expected = set(case.expected_tools) | set(case.expected_articles)
        for fact in case.expected_facts:
            unexpected = sorted(set(fact.sources) - expected)
            if unexpected:
                problems.append(
                    f"{case.id}: a required fact relies on {unexpected}, "
                    "which the case doesn't expect"
                )
        if case.category == "insufficient_or_ambiguous" and not case.cannot_establish:
            problems.append(f"{case.id}: doesn't say what can't be established")
    if problems:
        raise EvaluationError(f"Invalid golden cases in {path}: " + "; ".join(problems))
    return cases


def main() -> int:
    try:
        cases = load_golden_cases()
    except EvaluationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    required = sum(len(case.expected_facts) for case in cases)
    optional = sum(len(case.optional_facts) for case in cases)
    prohibited = sum(len(case.prohibited_claims) for case in cases)
    print(
        f"{GOLDEN_FILE.name} is valid: {len(cases)} cases, {required} required "
        f"facts, {optional} optional facts, {prohibited} prohibited claims."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
