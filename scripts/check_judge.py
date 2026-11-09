"""Check the faithfulness judge on answers whose verdicts are known.

eval/judge_examples.json takes three cases from the golden dataset. Each has the
evidence the assistant would have had, which is real output from the tools on the
seeded database and the help-article search, and two hand-written answers: one fully
backed by that evidence, and one with a single planted flaw that the file explains.
The judge gets only the question, an answer and the evidence, and should pass the
first answer and fail the second.

This makes six real calls to the judge model, so it isn't part of the test suite.
Run it from the repo root:

    uv run python -m scripts.check_judge
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from scripts.evaluate_assistant import ToolCall, format_faithfulness
from scripts.judge_faithfulness import (
    JUDGE_MODEL,
    FaithfulnessResult,
    Verdict,
    judge_faithfulness,
)

EXAMPLES_FILE = Path(__file__).resolve().parents[1] / "eval" / "judge_examples.json"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    input: dict[str, Any]
    output: Any  # what the tool returned, as JSON


class JudgeExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str  # a case in the golden dataset
    customer_id: str  # whose records the evidence holds
    question: str = Field(min_length=1)
    evidence: list[Evidence]
    supported_answer: str = Field(min_length=1)
    flawed_answer: str = Field(min_length=1)
    flaw: str = Field(min_length=1)  # what's wrong with the flawed answer

    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(
                item.tool,
                item.input,
                {
                    "type": "tool_result",
                    "tool_use_id": f"toolu_{number}",
                    "content": json.dumps(item.output),
                },
            )
            for number, item in enumerate(self.evidence, start=1)
        ]


def load_examples(path: Path = EXAMPLES_FILE) -> list[JudgeExample]:
    return TypeAdapter(list[JudgeExample]).validate_json(
        path.read_text(encoding="utf-8")
    )


def check_example(
    client: Anthropic, example: JudgeExample
) -> list[tuple[str, Verdict, FaithfulnessResult]]:
    """Judge both answers. Only the question, answer and evidence are sent."""
    checks = [
        ("supported", Verdict.passed, example.supported_answer),
        ("flawed", Verdict.failed, example.flawed_answer),
    ]
    return [
        (
            kind,
            expected,
            judge_faithfulness(client, example.question, answer, example.tool_calls()),
        )
        for kind, expected, answer in checks
    ]


def main() -> int:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY isn't set. Add it to your .env file.", file=sys.stderr)
        return 1
    client = Anthropic()
    lines = ["# Faithfulness judge check", "", f"Judge: {JUDGE_MODEL}", ""]
    matches = total = 0
    for example in load_examples():
        print(f"Judging {example.case_id}", file=sys.stderr)
        for kind, expected, result in check_example(client, example):
            total += 1
            matches += result.verdict == expected
            mark = "as expected" if result.verdict == expected else "NOT AS EXPECTED"
            lines += [
                f"## {example.case_id}: {kind} answer",
                "",
                f"Expected {expected}, judged {result.verdict}: {mark}.",
                "",
            ]
            if kind == "flawed":
                lines += [f"**Planted flaw:** {example.flaw}", ""]
            lines += format_faithfulness(result)
    lines.insert(4, f"{matches} of {total} answers judged as expected.\n")
    print("\n".join(lines))
    return 0 if matches == total else 1


if __name__ == "__main__":
    sys.exit(main())
