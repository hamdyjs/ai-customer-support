import json

import httpx2
import pytest

from scripts.evaluate_assistant import ToolCall
from scripts.judge_faithfulness import (
    JUDGE_MODEL,
    ClaimVerdict,
    FaithfulnessResult,
    Verdict,
    judge_faithfulness,
)
from tests.fake_claude import FakeClaude, message_response, text_block

QUESTION = "Was I charged twice in October?"
REPLY = "No. Your October 1 payment failed and the October 3 one went through."
PASSAGE = {
    "id": "failed-payments:0",
    "document_id": "failed-payments",
    "title": "Failed Payments and Overdue Invoices",
    "section_heading": None,
    "source": "failed-payments.md",
    "text": "A failed payment doesn't charge you.",
}
PAYMENT = {
    "id": "pay_016",
    "invoice_id": "inv_015",
    "amount_cents": 1900,
    "status": "failed",
    "attempted_at": "2026-10-01T08:05:00Z",
    "failure_reason": "expired_card",
}
TOOL_CALLS = [
    ToolCall(
        "search_knowledge_base",
        {"query": "failed payment charge"},
        {"type": "tool_result", "tool_use_id": "t1", "content": json.dumps([PASSAGE])},
    ),
    ToolCall(
        "get_payments",
        {},
        {"type": "tool_result", "tool_use_id": "t2", "content": json.dumps([PAYMENT])},
    ),
    ToolCall(
        "get_invoices",
        {},
        {
            "type": "tool_result",
            "tool_use_id": "t3",
            "content": "Error: the lookup failed",
            "is_error": True,
        },
    ),
]


def claim(verdict, text="The October 1 payment failed."):
    return {"claim": text, "verdict": verdict, "explanation": "pay_016 failed."}


def judgment_response(*claims, summary="Every claim is backed."):
    body = {"claims": list(claims), "summary": summary}
    return message_response([text_block(json.dumps(body))], model=JUDGE_MODEL)


def judge(*responses):
    claude = FakeClaude(*responses)
    return claude, judge_faithfulness(claude.client(), QUESTION, REPLY, TOOL_CALLS)


# The judge gets the question, the reply, and every tool result exactly as the
# assistant got it, help-article passages, account records and failed lookups alike,
# with each tool's description so it reads them as the assistant would. The request
# asks the judge model for a structured judgment.
def test_judge_sees_the_reply_and_every_tool_result():
    claude, _ = judge(judgment_response())

    [request] = claude.requests
    prompt = request["messages"][0]["content"]
    assert request["model"] == JUDGE_MODEL
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert f"<question>\n{QUESTION}\n</question>" in prompt
    assert f"<reply>\n{REPLY}\n</reply>" in prompt
    assert json.dumps([PASSAGE]) in prompt
    assert json.dumps([PAYMENT]) in prompt
    assert '<tool_result tool="get_invoices" status="error">' in prompt
    assert "<input>" + json.dumps({"query": "failed payment charge"}) in prompt
    assert "A failed attempt didn't charge the customer." in prompt


# Supported claims and admitted uncertainty both pass: saying "I couldn't confirm
# that" isn't a claim that needs evidence.
def test_supported_and_uncertain_claims_pass():
    _, result = judge(
        judgment_response(
            claim("supported"),
            claim("uncertain", "It couldn't confirm when the card expires."),
        )
    )

    assert result.verdict == Verdict.passed
    assert result.summary == "Every claim is backed."
    assert [c.verdict for c in result.claims] == ["supported", "uncertain"]
    assert result.failing_claims() == []
    assert (result.input_tokens, result.output_tokens) == (1, 1)


# One unsupported or contradicted claim fails the reply, and the result names that
# claim with the judge's explanation.
@pytest.mark.parametrize("verdict", ["unsupported", "contradicted"])
def test_an_unsupported_or_contradicted_claim_fails(verdict):
    _, result = judge(
        judgment_response(
            claim("supported"),
            claim(verdict, "You were refunded $19.00."),
            summary="One claim isn't backed.",
        )
    )

    assert result.verdict == Verdict.failed
    [failing] = result.failing_claims()
    assert failing.verdict == ClaimVerdict(verdict)
    assert failing.claim == "You were refunded $19.00."
    assert failing.explanation == "pay_016 failed."


# A response that isn't a usable judgment, because it's cut off, refused, not JSON,
# or not the expected shape, becomes an error result instead of crashing the run.
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            message_response([text_block('{"claims": [')], stop_reason="max_tokens"),
            "no judgment: stop_reason=max_tokens",
        ),
        (
            message_response([], stop_reason="refusal"),
            "no judgment: stop_reason=refusal",
        ),
        (message_response([text_block("Looks fine to me.")]), "malformed judgment"),
        (
            message_response([text_block(json.dumps({"summary": "Fine."}))]),
            "malformed judgment",
        ),
        (judgment_response(claim("partly supported")), "malformed judgment"),
    ],
    ids=["max_tokens", "refusal", "not_json", "missing_claims", "unknown_label"],
)
def test_an_unusable_judgment_is_an_error(response, message):
    _, result = judge(response)

    assert result.verdict == Verdict.error
    assert message in result.summary
    assert result.claims == []


# API failures, such as a server error or no connection, become an error result too,
# so one failed judgment doesn't stop the others.
@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "x"}}
        ),
        httpx2.ConnectError("no connection"),
    ],
    ids=["server_error", "connection_error"],
)
def test_an_api_failure_is_an_error(response):
    _, result = judge(response)

    assert result.verdict == Verdict.error
    assert result.summary.startswith("The judge request failed")


# The judge's cost is estimated from its token counts at the judge model's prices,
# so the report can show it apart from the assistant's.
def test_judge_cost_comes_from_its_tokens():
    result = FaithfulnessResult(
        Verdict.passed, "Fine.", input_tokens=1_000_000, output_tokens=100_000
    )

    assert result.estimated_cost() == pytest.approx(3.0)
