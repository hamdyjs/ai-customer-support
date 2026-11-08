import json
from collections import Counter
from datetime import UTC, date, datetime

import pytest
from sqlmodel import Session

from app.ai import get_llm_service
from app.db.database import get_session_factory
from app.db.models import AccountStatus, Customer
from app.main import app
from scripts.evaluate_assistant import (
    AssistantCase,
    CaseResult,
    EvaluationError,
    ToolCall,
    ToolRecorder,
    format_report,
    load_cases,
    run_cases,
    select_cases,
)
from scripts.ingest import KNOWLEDGE_BASE_DIR
from tests.fake_claude import (
    ANALYSIS,
    FakeClaude,
    reply_response,
    tool_call,
    tool_response,
)

# The seed script creates these eight customers, one per scenario.
SEEDED_CUSTOMERS = {f"cust_{number:03}" for number in range(1, 9)}
CASE = AssistantCase(
    id="both-refund",
    category="knowledge_base_and_account",
    customer_id="cust_001",
    question="Can I get a refund?",
    expected_behavior="Explains the refund policy.",
    expected_tools=["get_payments", "search_knowledge_base"],
    expected_articles=["refunds-and-duplicate-charges"],
)
PASSAGE = {
    "id": "cancellation:5",
    "document_id": "cancellation",
    "title": "Canceling Your Subscription",
    "section_heading": "Refunds after canceling",
    "source": "cancellation.md",
    "text": (
        "Canceling Your Subscription\nRefunds after canceling\n\n"
        "Canceling doesn't refund the current period."
    ),
}


def search_call(passages, is_error=False):
    result = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "Error: down" if is_error else json.dumps(passages),
    }
    if is_error:
        result["is_error"] = True
    return ToolCall("search_knowledge_base", {"query": "refund policy"}, result)


def write_cases(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    return path


# The dataset has the 12 planned cases, in the agreed mix of question types.
def test_dataset_has_the_planned_mix_of_cases():
    cases = load_cases()

    assert len(cases) == 12
    assert Counter(case.category for case in cases) == {
        "knowledge_base": 3,
        "account": 2,
        "knowledge_base_and_account": 2,
        "insufficient_or_ambiguous": 3,
        "multi_part": 2,
    }


# Every case is asked by a seeded customer and only expects articles that exist, so
# a typo can't quietly make a case meaningless.
def test_dataset_uses_seeded_customers_and_real_articles():
    for case in load_cases():
        assert case.customer_id in SEEDED_CUSTOMERS, case.id
        for article in case.expected_articles:
            assert (KNOWLEDGE_BASE_DIR / f"{article}.md").exists(), case.id


# Mistakes in the dataset stop the run before any API call: duplicate IDs, tools
# Claude doesn't have, and unknown fields or categories.
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "both-refund"}, "duplicate case ID 'both-refund'"),
        ({"expected_tools": ["get_everything"]}, "unknown tools"),
        ({"category": "other"}, "category"),
        ({"notes": "x"}, "notes"),
    ],
    ids=["duplicate_id", "unknown_tool", "unknown_category", "unknown_field"],
)
def test_load_cases_rejects_invalid_cases(tmp_path, change, message):
    case = CASE.model_dump()
    path = write_cases(tmp_path, [case, {**case, "id": "other", **change}])

    with pytest.raises(EvaluationError, match=message):
        load_cases(path)


# --case picks cases by ID in dataset order, and an unknown ID is an error rather
# than a run that silently skips it.
def test_select_cases_by_id():
    other = CASE.model_copy(update={"id": "other"})

    assert select_cases([CASE, other], ["other", "both-refund"]) == [CASE, other]
    assert select_cases([CASE, other], None) == [CASE, other]
    with pytest.raises(EvaluationError, match="Unknown case IDs: nope"):
        select_cases([CASE], ["nope"])


# The recorder notes each tool call and returns the real result unchanged, so the
# tool loop behaves as it does for customers.
def test_tool_recorder_returns_the_real_result():
    result = {"type": "tool_result", "tool_use_id": "toolu_1", "content": "[]"}
    recorder = ToolRecorder(lambda tool_call, session, customer_id: result)
    block = type("Block", (), {"name": "get_invoices", "input": {}})()

    assert recorder(block, None, "cust_001") is result
    assert recorder.calls == [ToolCall("get_invoices", {}, result)]


# A case goes through the real API route as its customer: the response, the tool
# calls with what the search returned, and the time are all captured. The app's
# console output goes to stderr, so stdout holds only the report.
def test_run_cases_uses_the_real_api_route(engine, knowledge_base, capsys):
    with Session(engine) as session:
        session.add(
            Customer(
                id="cust_001",
                name="Alex Morgan",
                email="alex@example.com",
                company_name="Acme Studio",
                account_status=AccountStatus.active,
                created_at=date(2026, 1, 15),
            )
        )
        session.commit()
    claude = FakeClaude(
        tool_response(
            tool_call("search_knowledge_base", tool_input={"query": "refunds"})
        ),
        reply_response("No refund.", citations=["cancellation:5"]),
    )
    app.dependency_overrides[get_session_factory] = lambda: lambda: Session(engine)
    app.dependency_overrides[get_llm_service] = claude.service
    try:
        [result] = run_cases([CASE])
    finally:
        app.dependency_overrides.clear()

    assert result.status_code == 200
    assert result.body["citations"] == ["cancellation:5"]
    assert claude.requests[0]["messages"][0]["content"] == CASE.question
    assert [search.query for search in result.searches()] == ["refunds"]
    assert result.searches()[0].passages[0]["id"] == "cancellation:5"
    assert result.elapsed_seconds > 0
    assert capsys.readouterr().out == ""


# The report puts everything a reviewer needs next to the expected behavior: the
# question, tools used against those expected, the search results, the reply, each
# cited passage's text, the analysis, the time and a review list.
def test_report_shows_each_case_for_review():
    result = CaseResult(
        case=CASE,
        status_code=200,
        body={
            "reply": "Canceling doesn't refund\nthe current period.",
            "citations": ["cancellation:5"],
            "analysis": ANALYSIS,
            "conversation_id": "conv_1",
        },
        tool_calls=[search_call([PASSAGE])],
        elapsed_seconds=1.54,
    )

    report = format_report([result], datetime(2026, 11, 8, 9, 30, tzinfo=UTC), "m")

    assert "Run at 2026-11-08 09:30 UTC with m: 1 cases, 0 failed requests" in report
    assert "| both-refund | Knowledge base and account | cust_001 | 200 |" in report
    assert "**Question:** Can I get a refund?" in report
    assert "**Expected behavior:** Explains the refund policy." in report
    assert "search_knowledge_base · not used: get_payments" in report
    assert "1. `refund policy`" in report
    assert "> Canceling doesn't refund\n> the current period." in report
    assert (
        "- `cancellation:5` Canceling Your Subscription › Refunds after canceling\n\n"
        "> Canceling doesn't refund the current period."
    ) in report
    assert "human review: no. The customer asked what they paid." in report
    assert "- **Time:** 1.5 s" in report
    assert "- [ ] Each citation supports a claim in the reply" in report


# A failed request and a failed search are shown as failures with their messages,
# instead of breaking the report.
def test_report_shows_failures():
    result = CaseResult(
        case=CASE,
        status_code=502,
        body={"detail": "The model returned an invalid reply"},
        tool_calls=[search_call([], is_error=True)],
        elapsed_seconds=2.0,
    )

    report = format_report([result], datetime(2026, 11, 8, tzinfo=UTC), "m")

    assert "1 cases, 1 failed requests" in report
    assert "   - Failed: Error: down" in report
    assert "**Request failed:** HTTP 502: The model returned an invalid reply" in report
    assert "**Reply:**" not in report
