import json
import subprocess
from collections import Counter
from datetime import UTC, date, datetime
from unittest.mock import Mock

import pytest
from sqlmodel import Session

from app import tracing
from app.ai import RequestUsage, get_llm_service
from app.db.database import get_session_factory
from app.db.models import AccountStatus, Customer
from app.main import app
from scripts import evaluate_assistant
from scripts.evaluate_assistant import (
    AssistantCase,
    CaseResult,
    EvaluationError,
    ToolCall,
    ToolRecorder,
    format_report,
    judge_results,
    load_cases,
    main,
    run_cases,
    select_cases,
)
from scripts.ingest import KNOWLEDGE_BASE_DIR
from scripts.judge_faithfulness import (
    JUDGE_MODEL,
    FaithfulnessResult,
    JudgedClaim,
    Verdict,
)
from scripts.validate_golden_dataset import load_golden_cases
from tests.fake_claude import (
    ANALYSIS,
    FakeClaude,
    message_response,
    reply_response,
    text_block,
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


def add_customer(engine):
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


def answered(case=CASE, reply="Canceling doesn't refund the current period."):
    body = {"reply": reply, "citations": [], "analysis": ANALYSIS}
    return CaseResult(case, 200, body, [search_call([PASSAGE])], 1.0)


def judged(verdict, summary, *claims):
    return FaithfulnessResult(
        verdict,
        summary,
        [JudgedClaim(claim=c, verdict=v, explanation=e) for c, v, e in claims],
    )


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
    add_customer(engine)
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


# Evaluation requests never reach Langfuse, even when its keys are set, so they don't
# mix with real customers' traces.
def test_run_cases_is_not_traced(engine, knowledge_base, monkeypatch):
    add_customer(engine)
    langfuse = Mock()
    monkeypatch.setattr(tracing, "get_langfuse", Mock(return_value=langfuse))
    claude = FakeClaude(reply_response("You paid $49.00."))
    app.dependency_overrides[get_session_factory] = lambda: lambda: Session(engine)
    app.dependency_overrides[get_llm_service] = claude.service
    try:
        [result] = run_cases([CASE])
    finally:
        app.dependency_overrides.clear()

    assert result.status_code == 200
    langfuse.start_observation.assert_not_called()


# The judge gets the question, the reply and the tool results, never the expected
# behavior or golden facts and claims, even for a golden case. Failed requests have
# no reply to judge, so they're skipped.
def test_judge_results_judges_replies_only_from_tool_results():
    golden = load_golden_cases()[0]
    reply = "Open Settings and look under Billing."
    failed = CaseResult(CASE, 502, {"detail": "Down"}, [], 1.0)
    claude = FakeClaude(
        message_response(
            [text_block(json.dumps({"claims": [], "summary": "No claims."}))]
        )
    )

    judge_results([answered(golden, reply), failed], claude.client())

    [request] = claude.requests
    prompt = request["messages"][0]["content"]
    assert reply in prompt and json.dumps([PASSAGE]) in prompt
    golden_text = [golden.expected_behavior, *golden.prohibited_claims]
    golden_text += [fact.text for fact in golden.expected_facts + golden.optional_facts]
    assert not [text for text in golden_text if text in prompt]
    assert failed.faithfulness is None


# The judge only runs, and only costs money, when --judge is passed. Without it,
# nothing is judged and the report has no faithfulness section.
@pytest.mark.parametrize(("argv", "judge"), [([], False), (["--judge"], True)])
def test_the_judge_runs_only_when_asked(monkeypatch, capsys, argv, judge):
    calls = []

    def fake_judge(client, question, reply, tool_calls):
        calls.append(question)
        return judged(Verdict.passed, "Every claim is backed.")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setattr(evaluate_assistant, "load_dotenv", lambda: None)
    monkeypatch.setattr(evaluate_assistant, "knowledge_base_problem", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluate_assistant, "run_cases", lambda cases: [answered(c) for c in cases]
    )
    monkeypatch.setattr(evaluate_assistant, "judge_faithfulness", fake_judge)

    assert main([*argv, "--case", "kb-invoice-pdf"]) == 0

    assert len(calls) == (1 if judge else 0)
    assert ("**Faithfulness:**" in capsys.readouterr().out) is judge


# A judged report counts the verdicts, adds a column for them, and lists the claims
# that failed or that the reply couldn't confirm, with the judge's reasons. Judge
# errors and cases that weren't judged are shown as such.
def test_report_shows_faithfulness():
    passing = answered()
    passing.faithfulness = judged(
        Verdict.passed,
        "Every claim is backed.",
        ("No refund for the current period.", "supported", "cancellation:5 says so."),
        ("It couldn't confirm the card.", "uncertain", "No card details."),
    )
    failing = answered()
    failing.faithfulness = judged(
        Verdict.failed,
        "One claim isn't backed.",
        ("You'll get $10 back.", "contradicted", "cancellation:5 says no refund."),
    )
    errored = answered()
    errored.faithfulness = judged(Verdict.error, "The judge request failed: down")
    unjudged = CaseResult(CASE, 502, {"detail": "Down"}, [], 1.0)

    report = format_report(
        [passing, failing, errored, unjudged], datetime(2026, 11, 9, tzinfo=UTC), "m"
    )

    assert (
        f"Faithfulness judged by {JUDGE_MODEL}: 1 passed, 1 failed, 1 judge errors, "
        "1 not judged."
    ) in report
    assert "| Time | Faithfulness |" in report
    assert "| 1.0 s | pass (1 supported, 1 uncertain) |" in report
    assert "| 1.0 s | fail (1 contradicted) |" in report
    assert "| 1.0 s | error |" in report and "| 1.0 s | not judged |" in report
    assert (
        "**Faithfulness:** pass (1 supported, 1 uncertain). Every claim is backed."
    ) in report
    assert "- **Uncertain:** It couldn't confirm the card. (No card details.)" in report
    assert (
        "- **Contradicted:** You'll get $10 back. (cancellation:5 says no refund.)"
    ) in report
    assert "No refund for the current period." not in report
    assert "**Faithfulness:** error. The judge request failed: down" in report


# Without --judge, the report is unchanged: no faithfulness column or section.
def test_report_without_the_judge_has_no_faithfulness():
    report = format_report([answered()], datetime(2026, 11, 9, tzinfo=UTC), "m")

    assert "Nothing here is scored." in report
    assert "Faithfulness" not in report


# Each case gets its own total of the assistant's Claude calls and tokens for the cost
# report, while the app still totals each request as usual.
def test_run_cases_totals_each_cases_claude_calls(engine, knowledge_base, capsys):
    add_customer(engine)
    search = tool_call("search_knowledge_base", tool_input={"query": "refunds"})
    claude = FakeClaude(
        tool_response(search, usage={"input_tokens": 100, "output_tokens": 10}),
        reply_response(
            usage={
                "input_tokens": 200,
                "output_tokens": 20,
                "cache_read_input_tokens": 50,
            }
        ),
        reply_response(usage={"input_tokens": 300, "output_tokens": 30}),
    )
    app.dependency_overrides[get_session_factory] = lambda: lambda: Session(engine)
    app.dependency_overrides[get_llm_service] = claude.service
    try:
        first, second = run_cases([CASE, CASE.model_copy(update={"id": "other"})])
    finally:
        app.dependency_overrides.clear()

    assert first.usage == RequestUsage(2, 300, 30, 50, 0)
    assert second.usage == RequestUsage(1, 300, 30, 0, 0)
    output = capsys.readouterr().err
    assert "[Request total] llm_calls=2 input_tokens=300" in output
    assert "[Request total] llm_calls=1 input_tokens=300" in output


# API errors and costs get their own section, apart from the verdicts: a failed
# request and a judge error are listed as errors, a faithfulness failure isn't, and
# the assistant's and the judge's costs are totalled and shown per case.
def test_report_lists_api_errors_and_costs_apart_from_verdicts():
    usage = RequestUsage(llm_calls=2, input_tokens=1_000_000, output_tokens=100_000)
    unfaithful = answered()
    unfaithful.usage = usage
    unfaithful.faithfulness = FaithfulnessResult(
        Verdict.failed,
        "One claim isn't backed.",
        [JudgedClaim(claim="A refund.", verdict="unsupported", explanation="None.")],
        input_tokens=10_000,
        output_tokens=1_000,
    )
    judge_error = answered()
    judge_error.faithfulness = FaithfulnessResult(
        Verdict.error, "The judge request failed: down"
    )
    request_error = CaseResult(CASE, 502, {"detail": "Down"}, [], 1.0, usage)

    report = format_report(
        [unfaithful, judge_error, request_error], datetime(2026, 11, 9, tzinfo=UTC), "m"
    )
    errors = report.split("**API errors:**")[1].split("**Assistant")[0]

    assert "## API errors and costs" in report
    assert errors == (
        "\n\n- both-refund: The judge request failed: down\n"
        "- both-refund: the assistant request failed with HTTP 502: Down\n\n"
    )
    assert (
        "**Assistant (m):** 4 Claude calls, 2,000,000 input and 200,000 output "
        "tokens, 0 cache reads and 0 cache writes: about $0.3000."
    ) in report
    assert (
        f"**Judge ({JUDGE_MODEL}):** 2 judgments, 10,000 input and 1,000 output "
        "tokens: about $0.0300."
    ) in report
    assert "| both-refund | 2 | $0.1500 | $0.0300 |" in report
    assert "| both-refund | - | - | $0.0000 |" in report
