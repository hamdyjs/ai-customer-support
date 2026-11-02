import asyncio
import json
from datetime import date

import httpx2
import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.ai import (
    DEFAULT_RETRY_AFTER_SECONDS,
    MAX_LLM_CALLS,
    ReplyError,
    SupportReply,
)
from app.db.models import AccountStatus, Customer, Invoice, InvoiceStatus
from app.tools import TOOL_DEFINITIONS
from tests.fake_claude import (
    ANALYSIS,
    FakeClaude,
    message_response,
    reply_response,
    text_block,
    tool_call,
    tool_response,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        for customer_id, invoice_id, email in [
            ("cust_001", "inv_1", "alex@example.com"),
            ("cust_008", "inv_8", "alex.morgan@example.com"),
        ]:
            session.add(
                Customer(
                    id=customer_id,
                    name="Alex Morgan",
                    email=email,
                    company_name="Acme Studio",
                    account_status=AccountStatus.active,
                    created_at=date(2026, 1, 15),
                )
            )
            session.add(
                Invoice(
                    id=invoice_id,
                    customer_id=customer_id,
                    subscription_id="sub_1",
                    amount_cents=4900,
                    currency="USD",
                    status=InvoiceStatus.paid,
                    issued_at=date(2026, 11, 1),
                    due_at=date(2026, 11, 8),
                    period_start=date(2026, 11, 1),
                    period_end=date(2026, 12, 1),
                )
            )
        session.commit()
        yield session


# Earlier messages are sent to Claude before the new one, so it has the context for
# a follow-up question.
def test_sends_history_before_the_new_message(session):
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": [text_block("Hello!")]},
    ]
    claude = FakeClaude(reply_response())

    claude.generate(session, history=history)

    assert claude.requests[0]["messages"] == [
        *history,
        {"role": "user", "content": "What did I pay?"},
    ]


# The messages to store contain the whole turn in order: the question, the tool
# call, its result and the final answer.
def test_returns_the_full_turn_for_storage(session):
    claude = FakeClaude(tool_response(tool_call("get_invoices")), reply_response())

    stored = claude.generate(session).llm_messages

    assert [message["role"] for message in stored] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert stored[0]["content"] == "What did I pay?"
    assert stored[1]["content"][0]["id"] == "toolu_1"
    assert stored[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert json.loads(stored[3]["content"][0]["text"])["reply"] == "You paid $49.00."
    json.dumps(stored)


# Thinking blocks are sent back during the turn but left out of the messages to
# store, because stored blocks would be rejected once the history changes.
def test_thinking_is_sent_during_the_turn_but_not_stored(session):
    thinking = {"type": "thinking", "thinking": "", "signature": "abc"}
    redacted = {"type": "redacted_thinking", "data": "xyz"}
    claude = FakeClaude(
        message_response([thinking, redacted, tool_call("get_invoices")], "tool_use"),
        message_response(
            [thinking, text_block(json.dumps({"reply": "Hi", "analysis": ANALYSIS}))]
        ),
    )

    stored = claude.generate(session).llm_messages

    assert claude.requests[1]["messages"][1]["content"][0]["type"] == "thinking"
    assert "thinking" not in json.dumps(stored)
    assert stored[1]["content"][0]["type"] == "tool_use"
    assert stored[3]["content"][0]["type"] == "text"


# Claude's final JSON answer becomes a validated SupportReply, with the reply and
# the analysis as real Python objects.
def test_returns_a_validated_reply_and_analysis(session):
    claude = FakeClaude(reply_response("Hello!"))

    result = claude.reply(session)

    assert isinstance(result, SupportReply)
    assert result.reply == "Hello!"
    assert result.analysis.category == "billing"
    assert result.analysis.human_review_recommended is False


# When Claude answers straight away, no tool runs and only one call is made. The
# first request must still offer Claude the tools.
def test_returns_the_reply_without_tools(session):
    claude = FakeClaude(reply_response())

    claude.reply(session)

    assert len(claude.requests) == 1
    assert claude.requests[0]["tools"] == TOOL_DEFINITIONS
    assert claude.requests[0]["messages"] == [
        {"role": "user", "content": "What did I pay?"}
    ]


# Every request sends a strict JSON schema with the reply and the nested analysis.
# This is what makes Claude return structured output.
def test_requests_structured_output(session):
    claude = FakeClaude(tool_response(tool_call("get_customer")), reply_response())

    claude.reply(session)

    for request in claude.requests:
        output_format = request["output_config"]["format"]
        schema = output_format["schema"]
        analysis = schema["$defs"]["QuestionAnalysis"]
        assert output_format["type"] == "json_schema"
        assert set(schema["properties"]) == {"reply", "analysis"}
        assert schema["additionalProperties"] is False
        assert set(analysis["properties"]) == set(ANALYSIS)
        assert analysis["additionalProperties"] is False


# The core loop: Claude asks for a tool, the tool's result goes back to Claude
# linked by its ID, and Claude's next answer becomes the reply.
def test_runs_the_tool_and_sends_the_result_back(session):
    claude = FakeClaude(
        tool_response(tool_call("get_invoices")),
        reply_response("You paid $49.00."),
    )

    assert claude.reply(session).reply == "You paid $49.00."

    assistant, user = claude.requests[1]["messages"][1:]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0]["name"] == "get_invoices"
    assert user["role"] == "user"
    tool_result = user["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_1"
    assert json.loads(tool_result["content"])[0]["id"] == "inv_1"


# Text Claude writes alongside a tool call isn't the final answer, so it isn't
# treated as JSON and doesn't fail the request.
def test_text_alongside_a_tool_call_is_not_parsed(session):
    claude = FakeClaude(
        message_response(
            [text_block("Let me check that."), tool_call("get_invoices")],
            stop_reason="tool_use",
        ),
        reply_response("You paid $49.00."),
    )

    assert claude.reply(session).reply == "You paid $49.00."
    assert len(claude.requests) == 2


# Each step of the loop is printed: the customer's message, Claude's tool use and
# text, and why each call stopped. Tool results stay out to keep the console short.
def test_prints_each_step_of_the_loop(session, capsys):
    claude = FakeClaude(
        tool_response(tool_call("get_invoices")),
        reply_response("You paid $49.00."),
    )

    claude.reply(session)

    output = capsys.readouterr().out
    assert "[Message from cust_001] What did I pay?" in output
    assert "[LLM call 1] stop_reason=tool_use" in output
    assert "Tool use: get_invoices input={}" in output
    assert "Tool result" not in output
    assert "inv_1" not in output
    assert "[LLM call 2] stop_reason=end_turn" in output
    assert "You paid $49.00." in output


# An unknown tool is printed as an error, so it's easy to spot in the output.
def test_prints_tool_errors(session, capsys):
    claude = FakeClaude(tool_response(tool_call("get_everything")), reply_response())

    claude.reply(session)

    output = capsys.readouterr().out
    assert "Tool error: get_everything -> Unknown tool: get_everything" in output


# Thinking blocks must be sent back unchanged with the rest of Claude's message,
# or the API can reject the next request.
def test_sends_thinking_blocks_back_to_claude(session):
    thinking = {"type": "thinking", "thinking": "", "signature": "abc"}
    claude = FakeClaude(
        message_response([thinking, tool_call("get_invoices")], "tool_use"),
        reply_response(),
    )

    claude.reply(session)

    assistant = claude.requests[1]["messages"][1]
    assert assistant["content"][0]["type"] == "thinking"
    assert assistant["content"][0]["signature"] == "abc"


# When Claude asks for several tools at once, all the results go back together in
# one message, each linked to the right call.
def test_sends_all_results_from_one_turn_in_one_message(session):
    claude = FakeClaude(
        tool_response(
            tool_call("get_invoices", "toolu_1"),
            tool_call("get_customer", "toolu_2"),
        ),
        reply_response(),
    )

    claude.reply(session)

    messages = claude.requests[1]["messages"]
    results = messages[-1]["content"]
    assert len(messages) == 3
    assert [result["tool_use_id"] for result in results] == ["toolu_1", "toolu_2"]
    assert json.loads(results[0]["content"])[0]["id"] == "inv_1"
    assert json.loads(results[1]["content"])["email"] == "alex@example.com"


# Claude can make several rounds of tool calls, and each request includes the
# whole conversation so far.
def test_supports_multiple_rounds_of_tools(session):
    claude = FakeClaude(
        tool_response(tool_call("get_customer", "toolu_1")),
        tool_response(tool_call("get_invoices", "toolu_2")),
        reply_response("Done."),
    )

    assert claude.reply(session).reply == "Done."
    assert [len(request["messages"]) for request in claude.requests] == [1, 3, 5]


# Tools use the customer given by our code, so results belong to that customer.
def test_tools_run_for_the_given_customer(session):
    claude = FakeClaude(tool_response(tool_call("get_invoices")), reply_response())

    claude.reply(session, customer_id="cust_008")

    result = claude.requests[1]["messages"][-1]["content"][0]
    assert [invoice["id"] for invoice in json.loads(result["content"])] == ["inv_8"]


# Even if Claude tries to pass another customer's ID, such as after a misleading
# message, its tool input is ignored and it only gets the current customer's data.
def test_ignores_a_customer_id_from_claude(session):
    claude = FakeClaude(
        tool_response(tool_call("get_invoices", tool_input={"customer_id": "cust_008"})),
        reply_response(),
    )

    claude.reply(session, customer_id="cust_001")

    result = claude.requests[1]["messages"][-1]["content"][0]
    assert [invoice["id"] for invoice in json.loads(result["content"])] == ["inv_1"]


# An unknown tool is reported back to Claude as an error so it can recover, and
# the request doesn't fail.
def test_unknown_tool_is_reported_back_to_claude(session):
    claude = FakeClaude(
        tool_response(tool_call("get_everything")),
        reply_response("Sorry, I can't look that up."),
    )

    assert claude.reply(session).reply == "Sorry, I can't look that up."

    result = claude.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "Unknown tool: get_everything" in result["content"]


# If Claude keeps asking for tools, the loop stops at the limit, which caps the
# cost of a single request.
def test_stops_after_too_many_calls(session):
    responses = [tool_response(tool_call("get_customer")) for _ in range(20)]
    claude = FakeClaude(*responses)

    with pytest.raises(ReplyError, match="too many tool calls"):
        claude.reply(session)

    assert len(claude.requests) == MAX_LLM_CALLS


# A refusal or a cut-off answer raises ReplyError, so a partial reply is never sent
# to the customer.
@pytest.mark.parametrize(
    ("content", "stop_reason"),
    [
        ([], "refusal"),
        ([text_block('{"reply": "You paid')], "max_tokens"),
    ],
    ids=["refusal", "cut_off"],
)
def test_unusable_answer_raises_reply_error(session, content, stop_reason):
    claude = FakeClaude(message_response(content, stop_reason))

    with pytest.raises(ReplyError, match=stop_reason):
        claude.reply(session)


# A finished answer that doesn't match the schema raises ReplyError instead of an
# unhandled validation error: empty or non-JSON text, missing parts, or bad values.
@pytest.mark.parametrize(
    "content",
    [
        [],
        [text_block("You paid $49.00.")],
        [text_block(json.dumps({"reply": "You paid $49.00."}))],
        [text_block(json.dumps({"analysis": ANALYSIS}))],
        [
            text_block(
                json.dumps(
                    {"reply": "Hi", "analysis": {**ANALYSIS, "category": "other"}}
                )
            )
        ],
    ],
    ids=["empty", "not_json", "missing_analysis", "missing_reply", "bad_category"],
)
def test_invalid_output_raises_reply_error(session, content):
    claude = FakeClaude(message_response(content))

    with pytest.raises(ReplyError, match="invalid reply"):
        claude.reply(session)


# API and network failures raise ReplyError, including partway through the loop.
@pytest.mark.parametrize(
    "failure",
    [
        httpx2.Response(
            429, json={"type": "error", "error": {"type": "x", "message": "m"}}
        ),
        httpx2.ConnectError("Network down"),
    ],
    ids=["rate_limit", "network_down"],
)
def test_api_failure_during_the_loop_raises_reply_error(session, failure):
    claude = FakeClaude(tool_response(tool_call("get_invoices")), failure)

    with pytest.raises(ReplyError, match="temporarily unavailable"):
        claude.reply(session)

    assert len(claude.requests) == 2


# A tool_use stop with no tool call raises ReplyError, instead of sending the API an
# empty message that it would reject.
def test_tool_use_stop_without_a_tool_call_raises_reply_error(session):
    claude = FakeClaude(tool_response(text_block("Let me check.")), reply_response())

    with pytest.raises(ReplyError, match="tool_use without a tool call"):
        claude.reply(session)

    assert len(claude.requests) == 1


# The whole loop shares one time limit, so several slow calls can't add up to a very
# long request.
def test_request_limit_covers_the_whole_loop(session, monkeypatch):
    monkeypatch.setattr("app.ai.REQUEST_TIMEOUT_SECONDS", 0.05)

    class SlowClaude(FakeClaude):
        async def __call__(self, request):
            await asyncio.sleep(0.03)
            return super().__call__(request)

    claude = SlowClaude(
        *[tool_response(tool_call("get_customer")) for _ in range(MAX_LLM_CALLS)]
    )

    with pytest.raises(ReplyError, match="temporarily unavailable") as error:
        claude.reply(session)

    assert error.value.retry_after == DEFAULT_RETRY_AFTER_SECONDS
    assert len(claude.requests) < MAX_LLM_CALLS


# Token usage is printed for each call and as a total for the request, but only to
# the console.
def test_prints_usage_for_each_call_and_the_request(session, capsys):
    claude = FakeClaude(tool_response(tool_call("get_customer")), reply_response())

    claude.reply(session)

    output = capsys.readouterr().out
    assert "[LLM call 1] stop_reason=tool_use input_tokens=1 output_tokens=1" in output
    assert "[LLM call 2] stop_reason=end_turn input_tokens=1 output_tokens=1" in output
    assert "[Request total] llm_calls=2 input_tokens=2 output_tokens=2" in output
    assert "estimated_cost=$0.000001" in output


# A failed request still prints what it spent before failing, because those calls
# were billed.
def test_failed_request_still_prints_its_usage(session, capsys):
    claude = FakeClaude(
        tool_response(tool_call("get_customer")),
        message_response([], stop_reason="refusal"),
    )

    with pytest.raises(ReplyError):
        claude.reply(session)

    assert "[Request total] llm_calls=2" in capsys.readouterr().out


# A failure partway through the loop is logged with its status code.
def test_failure_during_the_loop_is_logged(session, caplog):
    claude = FakeClaude(
        tool_response(tool_call("get_customer")),
        httpx2.Response(
            401, json={"type": "error", "error": {"type": "x", "message": "m"}}
        ),
    )

    with pytest.raises(ReplyError):
        claude.reply(session)

    assert "Claude API request failed: status=401" in caplog.text


# Reaching the call limit is logged, because nothing else would explain the failure.
def test_too_many_calls_is_logged(session, caplog):
    claude = FakeClaude(
        *[tool_response(tool_call("get_customer")) for _ in range(MAX_LLM_CALLS)]
    )

    with pytest.raises(ReplyError):
        claude.reply(session)

    assert f"Claude used all {MAX_LLM_CALLS} calls" in caplog.text
