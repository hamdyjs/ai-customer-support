import asyncio
import json
from datetime import date

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from sqlmodel import Session, SQLModel, create_engine

from app.ai import MAX_LLM_CALLS, MODEL, LLMService, ReplyError
from app.db.models import AccountStatus, Customer, Invoice, InvoiceStatus
from app.tools import TOOL_DEFINITIONS


def text_block(text):
    return {"type": "text", "text": text}


def tool_call(name, tool_id="toolu_1", tool_input=None):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input or {}}


def message_response(content, stop_reason="end_turn"):
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def tool_response(*calls):
    return message_response(list(calls), stop_reason="tool_use")


class FakeClaude:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def reply(self, session, customer_id="cust_001"):
        client = AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self)),
        )
        service = LLMService(client)
        return asyncio.run(
            service.generate_reply("What did I pay?", session, customer_id)
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


# When Claude answers straight away, no tool runs and only one call is made. The
# first request must still offer Claude the tools.
def test_returns_the_reply_without_tools(session):
    claude = FakeClaude(message_response([text_block("Hello!")]))

    assert claude.reply(session) == "Hello!"
    assert len(claude.requests) == 1
    assert claude.requests[0]["tools"] == TOOL_DEFINITIONS
    assert claude.requests[0]["messages"] == [
        {"role": "user", "content": "What did I pay?"}
    ]


# The core loop: Claude asks for a tool, the tool's result goes back to Claude
# linked by its ID, and Claude's next answer becomes the reply.
def test_runs_the_tool_and_sends_the_result_back(session):
    claude = FakeClaude(
        tool_response(tool_call("get_invoices")),
        message_response([text_block("You paid $49.00.")]),
    )

    assert claude.reply(session) == "You paid $49.00."

    assistant, user = claude.requests[1]["messages"][1:]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0]["name"] == "get_invoices"
    assert user["role"] == "user"
    tool_result = user["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_1"
    assert json.loads(tool_result["content"])[0]["id"] == "inv_1"


# Each step of the loop is printed: the customer's message, Claude's tool use and
# text, and why each call stopped. Tool results stay out to keep the console short.
def test_prints_each_step_of_the_loop(session, capsys):
    claude = FakeClaude(
        tool_response(tool_call("get_invoices")),
        message_response([text_block("You paid $49.00.")]),
    )

    claude.reply(session)

    output = capsys.readouterr().out
    assert "[Message from cust_001] What did I pay?" in output
    assert "[LLM call 1] stop_reason=tool_use" in output
    assert "Tool use: get_invoices input={}" in output
    assert "Tool result" not in output
    assert "inv_1" not in output
    assert "[LLM call 2] stop_reason=end_turn" in output
    assert "Text: You paid $49.00." in output


# An unknown tool is printed as an error, so it's easy to spot in the output.
def test_prints_tool_errors(session, capsys):
    claude = FakeClaude(
        tool_response(tool_call("get_everything")),
        message_response([text_block("Sorry.")]),
    )

    claude.reply(session)

    output = capsys.readouterr().out
    assert "Tool error: get_everything -> Unknown tool: get_everything" in output


# Thinking blocks must be sent back unchanged with the rest of Claude's message,
# or the API can reject the next request.
def test_sends_thinking_blocks_back_to_claude(session):
    thinking = {"type": "thinking", "thinking": "", "signature": "abc"}
    claude = FakeClaude(
        message_response([thinking, tool_call("get_invoices")], "tool_use"),
        message_response([text_block("You paid $49.00.")]),
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
        message_response([text_block("Done.")]),
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
        message_response([text_block("Done.")]),
    )

    assert claude.reply(session) == "Done."
    assert [len(request["messages"]) for request in claude.requests] == [1, 3, 5]


# Tools use the customer given by our code, so results belong to that customer.
def test_tools_run_for_the_given_customer(session):
    claude = FakeClaude(
        tool_response(tool_call("get_invoices")),
        message_response([text_block("Done.")]),
    )

    claude.reply(session, customer_id="cust_008")

    result = claude.requests[1]["messages"][-1]["content"][0]
    assert [invoice["id"] for invoice in json.loads(result["content"])] == ["inv_8"]


# Even if Claude tries to pass another customer's ID, such as after a misleading
# message, its tool input is ignored and it only gets the current customer's data.
def test_ignores_a_customer_id_from_claude(session):
    claude = FakeClaude(
        tool_response(tool_call("get_invoices", tool_input={"customer_id": "cust_008"})),
        message_response([text_block("Done.")]),
    )

    claude.reply(session, customer_id="cust_001")

    result = claude.requests[1]["messages"][-1]["content"][0]
    assert [invoice["id"] for invoice in json.loads(result["content"])] == ["inv_1"]


# An unknown tool is reported back to Claude as an error so it can recover, and
# the request doesn't fail.
def test_unknown_tool_is_reported_back_to_claude(session):
    claude = FakeClaude(
        tool_response(tool_call("get_everything")),
        message_response([text_block("Sorry, I can't look that up.")]),
    )

    assert claude.reply(session) == "Sorry, I can't look that up."

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


# A refusal, a cut-off answer or an empty answer raises ReplyError, so a partial
# reply is never sent to the customer.
@pytest.mark.parametrize(
    ("content", "stop_reason", "error"),
    [
        ([], "refusal", "refusal"),
        ([text_block("You paid")], "max_tokens", "max_tokens"),
        ([], "end_turn", "end_turn"),
    ],
    ids=["refusal", "cut_off", "empty"],
)
def test_unusable_answer_raises_reply_error(session, content, stop_reason, error):
    claude = FakeClaude(message_response(content, stop_reason))

    with pytest.raises(ReplyError, match=error):
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

    with pytest.raises(ReplyError, match="API request failed"):
        claude.reply(session)

    assert len(claude.requests) == 2
