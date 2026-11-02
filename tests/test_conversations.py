import json
from datetime import date

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.ai import MAX_LLM_CALLS, get_llm_service
from app.conversations import MAX_HISTORY_TURNS
from app.db.database import get_session
from app.db.models import AccountStatus, Conversation, ConversationTurn, Customer
from app.main import app
from tests.fake_claude import (
    ANALYSIS,
    FakeClaude,
    message_response,
    reply_response,
    text_block,
    tool_call,
    tool_response,
)

MESSAGES_URL = "/v1/support/messages"
CONVERSATIONS_URL = "/v1/support/conversations"
ALEX = "cust_001"
OTHER_ALEX = "cust_008"


def auth(customer_id):
    return {"Authorization": f"Bearer {customer_id}"}


def use_claude(*responses):
    claude = FakeClaude(*responses)
    app.dependency_overrides[get_llm_service] = claude.service
    return claude


def send(client, question, customer_id=ALEX, conversation_id=None):
    body = {"question": question}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    return client.post(MESSAGES_URL, json=body, headers=auth(customer_id))


def start_conversation(client, customer_id=ALEX):
    use_claude(reply_response("First reply"))
    return send(client, "First question", customer_id).json()["conversation_id"]


def count_rows(engine, model):
    with Session(engine) as session:
        return len(session.exec(select(model)).all())


# The API requires every tool_use to be answered by a tool_result in the very next
# message, and the conversation to start with a plain user message.
def assert_valid_sequence(messages):
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        tool_ids = [
            block["id"] for block in message["content"] if block["type"] == "tool_use"
        ]
        if tool_ids:
            results = messages[index + 1]["content"]
            assert [result["tool_use_id"] for result in results] == tool_ids


@pytest.fixture
def client(engine):
    def get_test_session():
        with Session(engine) as session:
            yield session

    with Session(engine) as session:
        for customer_id, email in [
            (ALEX, "alex@example.com"),
            (OTHER_ALEX, "alex.morgan@example.com"),
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
        session.commit()
    app.dependency_overrides[get_session] = get_test_session
    yield TestClient(app)
    app.dependency_overrides.clear()


# A message without an ID creates a conversation, returns its ID, and keeps the
# existing reply and analysis fields.
def test_first_message_creates_a_conversation(client, engine):
    use_claude(reply_response("Hello!"))

    response = send(client, "Hi")

    body = response.json()
    assert response.status_code == 200
    assert body["conversation_id"].startswith("conv_")
    assert body["reply"] == "Hello!"
    assert body["analysis"] == ANALYSIS
    with Session(engine) as session:
        conversation = session.get(Conversation, body["conversation_id"])
        assert conversation is not None
        assert conversation.customer_id == ALEX
    assert count_rows(engine, ConversationTurn) == 1


# Each new conversation gets its own ID and starts with no history, so an earlier
# conversation doesn't leak into it.
def test_new_conversations_are_separate(client):
    first_id = start_conversation(client)
    claude = use_claude(reply_response())

    second_id = send(client, "A new topic").json()["conversation_id"]

    assert first_id != second_id
    assert claude.requests[0]["messages"] == [
        {"role": "user", "content": "A new topic"}
    ]


# A follow-up reaches Claude with the whole first turn, including its tool call and
# result in a valid order, so Claude can answer from what it already looked up.
def test_follow_up_includes_the_previous_turn(client, engine):
    use_claude(
        tool_response(tool_call("get_customer", "toolu_abc")),
        reply_response("You're Alex."),
    )
    conversation_id = send(client, "Who am I?").json()["conversation_id"]
    claude = use_claude(reply_response("At Acme Studio."))

    response = send(client, "Where do I work?", conversation_id=conversation_id)

    messages = claude.requests[0]["messages"]
    assert response.status_code == 200
    assert response.json()["conversation_id"] == conversation_id
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == "Who am I?"
    assert messages[1]["content"][0]["id"] == "toolu_abc"
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_abc"
    assert "alex@example.com" in messages[2]["content"][0]["content"]
    assert json.loads(messages[3]["content"][0]["text"])["reply"] == "You're Alex."
    assert messages[4]["content"] == "Where do I work?"
    assert_valid_sequence(messages)
    assert count_rows(engine, Conversation) == 1
    assert count_rows(engine, ConversationTurn) == 2


# Stored turns can themselves be replayed: a third message still has both earlier
# turns, in the order they happened.
def test_history_keeps_growing_in_order(client):
    conversation_id = start_conversation(client)
    use_claude(reply_response("Second reply"))
    send(client, "Second question", conversation_id=conversation_id)
    claude = use_claude(reply_response())

    send(client, "Third question", conversation_id=conversation_id)

    messages = claude.requests[0]["messages"]
    questions = [m["content"] for m in messages if m["role"] == "user"]
    assert questions == ["First question", "Second question", "Third question"]


# A follow-up only gets its own conversation's history, not the same customer's
# other conversations.
def test_follow_up_excludes_other_conversations(client):
    first_id = start_conversation(client)
    use_claude(reply_response("Other reply"))
    send(client, "A different topic")
    claude = use_claude(reply_response())

    send(client, "Second question", conversation_id=first_id)

    questions = [
        message["content"]
        for message in claude.requests[0]["messages"]
        if message["role"] == "user"
    ]
    assert questions == ["First question", "Second question"]


# The history endpoint only returns the turns from the requested conversation.
def test_history_excludes_other_conversations(client):
    first_id = start_conversation(client)
    use_claude(reply_response("Other reply"))
    send(client, "A different topic")

    response = client.get(f"{CONVERSATIONS_URL}/{first_id}", headers=auth(ALEX))

    assert [turn["question"] for turn in response.json()["turns"]] == [
        "First question"
    ]


# One customer can't continue another's conversation, even with the same name. They
# get a 404 before any Claude call, and nothing is added to the conversation.
def test_cannot_continue_another_customers_conversation(client, engine):
    conversation_id = start_conversation(client, ALEX)
    claude = use_claude(reply_response())

    response = send(client, "Hi", OTHER_ALEX, conversation_id)

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}
    assert claude.requests == []
    assert count_rows(engine, ConversationTurn) == 1


# One customer can't read another's history, and gets the same 404 as for an ID
# that doesn't exist, so the response doesn't reveal that it exists.
def test_cannot_read_another_customers_conversation(client):
    conversation_id = start_conversation(client, ALEX)

    response = client.get(
        f"{CONVERSATIONS_URL}/{conversation_id}", headers=auth(OTHER_ALEX)
    )
    missing = client.get(f"{CONVERSATIONS_URL}/conv_missing", headers=auth(OTHER_ALEX))

    assert response.status_code == 404
    assert response.json() == missing.json()


# The list only contains the signed-in customer's own conversations.
def test_list_only_shows_the_customers_conversations(client):
    alex_first = start_conversation(client, ALEX)
    alex_second = start_conversation(client, ALEX)
    other_alex = start_conversation(client, OTHER_ALEX)

    alex = client.get(CONVERSATIONS_URL, headers=auth(ALEX)).json()
    other = client.get(CONVERSATIONS_URL, headers=auth(OTHER_ALEX)).json()

    assert {item["id"] for item in alex} == {alex_first, alex_second}
    assert [item["id"] for item in other] == [other_alex]


# Unknown and malformed IDs get a 404 on sending a message, and never reach Claude
# or create anything.
@pytest.mark.parametrize(
    "conversation_id",
    ["conv_missing", "not-a-real-id", "", "conv_' OR '1'='1"],
    ids=["unknown", "malformed", "empty", "sql_like"],
)
def test_invalid_id_on_message_returns_404(client, engine, conversation_id):
    claude = use_claude(reply_response())

    response = send(client, "Hi", conversation_id=conversation_id)

    assert response.status_code == 404
    assert claude.requests == []
    assert count_rows(engine, Conversation) == 0


# Unknown and malformed IDs also get a 404 when reading history.
@pytest.mark.parametrize(
    "conversation_id",
    ["conv_missing", "not-a-real-id", "conv_' OR '1'='1"],
    ids=["unknown", "malformed", "sql_like"],
)
def test_invalid_id_on_history_returns_404(client, conversation_id):
    response = client.get(
        f"{CONVERSATIONS_URL}/{conversation_id}", headers=auth(ALEX)
    )

    assert response.status_code == 404


# History is read back from the database in order, without calling Claude, and
# shows what the customer asked and saw.
def test_history_persists_and_is_returned_in_order(client):
    conversation_id = start_conversation(client)
    use_claude(reply_response("Second reply"))
    send(client, "Second question", conversation_id=conversation_id)
    claude = use_claude()

    response = client.get(f"{CONVERSATIONS_URL}/{conversation_id}", headers=auth(ALEX))

    body = response.json()
    assert response.status_code == 200
    assert body["id"] == conversation_id
    assert [turn["question"] for turn in body["turns"]] == [
        "First question",
        "Second question",
    ]
    assert [turn["reply"] for turn in body["turns"]] == [
        "First reply",
        "Second reply",
    ]
    assert body["turns"][0]["analysis"] == ANALYSIS
    assert claude.requests == []


# The database row holds the question, the reply and the internal messages, while
# the history endpoint leaves out the internal messages and tool data.
def test_internal_messages_are_stored_but_not_returned(client, engine):
    use_claude(tool_response(tool_call("get_customer")), reply_response("Hi"))
    conversation_id = send(client, "Who am I?").json()["conversation_id"]

    response = client.get(f"{CONVERSATIONS_URL}/{conversation_id}", headers=auth(ALEX))

    with Session(engine) as session:
        turn = session.exec(select(ConversationTurn)).one()
    assert turn.question == "Who am I?"
    assert turn.reply == "Hi"
    assert len(turn.llm_messages) == 4
    assert "llm_messages" not in response.text
    assert "tool_use" not in response.text
    assert "alex@example.com" not in response.text


FAILURES = [
    pytest.param(lambda: [message_response([], "refusal")], id="refusal"),
    pytest.param(
        lambda: [message_response([text_block('{"reply": "Hi')], "max_tokens")],
        id="cut_off",
    ),
    pytest.param(lambda: [message_response([text_block("Not JSON")])], id="bad_output"),
    pytest.param(
        lambda: [
            httpx2.Response(
                429, json={"type": "error", "error": {"type": "x", "message": "m"}}
            )
        ],
        id="rate_limit",
    ),
    pytest.param(
        lambda: [httpx2.ConnectError("Network down")],
        id="network_down",
    ),
    pytest.param(
        lambda: [
            tool_response(tool_call("get_customer")),
            httpx2.ConnectError("Network down"),
        ],
        id="fails_after_a_tool_call",
    ),
    pytest.param(
        lambda: [tool_response(tool_call("get_customer"))] * MAX_LLM_CALLS,
        id="too_many_tool_calls",
    ),
]


# A failure on the first message creates nothing, not even an empty conversation.
@pytest.mark.parametrize("responses", FAILURES)
def test_failed_first_message_saves_nothing(client, engine, responses):
    use_claude(*responses())

    response = send(client, "Hi")

    assert response.status_code == 502
    assert count_rows(engine, Conversation) == 0
    assert count_rows(engine, ConversationTurn) == 0


# A failed follow-up leaves the conversation as it was, and a retry works with a
# valid history that has no trace of the failed attempt.
@pytest.mark.parametrize("responses", FAILURES)
def test_failed_follow_up_saves_nothing_and_can_be_retried(client, engine, responses):
    conversation_id = start_conversation(client)
    use_claude(*responses())

    response = send(client, "Failed question", conversation_id=conversation_id)

    assert response.status_code == 502
    assert count_rows(engine, ConversationTurn) == 1

    claude = use_claude(reply_response("Worked"))
    retry = send(client, "Retried question", conversation_id=conversation_id)

    messages = claude.requests[0]["messages"]
    questions = [m["content"] for m in messages if m["role"] == "user"]
    assert retry.status_code == 200
    assert questions == ["First question", "Retried question"]
    assert_valid_sequence(messages)


# Only the most recent turns are sent to Claude. Old turns are dropped whole, so the
# sequence stays valid, while the history endpoint still returns every turn.
def test_history_limit_drops_whole_turns(client, engine):
    conversation_id = start_conversation(client)
    total_turns = MAX_HISTORY_TURNS + 2
    for number in range(2, total_turns + 1):
        use_claude(
            tool_response(tool_call("get_customer", f"toolu_{number}")),
            reply_response(f"Reply {number}"),
        )
        send(client, f"Question {number}", conversation_id=conversation_id)
    claude = use_claude(reply_response("Final"))

    send(client, "Newest question", conversation_id=conversation_id)

    messages = claude.requests[0]["messages"]
    questions = [m["content"] for m in messages if isinstance(m["content"], str)]
    assert questions == [
        *[
            f"Question {number}"
            for number in range(total_turns - MAX_HISTORY_TURNS + 1, total_turns + 1)
        ],
        "Newest question",
    ]
    assert_valid_sequence(messages)

    history = client.get(f"{CONVERSATIONS_URL}/{conversation_id}", headers=auth(ALEX))
    assert len(history.json()["turns"]) == total_turns + 1


# Thinking blocks from an earlier turn are never replayed, so dropping old turns or
# changing the prompt can't invalidate them.
def test_follow_up_does_not_replay_thinking(client, engine):
    thinking = {"type": "thinking", "thinking": "", "signature": "abc"}
    use_claude(
        message_response([thinking, tool_call("get_customer")], "tool_use"),
        reply_response(),
    )
    conversation_id = send(client, "Who am I?").json()["conversation_id"]
    claude = use_claude(reply_response())

    send(client, "Thanks", conversation_id=conversation_id)

    assert "thinking" not in json.dumps(claude.requests[0]["messages"])
    with Session(engine) as session:
        turn = session.exec(select(ConversationTurn)).first()
        assert turn is not None
        assert "thinking" not in json.dumps(turn.llm_messages)


# A turn can't be saved without a real conversation, because the database enforces
# the link between them.
def test_turn_requires_an_existing_conversation(engine):
    with Session(engine) as session:
        session.add(
            ConversationTurn(
                conversation_id="conv_missing",
                question="Hi",
                reply="Hello",
                analysis=ANALYSIS,
                llm_messages=[],
            )
        )
        with pytest.raises(Exception, match="FOREIGN KEY"):
            session.commit()
