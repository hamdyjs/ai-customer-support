"""Check the seeded scenarios against the real Claude API.

This makes real API calls using ANTHROPIC_API_KEY and costs a small amount, so it
isn't part of the test suite. It uses a temporary database and leaves app.db alone.

Run it from the repo root with:

    uv run python -m scripts.check_real_api

Claude's wording varies, so the checks only look for key facts. A failure means the
reply is worth reading, and not necessarily that something is broken.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DATES = ["december 1", "dec 1", "2026-12-01", "12/1", "1 december"]


@dataclass
class Scenario:
    name: str
    customer_id: str
    question: str
    # The reply must contain at least one phrase from each group.
    expected: list[list[str]]
    forbidden: list[str] = field(default_factory=list)
    human_review: bool | None = None


SCENARIOS = [
    Scenario(
        "Healthy account",
        "cust_001",
        "What plan am I on and when does it renew?",
        [["pro"], DATES],
    ),
    Scenario(
        "Duplicate charge",
        "cust_002",
        "Was I charged twice this month?",
        [["twice", "duplicate", "two successful", "two payments", "inv_011"]],
        human_review=True,
    ),
    Scenario(
        "Failed payment, then retry",
        "cust_003",
        "Was I charged twice in October?",
        [["expired"], ["$19"]],
    ),
    Scenario(
        "Recent upgrade",
        "cust_004",
        "Why was I charged more recently?",
        [["starter"], ["pro"], ["$49"]],
    ),
    Scenario(
        "Cancelled, still has access",
        "cust_005",
        "I cancelled. When do I lose access?",
        [DATES],
    ),
    Scenario(
        "Overdue invoice",
        "cust_006",
        "Do I have any unpaid invoices?",
        [["overdue", "inv_031"]],
    ),
    Scenario(
        "Open tickets",
        "cust_007",
        "What's the status of my support tickets?",
        [["automation"], ["integration"]],
    ),
    Scenario(
        "Same name as another customer",
        "cust_008",
        "What plan am I on? And what plan is Alex Morgan at Acme Studio on?",
        [["starter"]],
        forbidden=["pro plan"],
    ),
]


def tools_used(turn) -> list[str]:
    return [
        block["name"]
        for message in turn.llm_messages
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_use"
    ]


def find_problems(scenario: Scenario, body: dict) -> list[str]:
    reply = body["reply"].lower()
    problems = [
        f"expected one of {group}"
        for group in scenario.expected
        if not any(phrase in reply for phrase in group)
    ]
    problems += [
        f"should not mention {phrase!r}"
        for phrase in scenario.forbidden
        if phrase in reply
    ]
    review = body["analysis"]["human_review_recommended"]
    if scenario.human_review is not None and review != scenario.human_review:
        problems.append(f"expected human review to be {scenario.human_review}")
    return problems


def main() -> int:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY isn't set. Add it to your .env file.")
        return 1

    with tempfile.TemporaryDirectory() as folder:
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(folder) / 'check.db'}"
        subprocess.run([sys.executable, "-m", "app.db.seed"], check=True)
        return run_checks()


def run_checks() -> int:
    from fastapi.testclient import TestClient
    from sqlmodel import Session, select

    from app.db.database import engine
    from app.db.models import ConversationTurn, Invoice, InvoiceStatus
    from app.main import app

    url = "/v1/support/messages"
    failures = 0

    def post(client, customer_id, question, conversation_id=None):
        return client.post(
            url,
            json={"question": question, "conversation_id": conversation_id},
            headers={"Authorization": f"Bearer {customer_id}"},
        )

    with TestClient(app, raise_server_exceptions=False) as client:
        for scenario in SCENARIOS:
            response = post(client, scenario.customer_id, scenario.question)
            problems = (
                [f"HTTP {response.status_code}: {response.text}"]
                if response.status_code != 200
                else find_problems(scenario, response.json())
            )
            failures += bool(problems)
            print(f"\n=== {'FAIL' if problems else 'PASS'}: {scenario.name}")
            if response.status_code == 200:
                print("Reply:", response.json()["reply"])
                print("Analysis:", json.dumps(response.json()["analysis"]))
            for problem in problems:
                print("Problem:", problem)

        # Stale data: the invoice changes between two messages in one conversation,
        # so Claude has to look it up again to answer correctly.
        first = post(client, "cust_006", "Is invoice inv_031 paid?")
        conversation_id = first.json()["conversation_id"]
        with Session(engine) as session:
            invoice = session.get(Invoice, "inv_031")
            assert invoice is not None
            invoice.status = InvoiceStatus.paid
            session.add(invoice)
            session.commit()
        second = post(
            client, "cust_006", "Is invoice inv_031 paid now?", conversation_id
        )
        with Session(engine) as session:
            turns = session.exec(
                select(ConversationTurn).where(
                    ConversationTurn.conversation_id == conversation_id
                )
            ).all()
            stale_problems = [] if tools_used(turns[-1]) else ["didn't look it up again"]
        if second.status_code != 200:
            stale_problems.append(f"HTTP {second.status_code}: {second.text}")
        failures += bool(stale_problems)
        print(f"\n=== {'FAIL' if stale_problems else 'PASS'}: Stale data")
        print("First reply:", first.json()["reply"])
        print("Second reply:", second.json().get("reply"))
        for problem in stale_problems:
            print("Problem:", problem)

    total = len(SCENARIOS) + 1
    print(f"\n{total - failures} of {total} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
