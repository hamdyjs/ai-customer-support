import json
from datetime import UTC, date, datetime

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.db.models import (
    AccountStatus,
    BillingCycle,
    Customer,
    Invoice,
    InvoiceStatus,
    Payment,
    PaymentStatus,
    Plan,
    Subscription,
    SubscriptionStatus,
    SupportTicket,
    TicketPriority,
    TicketStatus,
)
from app.tools import (
    CUSTOMER_TOOLS,
    TOOL_DEFINITIONS,
    get_customer,
    get_invoices,
    get_payments,
    get_subscription,
    get_support_tickets,
    run_tool,
)


def add_billing_history(session: Session, customer_id: str, number: int) -> None:
    subscription_id = f"sub_{number}"
    invoice_id = f"inv_{number}"
    session.add(
        Subscription(
            id=subscription_id,
            customer_id=customer_id,
            plan_id="plan_pro",
            status=SubscriptionStatus.active,
            billing_cycle=BillingCycle.monthly,
            started_at=date(2026, 10, 1),
            current_period_end=date(2026, 12, 1),
        )
    )
    session.add(
        Invoice(
            id=invoice_id,
            customer_id=customer_id,
            subscription_id=subscription_id,
            amount_cents=4900,
            currency="USD",
            status=InvoiceStatus.paid,
            issued_at=date(2026, 11, 1),
            due_at=date(2026, 11, 8),
            period_start=date(2026, 11, 1),
            period_end=date(2026, 12, 1),
        )
    )
    session.add(
        Payment(
            id=f"pay_{number}",
            customer_id=customer_id,
            invoice_id=invoice_id,
            amount_cents=4900,
            status=PaymentStatus.succeeded,
            attempted_at=datetime(2026, 11, 1, 9, 30, tzinfo=UTC),
        )
    )
    session.add(
        SupportTicket(
            id=f"ticket_{number}",
            customer_id=customer_id,
            subject="Unexpected subscription charge",
            description="I was charged twice",
            status=TicketStatus.open,
            priority=TicketPriority.high,
            created_at=date(2026, 11, 1),
        )
    )


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        for customer_id, email, company_name in [
            ("cust_001", "alex@example.com", "Acme Studio"),
            ("cust_008", "alex.morgan@example.com", "Morgan Design"),
        ]:
            session.add(
                Customer(
                    id=customer_id,
                    name="Alex Morgan",
                    email=email,
                    company_name=company_name,
                    account_status=AccountStatus.active,
                    created_at=date(2026, 1, 15),
                )
            )
        for plan_id, name, monthly_price_cents in [
            ("plan_starter", "Starter", 1900),
            ("plan_pro", "Pro", 4900),
        ]:
            session.add(
                Plan(
                    id=plan_id,
                    name=name,
                    monthly_price_cents=monthly_price_cents,
                    annual_price_cents=monthly_price_cents * 10,
                    description=f"{name} plan",
                    features=["Reports", "Integrations"],
                )
            )
        add_billing_history(session, "cust_001", 1)
        add_billing_history(session, "cust_008", 8)
        session.commit()
        yield session


# get_customer returns the profile of the customer it's given.
def test_get_customer(session):
    customer = get_customer(session, "cust_008")

    assert customer.email == "alex.morgan@example.com"
    assert customer.company_name == "Morgan Design"
    assert customer.account_status == AccountStatus.active


# A subscription includes its plan details, so Claude doesn't need a separate
# plan tool to explain pricing.
def test_get_subscription_includes_the_plan(session):
    subscription = get_subscription(session, "cust_001")[0]

    assert subscription.plan_name == "Pro"
    assert subscription.monthly_price_cents == 4900
    assert subscription.annual_price_cents == 49000
    assert subscription.plan_features == ["Reports", "Integrations"]
    assert subscription.canceled_at is None


# Past subscriptions are returned newest first, so Claude can explain why a
# charge changed after an upgrade.
def test_get_subscription_returns_history_newest_first(session):
    session.add(
        Subscription(
            id="sub_old",
            customer_id="cust_001",
            plan_id="plan_starter",
            status=SubscriptionStatus.expired,
            billing_cycle=BillingCycle.monthly,
            started_at=date(2026, 6, 1),
            current_period_end=date(2026, 10, 1),
        )
    )
    session.commit()

    subscriptions = get_subscription(session, "cust_001")

    assert [s.id for s in subscriptions] == ["sub_1", "sub_old"]
    assert [s.plan_name for s in subscriptions] == ["Pro", "Starter"]


# Invoices are returned newest first, with the billing period Claude needs.
def test_get_invoices_returns_newest_first(session):
    session.add(
        Invoice(
            id="inv_old",
            customer_id="cust_001",
            subscription_id="sub_1",
            amount_cents=4900,
            currency="USD",
            status=InvoiceStatus.overdue,
            issued_at=date(2026, 10, 1),
            due_at=date(2026, 10, 8),
            period_start=date(2026, 10, 1),
            period_end=date(2026, 11, 1),
        )
    )
    session.commit()

    invoices = get_invoices(session, "cust_001")

    assert [i.id for i in invoices] == ["inv_1", "inv_old"]
    assert invoices[1].status == InvoiceStatus.overdue
    assert invoices[1].period_end == date(2026, 11, 1)


# Failed attempts are included, newest first, so Claude can tell a retry from a
# duplicate charge. The times matter because retries can happen on the same day.
def test_get_payments_includes_failed_attempts_newest_first(session):
    session.add(
        Payment(
            id="pay_failed",
            customer_id="cust_001",
            invoice_id="inv_1",
            amount_cents=4900,
            status=PaymentStatus.failed,
            attempted_at=datetime(2026, 11, 1, 9, 0, tzinfo=UTC),
            failure_reason="card_declined",
        )
    )
    session.commit()

    payments = get_payments(session, "cust_001")

    assert [p.id for p in payments] == ["pay_1", "pay_failed"]
    assert payments[1].status == PaymentStatus.failed
    assert payments[1].failure_reason == "card_declined"


# Tickets are returned newest first, with their status and resolution date.
def test_get_support_tickets_returns_newest_first(session):
    session.add(
        SupportTicket(
            id="ticket_old",
            customer_id="cust_001",
            subject="Change my email",
            description="Please update my email",
            status=TicketStatus.resolved,
            priority=TicketPriority.low,
            created_at=date(2026, 8, 1),
            resolved_at=date(2026, 8, 2),
        )
    )
    session.commit()

    tickets = get_support_tickets(session, "cust_001")

    assert [t.id for t in tickets] == ["ticket_1", "ticket_old"]
    assert tickets[1].resolved_at == date(2026, 8, 2)


# Tools only return the given customer's data, even when another customer with
# the same name has records. This is the customer isolation guarantee.
@pytest.mark.parametrize(
    ("tool", "expected_id"),
    [
        (get_subscription, "sub_8"),
        (get_invoices, "inv_8"),
        (get_payments, "pay_8"),
        (get_support_tickets, "ticket_8"),
    ],
    ids=["subscription", "invoices", "payments", "support_tickets"],
)
def test_tools_only_return_the_customers_own_records(session, tool, expected_id):
    assert [record.id for record in tool(session, "cust_008")] == [expected_id]


# A customer with no records gets an empty list rather than an error.
@pytest.mark.parametrize(
    "tool",
    [get_subscription, get_invoices, get_payments, get_support_tickets],
    ids=["subscription", "invoices", "payments", "support_tickets"],
)
def test_tools_return_empty_lists_without_records(session, tool):
    assert tool(session, "cust_999") == []


# Claude can't choose a customer: the customer tools accept no input, and the search
# tool accepts only a query. The customer ID always comes from our code.
def test_tool_definitions_take_no_customer_input():
    schemas = {
        definition["name"]: definition["input_schema"]
        for definition in TOOL_DEFINITIONS
    }

    for name in CUSTOMER_TOOLS:
        assert schemas[name] == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    search = schemas["search_knowledge_base"]
    assert set(search["properties"]) == {"query"}
    assert search["required"] == ["query"]
    assert search["additionalProperties"] is False


# Every tool Claude is told about has a function behind it, and vice versa.
def test_tool_definitions_match_the_tool_functions():
    assert {definition["name"] for definition in TOOL_DEFINITIONS} == {
        *CUSTOMER_TOOLS,
        "search_knowledge_base",
    }


# run_tool returns JSON text, which is what's sent back to Claude. Dates and
# statuses become plain strings.
def test_run_tool_returns_json(session):
    invoices = json.loads(run_tool("get_invoices", {}, session, "cust_001"))
    payments = json.loads(run_tool("get_payments", {}, session, "cust_001"))
    customer = json.loads(run_tool("get_customer", {}, session, "cust_001"))

    assert invoices[0]["status"] == "paid"
    assert invoices[0]["issued_at"] == "2026-11-01"
    assert payments[0]["attempted_at"].startswith("2026-11-01T09:30:00")
    assert customer["account_status"] == "active"


# run_tool passes the customer ID through to the tool it runs.
def test_run_tool_uses_the_given_customer(session):
    invoices = json.loads(run_tool("get_invoices", {}, session, "cust_008"))

    assert [invoice["id"] for invoice in invoices] == ["inv_8"]


# A tool name we don't recognize raises an error instead of running anything.
def test_run_tool_rejects_unknown_tools(session):
    with pytest.raises(ValueError, match="Unknown tool: get_everything"):
        run_tool("get_everything", {}, session, "cust_001")


# The search tool sends Claude's query to retrieval and returns each passage's ID,
# article, title, heading, source and text, best match first. The score stays out.
def test_search_knowledge_base_returns_passages(session, knowledge_base):
    passages = json.loads(
        run_tool("search_knowledge_base", {"query": "refunds"}, session, "cust_001")
    )

    assert knowledge_base.queries == ["refunds"]
    assert passages == [
        {
            "id": "cancellation:5",
            "document_id": "cancellation",
            "title": "Canceling Your Subscription",
            "section_heading": "Refunds after canceling",
            "source": "cancellation.md",
            "text": (
                "Canceling Your Subscription\nRefunds after canceling\n\n"
                "Canceling doesn't refund the current period."
            ),
        },
        {
            "id": "failed-payments:0",
            "document_id": "failed-payments",
            "title": "Failed Payments",
            "section_heading": None,
            "source": "failed-payments.md",
            "text": "Failed Payments\n\nWe retry a failed payment three times.",
        },
    ]


# A missing, blank or non-text query raises an error before anything is searched, so
# Claude is told to correct its call.
@pytest.mark.parametrize(
    "tool_input",
    [{}, {"query": "  "}, {"query": 5}],
    ids=["missing", "blank", "not_text"],
)
def test_search_knowledge_base_needs_a_query(session, knowledge_base, tool_input):
    with pytest.raises(ValueError, match="search_knowledge_base needs a query"):
        run_tool("search_knowledge_base", tool_input, session, "cust_001")

    assert knowledge_base.queries == []
