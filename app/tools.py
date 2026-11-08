from datetime import date, datetime

from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json
from sqlmodel import Session, col, select

from app import retrieval
from app.db.models import (
    AccountStatus,
    BillingCycle,
    Customer,
    Invoice,
    InvoiceStatus,
    Payment,
    PaymentStatus,
    Subscription,
    SubscriptionStatus,
    SupportTicket,
    TicketPriority,
    TicketStatus,
)
from app.tracing import trace_tool


class ToolResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class CustomerResult(ToolResult):
    name: str
    email: str
    company_name: str
    account_status: AccountStatus
    created_at: date


class SubscriptionResult(ToolResult):
    id: str
    plan_name: str
    plan_description: str
    plan_features: list[str]
    monthly_price_cents: int
    annual_price_cents: int
    status: SubscriptionStatus
    billing_cycle: BillingCycle
    started_at: date
    current_period_end: date
    canceled_at: date | None


class InvoiceResult(ToolResult):
    id: str
    subscription_id: str
    amount_cents: int
    currency: str
    status: InvoiceStatus
    issued_at: date
    due_at: date
    period_start: date
    period_end: date


class PaymentResult(ToolResult):
    id: str
    invoice_id: str
    amount_cents: int
    status: PaymentStatus
    attempted_at: datetime
    failure_reason: str | None


class SupportTicketResult(ToolResult):
    id: str
    subject: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    created_at: date
    resolved_at: date | None


class PassageResult(ToolResult):
    id: str  # the chunk's ID: document_id:chunk_index
    document_id: str
    title: str
    section_heading: str | None
    source: str
    text: str


def get_customer(session: Session, customer_id: str) -> CustomerResult:
    customer = session.exec(select(Customer).where(Customer.id == customer_id)).one()
    return CustomerResult.model_validate(customer)


def get_subscription(session: Session, customer_id: str) -> list[SubscriptionResult]:
    subscriptions = session.exec(
        select(Subscription)
        .where(Subscription.customer_id == customer_id)
        .order_by(col(Subscription.started_at).desc())
    ).all()
    return [
        SubscriptionResult(
            id=subscription.id,
            plan_name=subscription.plan.name,
            plan_description=subscription.plan.description,
            plan_features=subscription.plan.features,
            monthly_price_cents=subscription.plan.monthly_price_cents,
            annual_price_cents=subscription.plan.annual_price_cents,
            status=subscription.status,
            billing_cycle=subscription.billing_cycle,
            started_at=subscription.started_at,
            current_period_end=subscription.current_period_end,
            canceled_at=subscription.canceled_at,
        )
        for subscription in subscriptions
    ]


def get_invoices(session: Session, customer_id: str) -> list[InvoiceResult]:
    invoices = session.exec(
        select(Invoice)
        .where(Invoice.customer_id == customer_id)
        .order_by(col(Invoice.issued_at).desc())
    ).all()
    return [InvoiceResult.model_validate(invoice) for invoice in invoices]


def get_payments(session: Session, customer_id: str) -> list[PaymentResult]:
    payments = session.exec(
        select(Payment)
        .where(Payment.customer_id == customer_id)
        .order_by(col(Payment.attempted_at).desc())
    ).all()
    return [PaymentResult.model_validate(payment) for payment in payments]


def get_support_tickets(
    session: Session, customer_id: str
) -> list[SupportTicketResult]:
    tickets = session.exec(
        select(SupportTicket)
        .where(SupportTicket.customer_id == customer_id)
        .order_by(col(SupportTicket.created_at).desc())
    ).all()
    return [SupportTicketResult.model_validate(ticket) for ticket in tickets]


def search_knowledge_base(query: str) -> list[PassageResult]:
    results = retrieval.search_knowledge_base(retrieval.get_qdrant_client(), query)
    return [PassageResult.model_validate(result) for result in results]


# Tools for the signed-in customer's own data. They take no input, so Claude can't
# choose whose data they return.
CUSTOMER_TOOLS = {
    "get_customer": get_customer,
    "get_subscription": get_subscription,
    "get_invoices": get_invoices,
    "get_payments": get_payments,
    "get_support_tickets": get_support_tickets,
}

NO_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

TOOL_DEFINITIONS: list[ToolParam] = [
    {
        "name": "get_customer",
        "description": (
            "Get the current customer's profile: name, email, company, account "
            "status, and the date they joined."
        ),
        "input_schema": NO_INPUT_SCHEMA,
    },
    {
        "name": "get_subscription",
        "description": (
            "Get the current customer's subscriptions, newest first, including "
            "past ones. Each has its plan name, features, prices in cents, status, "
            "billing cycle, start date, current period end, and cancellation date. "
            "An active subscription with a cancellation date keeps its access until "
            "the current period ends. Use this for plan, renewal, cancellation and "
            "upgrade questions."
        ),
        "input_schema": NO_INPUT_SCHEMA,
    },
    {
        "name": "get_invoices",
        "description": (
            "Get the current customer's invoices, newest first. Each has its amount "
            "in cents, currency, status, issue and due dates, and billing period. "
            "Use this for what was billed and whether an invoice is paid or overdue."
        ),
        "input_schema": NO_INPUT_SCHEMA,
    },
    {
        "name": "get_payments",
        "description": (
            "Get the current customer's payment attempts, newest first, including "
            "failed ones. Each has its invoice, amount in cents, status, time, and "
            "failure reason. A failed attempt didn't charge the customer. Two "
            "successful payments on one invoice are a possible duplicate charge."
        ),
        "input_schema": NO_INPUT_SCHEMA,
    },
    {
        "name": "get_support_tickets",
        "description": (
            "Get the current customer's support tickets, newest first. Each has its "
            "subject, description, status, priority, creation date, and resolution "
            "date."
        ),
        "input_schema": NO_INPUT_SCHEMA,
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Search Flowdesk's help articles, which cover account settings, "
            "cancellation, failed payments, invoices, refunds and duplicate charges, "
            "subscription changes, support tickets, and troubleshooting. Returns the "
            "closest passages, best first, each with its chunk ID, article ID, title, "
            "section heading, source file, and text. Use this for how Flowdesk works "
            "and its policies. It has no customer data. Describe the topic in the "
            "product's own terms rather than repeating the customer's message, and "
            "search again with different words if the passages don't answer it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, such as 'refund after canceling'",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


def run_tool(
    name: str, tool_input: dict[str, object], session: Session, customer_id: str
) -> str:
    if name == "search_knowledge_base":
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search_knowledge_base needs a query")
        with trace_tool(name) as tool:
            passages = search_knowledge_base(query)
            chunk_ids = [passage.id for passage in passages]
            tool.update(
                metadata={"chunk_count": len(chunk_ids), "chunk_ids": chunk_ids}
            )
        return to_json(passages).decode()
    if name not in CUSTOMER_TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    # Claude's input is ignored, so the customer always comes from our code.
    with trace_tool(name):
        records = CUSTOMER_TOOLS[name](session, customer_id)
    return to_json(records).decode()
