from datetime import date, datetime

from pydantic import BaseModel, ConfigDict
from pydantic_core import to_json
from sqlmodel import Session, col, select

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


TOOLS = {
    "get_customer": get_customer,
    "get_subscription": get_subscription,
    "get_invoices": get_invoices,
    "get_payments": get_payments,
    "get_support_tickets": get_support_tickets,
}

NO_INPUT_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}

TOOL_DEFINITIONS = [
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
]


def run_tool(name: str, session: Session, customer_id: str) -> str:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return to_json(TOOLS[name](session, customer_id)).decode()
