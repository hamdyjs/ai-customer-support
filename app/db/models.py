from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlmodel import JSON, Field, Relationship, SQLModel


def new_conversation_id() -> str:
    return f"conv_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class AccountStatus(StrEnum):
    active = "active"
    suspended = "suspended"
    closed = "closed"


class SubscriptionStatus(StrEnum):
    active = "active"
    canceled = "canceled"
    expired = "expired"


class BillingCycle(StrEnum):
    monthly = "monthly"
    annual = "annual"


class InvoiceStatus(StrEnum):
    open = "open"
    paid = "paid"
    overdue = "overdue"


class PaymentStatus(StrEnum):
    succeeded = "succeeded"
    failed = "failed"


class TicketStatus(StrEnum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"


class TicketPriority(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class Customer(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    email: str = Field(unique=True)
    company_name: str
    account_status: AccountStatus
    created_at: date

    subscriptions: list["Subscription"] = Relationship(back_populates="customer")
    invoices: list["Invoice"] = Relationship(back_populates="customer")
    payments: list["Payment"] = Relationship(back_populates="customer")
    tickets: list["SupportTicket"] = Relationship(back_populates="customer")


class Plan(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str = Field(unique=True)
    monthly_price_cents: int
    annual_price_cents: int
    description: str
    features: list[str] = Field(sa_type=JSON)


class Subscription(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.id", index=True)
    plan_id: str = Field(foreign_key="plan.id")
    status: SubscriptionStatus
    billing_cycle: BillingCycle
    started_at: date
    current_period_end: date
    canceled_at: date | None = None

    customer: Customer = Relationship(back_populates="subscriptions")
    plan: Plan = Relationship()
    invoices: list["Invoice"] = Relationship(back_populates="subscription")


class Invoice(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.id", index=True)
    subscription_id: str = Field(foreign_key="subscription.id", index=True)
    amount_cents: int
    currency: str
    status: InvoiceStatus
    issued_at: date
    due_at: date
    period_start: date
    period_end: date

    customer: Customer = Relationship(back_populates="invoices")
    subscription: Subscription = Relationship(back_populates="invoices")
    payments: list["Payment"] = Relationship(back_populates="invoice")


class Payment(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.id", index=True)
    invoice_id: str = Field(foreign_key="invoice.id", index=True)
    amount_cents: int
    status: PaymentStatus
    attempted_at: datetime
    failure_reason: str | None = None

    customer: Customer = Relationship(back_populates="payments")
    invoice: Invoice = Relationship(back_populates="payments")


class SupportTicket(SQLModel, table=True):
    __tablename__ = "support_ticket"  # type: ignore

    id: str = Field(primary_key=True)
    customer_id: str = Field(foreign_key="customer.id", index=True)
    subject: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    created_at: date
    resolved_at: date | None = None

    customer: Customer = Relationship(back_populates="tickets")


class Conversation(SQLModel, table=True):
    id: str = Field(default_factory=new_conversation_id, primary_key=True)
    customer_id: str = Field(foreign_key="customer.id", index=True)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationTurn(SQLModel, table=True):
    __tablename__ = "conversation_turn"  # type: ignore

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(foreign_key="conversation.id", index=True)
    question: str
    reply: str
    citations: list[str] = Field(default_factory=list, sa_type=JSON)
    analysis: dict[str, Any] = Field(sa_type=JSON)
    llm_messages: list[dict[str, Any]] = Field(sa_type=JSON)
    created_at: datetime = Field(default_factory=utc_now)
