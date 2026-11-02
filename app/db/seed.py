import random
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from itertools import count

from faker import Faker
from sqlmodel import Session, SQLModel

from app.db.database import engine
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

SEED = 42
CURRENT_PERIOD_START = date(2026, 11, 1)
CURRENT_PERIOD_END = date(2026, 12, 1)

fake = Faker()
fake.seed_instance(SEED)
rng = random.Random(SEED)
counters = defaultdict(lambda: count(1))

STARTER = Plan(
    id="plan_starter",
    name="Starter",
    monthly_price_cents=1900,
    annual_price_cents=19000,
    description="Essential project management",
    features=["Task boards", "Basic reports", "Email support"],
)
PRO = Plan(
    id="plan_pro",
    name="Pro",
    monthly_price_cents=4900,
    annual_price_cents=49000,
    description="Advanced project management",
    features=["Advanced reports", "Integrations", "Automation"],
)
BUSINESS = Plan(
    id="plan_business",
    name="Business",
    monthly_price_cents=9900,
    annual_price_cents=99000,
    description="Project management for growing teams",
    features=["Everything in Pro", "Team permissions", "Priority support"],
)


def next_id(prefix: str) -> str:
    return f"{prefix}_{next(counters[prefix]):03}"


def add_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    return date(day.year + month_index // 12, month_index % 12 + 1, day.day)


def month_starts(start: date, end: date) -> list[date]:
    months = []
    while start < end:
        months.append(start)
        start = add_months(start, 1)
    return months


def at(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)


def make_customer(
    joined_before: date,
    name: str | None = None,
    email: str | None = None,
    company_name: str | None = None,
    created_at: date | None = None,
) -> Customer:
    if name is None:
        first_name, last_name = fake.first_name(), fake.last_name()
        name = f"{first_name} {last_name}"
        email = f"{first_name}.{last_name}@example.com".lower()
    return Customer(
        id=next_id("cust"),
        name=name,
        email=email,
        company_name=company_name or fake.company(),
        account_status=AccountStatus.active,
        created_at=created_at or joined_before - timedelta(days=rng.randint(1, 30)),
    )


def make_subscription(
    customer: Customer,
    plan: Plan,
    started_at: date,
    current_period_end: date = CURRENT_PERIOD_END,
    status: SubscriptionStatus = SubscriptionStatus.active,
    billing_cycle: BillingCycle = BillingCycle.monthly,
    canceled_at: date | None = None,
) -> Subscription:
    return Subscription(
        id=next_id("sub"),
        customer=customer,
        plan=plan,
        status=status,
        billing_cycle=billing_cycle,
        started_at=started_at,
        current_period_end=current_period_end,
        canceled_at=canceled_at,
    )


def make_invoice(
    subscription: Subscription,
    period_start: date,
    status: InvoiceStatus = InvoiceStatus.paid,
) -> Invoice:
    if subscription.billing_cycle == BillingCycle.annual:
        amount_cents = subscription.plan.annual_price_cents
        period_end = add_months(period_start, 12)
    else:
        amount_cents = subscription.plan.monthly_price_cents
        period_end = add_months(period_start, 1)
    return Invoice(
        id=next_id("inv"),
        customer=subscription.customer,
        subscription=subscription,
        amount_cents=amount_cents,
        currency="USD",
        status=status,
        issued_at=period_start,
        due_at=period_start + timedelta(days=7),
        period_start=period_start,
        period_end=period_end,
    )


def make_payment(
    invoice: Invoice,
    attempted_at: datetime,
    status: PaymentStatus = PaymentStatus.succeeded,
    failure_reason: str | None = None,
) -> Payment:
    return Payment(
        id=next_id("pay"),
        customer=invoice.customer,
        invoice=invoice,
        amount_cents=invoice.amount_cents,
        status=status,
        attempted_at=attempted_at,
        failure_reason=failure_reason,
    )


def make_paid_invoice(subscription: Subscription, period_start: date) -> Invoice:
    invoice = make_invoice(subscription, period_start)
    make_payment(invoice, at(period_start, rng.randint(6, 20), rng.randint(0, 59)))
    return invoice


def make_paid_invoices(subscription: Subscription, start: date, end: date) -> None:
    for period_start in month_starts(start, end):
        make_paid_invoice(subscription, period_start)


def make_ticket(
    customer: Customer,
    subject: str,
    description: str,
    priority: TicketPriority,
    created_at: date,
    status: TicketStatus = TicketStatus.resolved,
    resolved_at: date | None = None,
) -> SupportTicket:
    return SupportTicket(
        id=next_id("ticket"),
        customer=customer,
        subject=subject,
        description=description,
        status=status,
        priority=priority,
        created_at=created_at,
        resolved_at=resolved_at,
    )


def seed_healthy_subscription() -> Customer:
    started_at = date(2026, 6, 1)
    customer = make_customer(
        started_at,
        name="Alex Morgan",
        email="alex@example.com",
        company_name="Acme Studio",
        created_at=date(2026, 1, 15),
    )
    subscription = make_subscription(customer, PRO, started_at)
    make_paid_invoices(subscription, started_at, CURRENT_PERIOD_END)
    make_ticket(
        customer,
        "Where can I download my invoices?",
        "I need copies of my invoices for our accountant.",
        TicketPriority.low,
        date(2026, 8, 12),
        resolved_at=date(2026, 8, 13),
    )
    return customer


def seed_duplicate_charge() -> Customer:
    started_at = date(2026, 7, 1)
    customer = make_customer(started_at)
    subscription = make_subscription(customer, PRO, started_at)
    make_paid_invoices(subscription, started_at, CURRENT_PERIOD_START)
    invoice = make_invoice(subscription, CURRENT_PERIOD_START)
    make_payment(invoice, at(CURRENT_PERIOD_START, 9, 14))
    make_payment(invoice, at(CURRENT_PERIOD_START, 9, 16))
    make_ticket(
        customer,
        "Unexpected subscription charge",
        "I was charged twice",
        TicketPriority.high,
        CURRENT_PERIOD_START,
        status=TicketStatus.open,
    )
    make_ticket(
        customer,
        "Add our VAT number to invoices",
        "Please show our VAT number on future invoices.",
        TicketPriority.low,
        date(2026, 9, 9),
        resolved_at=date(2026, 9, 10),
    )
    return customer


def seed_failed_payment_then_retry() -> Customer:
    started_at = date(2026, 7, 1)
    retried_period = date(2026, 10, 1)
    customer = make_customer(started_at)
    subscription = make_subscription(customer, STARTER, started_at)
    make_paid_invoices(subscription, started_at, retried_period)
    invoice = make_invoice(subscription, retried_period)
    make_payment(
        invoice,
        at(retried_period, 8, 5),
        status=PaymentStatus.failed,
        failure_reason="expired_card",
    )
    make_payment(invoice, at(date(2026, 10, 3), 10, 30))
    make_paid_invoice(subscription, CURRENT_PERIOD_START)
    make_ticket(
        customer,
        "Payment failed notification",
        "I got an email saying my payment failed, so I updated my card. Did it go through?",
        TicketPriority.medium,
        date(2026, 10, 2),
        resolved_at=date(2026, 10, 3),
    )
    make_ticket(
        customer,
        "Change our billing email",
        "Please send invoices to our finance team instead of me.",
        TicketPriority.low,
        date(2026, 10, 28),
        status=TicketStatus.in_progress,
    )
    return customer


def seed_recent_upgrade() -> Customer:
    started_at = date(2026, 6, 1)
    upgraded_at = date(2026, 10, 1)
    customer = make_customer(started_at)
    old_subscription = make_subscription(
        customer,
        STARTER,
        started_at,
        current_period_end=upgraded_at,
        status=SubscriptionStatus.expired,
    )
    make_paid_invoices(old_subscription, started_at, upgraded_at)
    subscription = make_subscription(customer, PRO, upgraded_at)
    make_paid_invoices(subscription, upgraded_at, CURRENT_PERIOD_END)
    make_ticket(
        customer,
        "Will my reports carry over?",
        "We're planning to upgrade to Pro. Will we keep our existing reports?",
        TicketPriority.low,
        date(2026, 9, 28),
        resolved_at=date(2026, 9, 29),
    )
    make_ticket(
        customer,
        "Why was I charged more this month?",
        "My last payment was higher than usual and I'd like to know why.",
        TicketPriority.medium,
        date(2026, 10, 2),
        resolved_at=date(2026, 10, 3),
    )
    return customer


def seed_canceled_subscription() -> Customer:
    started_at = date(2026, 7, 1)
    customer = make_customer(started_at)
    # Stays active with canceled_at set: cancellation was requested, and access
    # continues until the end of the paid period.
    subscription = make_subscription(
        customer, PRO, started_at, canceled_at=CURRENT_PERIOD_START
    )
    make_paid_invoices(subscription, started_at, CURRENT_PERIOD_END)
    make_ticket(
        customer,
        "How do I cancel my subscription?",
        "We're moving to another tool and would like to cancel.",
        TicketPriority.low,
        date(2026, 10, 30),
        resolved_at=date(2026, 10, 31),
    )
    make_ticket(
        customer,
        "When does my access end?",
        "I cancelled my subscription. How long can my team keep using Flowdesk?",
        TicketPriority.medium,
        CURRENT_PERIOD_START,
        status=TicketStatus.open,
    )
    return customer


def seed_overdue_invoice() -> Customer:
    started_at = date(2026, 7, 1)
    overdue_period = date(2026, 10, 1)
    customer = make_customer(started_at)
    subscription = make_subscription(customer, STARTER, started_at)
    make_paid_invoices(subscription, started_at, overdue_period)
    invoice = make_invoice(subscription, overdue_period, status=InvoiceStatus.overdue)
    for attempted_at in (at(overdue_period, 7, 40), at(date(2026, 10, 4), 7, 40)):
        make_payment(
            invoice,
            attempted_at,
            status=PaymentStatus.failed,
            failure_reason="insufficient_funds",
        )
    make_invoice(subscription, CURRENT_PERIOD_START, status=InvoiceStatus.open)
    make_ticket(
        customer,
        "My card keeps getting declined",
        "I've tried paying my October invoice but the payment doesn't go through.",
        TicketPriority.medium,
        date(2026, 10, 6),
        status=TicketStatus.in_progress,
    )
    make_ticket(
        customer,
        "Overdue invoice notice",
        "I received an overdue notice. Will my account be affected?",
        TicketPriority.high,
        date(2026, 10, 20),
        status=TicketStatus.open,
    )
    return customer


def seed_open_ticket() -> Customer:
    started_at = date(2026, 3, 1)
    customer = make_customer(started_at)
    subscription = make_subscription(
        customer,
        BUSINESS,
        started_at,
        current_period_end=add_months(started_at, 12),
        billing_cycle=BillingCycle.annual,
    )
    make_paid_invoice(subscription, started_at)
    make_ticket(
        customer,
        "Adding more team members",
        "How do we give new team members the right permissions?",
        TicketPriority.low,
        date(2026, 4, 10),
        resolved_at=date(2026, 4, 11),
    )
    make_ticket(
        customer,
        "Help connecting an integration",
        "Our calendar integration keeps asking us to reconnect.",
        TicketPriority.medium,
        date(2026, 10, 25),
        status=TicketStatus.in_progress,
    )
    make_ticket(
        customer,
        "Automation rules not running",
        "Our automation rules stopped running this week.",
        TicketPriority.high,
        date(2026, 10, 30),
        status=TicketStatus.open,
    )
    return customer


def seed_same_name_customer() -> Customer:
    started_at = date(2026, 9, 1)
    customer = make_customer(
        started_at,
        name="Alex Morgan",
        email="alex.morgan@example.com",
    )
    subscription = make_subscription(customer, STARTER, started_at)
    make_paid_invoices(subscription, started_at, CURRENT_PERIOD_END)
    make_ticket(
        customer,
        "Change my account email",
        "I'd like to use a different email address to sign in.",
        TicketPriority.low,
        date(2026, 10, 5),
        resolved_at=date(2026, 10, 6),
    )
    return customer


def main() -> None:
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                seed_healthy_subscription(),
                seed_duplicate_charge(),
                seed_failed_payment_then_retry(),
                seed_recent_upgrade(),
                seed_canceled_subscription(),
                seed_overdue_invoice(),
                seed_open_ticket(),
                seed_same_name_customer(),
            ]
        )
        session.commit()


if __name__ == "__main__":
    main()
