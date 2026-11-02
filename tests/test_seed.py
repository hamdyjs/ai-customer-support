import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_seed(database):
    subprocess.run(
        [sys.executable, "-m", "app.db.seed"],
        cwd=ROOT,
        env={**os.environ, "DATABASE_URL": f"sqlite:///{database}"},
        check=True,
        capture_output=True,
    )


def query(database, sql):
    with sqlite3.connect(database) as connection:
        return connection.execute(sql).fetchall()


# Rows in a stable order, so two databases can be compared regardless of how SQLite
# happened to order its schema.
def all_rows(database):
    with sqlite3.connect(database) as connection:
        return sorted(
            line for line in connection.iterdump() if line.startswith("INSERT")
        )


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    path = tmp_path_factory.mktemp("seed") / "seed.db"
    run_seed(path)
    return path


# The seed creates the row counts the test scenarios were designed around, in a
# temporary database that leaves the real app.db alone.
@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ("plan", 3),
        ("customer", 8),
        ("subscription", 9),
        ("invoice", 36),
        ("payment", 38),
        ("support_ticket", 15),
    ],
)
def test_seed_creates_the_expected_rows(database, table, expected):
    assert query(database, f"SELECT COUNT(*) FROM {table}") == [(expected,)]


# Seeding includes the conversation tables, and leaves them empty.
@pytest.mark.parametrize("table", ["conversation", "conversation_turn"])
def test_seed_creates_empty_conversation_tables(database, table):
    assert query(database, f"SELECT COUNT(*) FROM {table}") == [(0,)]


# Seeding twice gives identical data, so scenarios and demos stay the same, and the
# second run replaces the old rows instead of failing or adding to them.
def test_seed_is_repeatable(database, tmp_path):
    second = tmp_path / "second.db"
    run_seed(second)
    first_rows = all_rows(second)

    run_seed(second)

    assert all_rows(second) == first_rows == all_rows(database)


# Customer 2 has two successful payments on one invoice: the duplicate charge.
def test_duplicate_charge_scenario(database):
    rows = query(
        database,
        "SELECT invoice_id, COUNT(*) FROM payment WHERE customer_id = 'cust_002' "
        "AND status = 'succeeded' GROUP BY invoice_id HAVING COUNT(*) > 1",
    )

    assert rows == [("inv_011", 2)]


# Customer 3 has a failed payment followed by a successful retry on one invoice.
def test_failed_payment_then_retry_scenario(database):
    rows = query(
        database,
        "SELECT status, failure_reason FROM payment WHERE invoice_id = 'inv_015' "
        "ORDER BY attempted_at",
    )

    assert rows == [("failed", "expired_card"), ("succeeded", None)]


# Customer 4 moved from Starter to Pro, with the old subscription kept as history.
def test_upgrade_scenario(database):
    rows = query(
        database,
        "SELECT plan_id, status FROM subscription WHERE customer_id = 'cust_004' "
        "ORDER BY started_at",
    )

    assert rows == [("plan_starter", "expired"), ("plan_pro", "active")]


# Customer 5 requested cancellation but keeps access until the period ends.
def test_cancellation_scenario(database):
    rows = query(
        database,
        "SELECT status, canceled_at, current_period_end FROM subscription "
        "WHERE customer_id = 'cust_005'",
    )

    assert rows == [("active", "2026-11-01", "2026-12-01")]


# Customer 6 has an overdue invoice with only failed attempts, and an open one.
def test_overdue_scenario(database):
    statuses = query(
        database,
        "SELECT id, status FROM invoice WHERE customer_id = 'cust_006' "
        "AND status != 'paid' ORDER BY id",
    )
    attempts = query(
        database, "SELECT DISTINCT status FROM payment WHERE invoice_id = 'inv_031'"
    )

    assert statuses == [("inv_031", "overdue"), ("inv_032", "open")]
    assert attempts == [("failed",)]


# Customer 7 has an open ticket, plus tickets in the other two statuses.
def test_open_ticket_scenario(database):
    rows = query(
        database,
        "SELECT status FROM support_ticket WHERE customer_id = 'cust_007' "
        "ORDER BY status",
    )

    assert rows == [("in_progress",), ("open",), ("resolved",)]


# Customers 1 and 8 share a name but are separate accounts, for isolation testing.
def test_same_name_scenario(database):
    rows = query(
        database,
        "SELECT id FROM customer WHERE name = (SELECT name FROM customer "
        "WHERE id = 'cust_001') ORDER BY id",
    )

    assert rows == [("cust_001",), ("cust_008",)]
