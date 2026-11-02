import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_python(code, cwd, database=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("DATABASE_URL", None)
    if database is not None:
        env["DATABASE_URL"] = f"sqlite:///{database}"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def table_names(database):
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in rows}


# Starts the real app, which runs its startup steps, then sends one request.
START_APP_AND_CALL = """
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    response = client.post(
        "/v1/support/messages",
        json={"question": "Hi"},
        headers={"Authorization": "Bearer cust_001"},
    )
print(response.status_code)
"""


# The default database is always the project's app.db, whichever folder the app runs
# from, and importing the app there doesn't leave a stray file behind.
def test_default_database_path_ignores_the_current_folder(tmp_path):
    code = "from app.db.database import engine; print(engine.url.database)"

    assert run_python(code, tmp_path) == str(ROOT / "app.db")
    assert list(tmp_path.iterdir()) == []


# A brand-new database gets its tables at startup, so an unknown customer gets a
# normal 401 rather than a "no such table" 500.
def test_startup_creates_tables_in_a_new_database(tmp_path):
    database = tmp_path / "new.db"

    status = run_python(START_APP_AND_CALL, tmp_path, database)

    assert status == "401"
    assert {"customer", "conversation", "conversation_turn"} <= table_names(database)


# An older database gets its missing tables at startup without losing existing data,
# which was the cause of the "no such table: conversation" error.
def test_startup_adds_missing_tables_and_keeps_existing_data(tmp_path):
    database = tmp_path / "old.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE note (id INTEGER PRIMARY KEY, body TEXT)")
        connection.execute("INSERT INTO note (body) VALUES ('keep me')")

    run_python(START_APP_AND_CALL, tmp_path, database)

    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT body FROM note").fetchall()
    assert rows == [("keep me",)]
    assert "conversation" in table_names(database)
