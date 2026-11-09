import json
import re
import sqlite3

import pytest

from scripts.evaluate_assistant import EvaluationError, load_cases
from scripts.validate_golden_dataset import GOLDEN_FILE, load_golden_cases, main
from tests.test_seed import run_seed

GOLDEN = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
RECORD_ID = re.compile(r"\b(?:inv|pay|ticket|sub)_\d{3}\b")
RECORD_TABLES = ["invoice", "payment", "support_ticket", "subscription"]


def write_golden(tmp_path, cases):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(cases))
    return path


# The real dataset passes validation and has one golden case per assistant case, in
# the same order.
def test_golden_dataset_covers_every_assistant_case():
    assert [case.id for case in load_golden_cases()] == [
        case.id for case in load_cases()
    ]


# Every invoice, payment, ticket and subscription ID the dataset mentions belongs to
# the case's customer in the seeded database, so no expectation points at a record
# that doesn't exist or belongs to someone else.
def test_record_ids_belong_to_the_case_customer(tmp_path):
    database = tmp_path / "seed.db"
    run_seed(database)
    with sqlite3.connect(database) as connection:
        owners = {
            record_id: customer_id
            for table in RECORD_TABLES
            for record_id, customer_id in connection.execute(
                f"SELECT id, customer_id FROM {table}"
            )
        }

    for case in load_golden_cases():
        for record_id in RECORD_ID.findall(case.model_dump_json()):
            assert owners.get(record_id) == case.customer_id, (case.id, record_id)


# Golden cases with empty or blank expectations, unknown or unexpected sources, edits
# to the reused assistant case, or unknown fields are rejected.
@pytest.mark.parametrize(
    ("case_id", "change", "message"),
    [
        ("kb-invoice-pdf", {"expected_facts": []}, "expected_facts"),
        ("kb-invoice-pdf", {"prohibited_claims": []}, "prohibited_claims"),
        ("kb-invoice-pdf", {"prohibited_claims": ["  "]}, "prohibited_claims"),
        (
            "kb-invoice-pdf",
            {"optional_facts": [{"text": "A fact.", "sources": []}]},
            "sources",
        ),
        (
            "kb-invoice-pdf",
            {"optional_facts": [{"text": "A fact.", "sources": ["pricing"]}]},
            r"unknown sources \['pricing'\]",
        ),
        (
            "kb-invoice-pdf",
            {"expected_facts": [{"text": "A fact.", "sources": ["get_invoices"]}]},
            "relies on \\['get_invoices'\\], which the case doesn't expect",
        ),
        ("kb-invoice-pdf", {"question": "Other?"}, "question differs"),
        ("kb-invoice-pdf", {"expected_tools": []}, "expected_tools differs"),
        ("gap-card-details", {"cannot_establish": []}, "can't be established"),
        ("kb-invoice-pdf", {"id": "kb-payment-retries"}, "duplicate case ID"),
        ("kb-invoice-pdf", {"notes": "x"}, "notes"),
    ],
    ids=[
        "no_expected_facts",
        "no_prohibited_claims",
        "blank_claim",
        "no_sources",
        "unknown_source",
        "unexpected_required_source",
        "changed_question",
        "changed_tools",
        "no_cannot_establish",
        "duplicate_id",
        "unknown_field",
    ],
)
def test_load_golden_cases_rejects_invalid_cases(tmp_path, case_id, change, message):
    cases = [{**case, **change} if case["id"] == case_id else case for case in GOLDEN]

    with pytest.raises(EvaluationError, match=message):
        load_golden_cases(write_golden(tmp_path, cases))


# The golden dataset covers exactly the assistant cases, with none missing or extra.
def test_load_golden_cases_rejects_missing_and_extra_cases(tmp_path):
    with pytest.raises(EvaluationError, match="kb-invoice-pdf from .* is missing"):
        load_golden_cases(write_golden(tmp_path, GOLDEN[1:]))

    extra = {**GOLDEN[0], "id": "new-case"}
    with pytest.raises(EvaluationError, match="new-case isn't in"):
        load_golden_cases(write_golden(tmp_path, [*GOLDEN, extra]))


# The command line prints a summary of a valid dataset.
def test_main_reports_a_valid_dataset(capsys):
    assert main() == 0
    assert "golden_dataset.json is valid: 12 cases" in capsys.readouterr().out
