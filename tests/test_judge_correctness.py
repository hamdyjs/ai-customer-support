import json

import httpx2
import pytest

from scripts.judge_correctness import judge_correctness
from scripts.judge_faithfulness import JUDGE_MODEL, Verdict
from scripts.validate_golden_dataset import load_golden_cases
from tests.fake_claude import FakeClaude, message_response, text_block

GOLDEN = {case.id: case for case in load_golden_cases()}
# Three unpaid-invoice facts, six prohibited claims, nothing it can't establish.
OWED = GOLDEN["account-amount-owed"]
# Accepts a clarifying question instead of the invoice facts.
WHICH_INVOICE = GOLDEN["ambiguous-which-invoice"]
# Can't establish which card paid or when it expires.
CARD = GOLDEN["gap-card-details"]


def judgment(
    case,
    missing=(),
    prohibited=(),
    unestablished=(),
    incorrect=(),
    cleared=(),
    clarifying=False,
    numbers=None,
):
    """A judgment checking every item in the case, failing the given numbers.

    Claims in `incorrect` really contradict a fact. Claims in `cleared` are listed as
    possible contradictions that the judge then decided aren't.
    """

    def checks(items, key, value):
        return [
            {"number": number, key: value(number), "explanation": f"Item {number}."}
            for number in range(1, len(items) + 1)
        ]

    facts = checks(case.expected_facts, "communicated", lambda n: n not in missing)
    body = {
        "required_facts": facts,
        "prohibited_claims": checks(
            case.prohibited_claims, "violated", lambda n: n in prohibited
        ),
        "cannot_establish": checks(
            case.cannot_establish, "violated", lambda n: n in unestablished
        ),
        "incorrect_claims": [
            {
                "claim": claim,
                "explanation": "Compared with a fact.",
                "contradicts": True,
            }
            for claim in incorrect
        ]
        + [
            {"claim": claim, "explanation": "Same meaning.", "contradicts": False}
            for claim in cleared
        ],
        "asks_clarifying_question": clarifying,
        "summary": "Summary.",
    }
    if numbers is not None:
        body["required_facts"] = [dict(facts[0], number=n) for n in numbers]
    return message_response([text_block(json.dumps(body))], model=JUDGE_MODEL)


def judge(case, *responses, reply="You owe $38.00."):
    claude = FakeClaude(*responses)
    return claude, judge_correctness(claude.client(), case, reply)


# The judge gets the question, the reply and the golden expectations, numbered so
# each can be checked, with optional facts marked as optional. It never gets the
# tool results or the case's expected behavior, so correctness stays separate from
# faithfulness.
def test_judge_sees_the_golden_expectations_and_nothing_else():
    claude, _ = judge(CARD, judgment(CARD), reply="I can't see your card.")

    [request] = claude.requests
    prompt = request["messages"][0]["content"]
    assert request["model"] == JUDGE_MODEL
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert f"<question>\n{CARD.question}\n</question>" in prompt
    assert "<reply>\nI can't see your card.\n</reply>" in prompt
    assert f"1. {CARD.expected_facts[0].text}" in prompt
    assert f"- {CARD.optional_facts[0].text}" in prompt
    assert f"4. {CARD.prohibited_claims[3]}" in prompt
    assert f"2. {CARD.cannot_establish[1]}" in prompt
    assert "tool_result" not in prompt
    assert CARD.expected_behavior not in prompt


# A complete answer gets every required fact across and breaks nothing, so it passes
# even though the judge wasn't asked about the optional facts it left out.
def test_a_complete_answer_passes():
    _, result = judge(OWED, judgment(OWED))

    assert result.verdict == Verdict.passed
    assert result.summary == "Summary."
    assert not (result.missing_facts or result.prohibited_claims)
    assert not (result.unestablished or result.incorrect_claims)


# An incomplete answer fails, and names the required fact it missed, in the golden
# dataset's words, with the judge's reason.
def test_an_incomplete_answer_fails_with_the_missing_fact():
    _, result = judge(OWED, judgment(OWED, missing=[3]))

    assert result.verdict == Verdict.failed
    [missing] = result.missing_facts
    assert missing.expectation == OWED.expected_facts[2].text
    assert missing.explanation == "Item 3."


# An incorrect answer fails whether it makes a prohibited claim, answers something
# that can't be established, or contradicts a golden fact, and the result names it.
@pytest.mark.parametrize(
    ("case", "problem", "findings", "expected"),
    [
        (OWED, {"prohibited": [3]}, "prohibited_claims", OWED.prohibited_claims[2]),
        (CARD, {"unestablished": [2]}, "unestablished", CARD.cannot_establish[1]),
        (
            OWED,
            {"incorrect": ["Your account is suspended."]},
            "incorrect_claims",
            "Your account is suspended.",
        ),
    ],
    ids=["prohibited_claim", "cannot_be_established", "incorrect_claim"],
)
def test_an_incorrect_answer_fails_with_what_it_got_wrong(
    case, problem, findings, expected
):
    _, result = judge(case, judgment(case, **problem))

    assert result.verdict == Verdict.failed
    [finding] = getattr(result, findings)
    assert finding.expectation == expected


# A claim the judge lists as a possible contradiction but then decides is only a
# difference in wording doesn't fail the reply, so a cleared non-finding can't turn
# into a failure.
def test_a_claim_the_judge_clears_doesnt_fail_the_answer():
    response = judgment(OWED, cleared=["Only the October invoice is overdue."])

    _, result = judge(OWED, response)

    assert result.verdict == Verdict.passed
    assert result.incorrect_claims == []


# Where the case accepts it, a clarifying question passes without the invoice facts,
# and the result says the question stood in for them.
def test_an_acceptable_clarifying_question_passes_without_the_facts():
    response = judgment(WHICH_INVOICE, missing=[1, 2, 3], clarifying=True)

    _, result = judge(WHICH_INVOICE, response, reply="Which invoice do you mean?")

    assert result.verdict == Verdict.passed
    assert result.clarified
    assert result.missing_facts == []


# A reply that gives every fact and also asks which invoice was meant passes on its
# facts, so it isn't marked as having leaned on the clarifying question.
def test_a_complete_answer_that_also_asks_isnt_marked_clarified():
    response = judgment(WHICH_INVOICE, clarifying=True)

    _, result = judge(WHICH_INVOICE, response)

    assert result.verdict == Verdict.passed
    assert not result.clarified


# A clarifying question doesn't excuse missing facts in a case that doesn't accept
# one, and doesn't excuse a prohibited claim even where it's accepted.
@pytest.mark.parametrize(
    ("case", "problems"),
    [
        (OWED, {"missing": [1]}),
        (WHICH_INVOICE, {"missing": [1, 2, 3], "prohibited": [2]}),
    ],
    ids=["not_accepted_for_the_case", "prohibited_claim_still_fails"],
)
def test_a_clarifying_question_doesnt_excuse_everything(case, problems):
    _, result = judge(case, judgment(case, clarifying=True, **problems))

    assert result.verdict == Verdict.failed


# A judgment that skips or repeats an expectation can't be trusted, and an API
# failure gives no judgment, so both become errors instead of verdicts.
@pytest.mark.parametrize(
    "response",
    [
        judgment(OWED, numbers=[1, 2]),
        judgment(OWED, numbers=[1, 1, 3]),
        httpx2.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "x"}}
        ),
    ],
    ids=["skipped_fact", "repeated_fact", "api_failure"],
)
def test_an_unusable_judgment_is_an_error(response):
    _, result = judge(OWED, response)

    assert result.verdict == Verdict.error
    assert result.summary.startswith(
        ("The judge returned a malformed judgment", "The judge request failed")
    )
