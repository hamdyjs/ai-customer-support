import json

from scripts.check_judge import check_example, load_examples
from scripts.evaluate_assistant import TOOL_NAMES
from scripts.validate_golden_dataset import load_golden_cases
from tests.fake_claude import FakeClaude, message_response, text_block


def judgment(verdict=None):
    claims = (
        []
        if verdict is None
        else [{"claim": "A claim.", "verdict": verdict, "explanation": "Why."}]
    )
    body = {"claims": claims, "summary": "Summary."}
    return message_response([text_block(json.dumps(body))])


# The examples cover the three chosen golden cases, ask each case's real question as
# its customer, and only use evidence from tools the assistant has.
def test_examples_match_their_golden_cases():
    examples = load_examples()
    golden = {case.id: case for case in load_golden_cases()}

    assert [example.case_id for example in examples] == [
        "both-duplicate-refund",
        "kb-payment-retries",
        "gap-pause-subscription",
    ]
    for example in examples:
        assert example.question == golden[example.case_id].question
        assert example.customer_id == golden[example.case_id].customer_id
        assert {item.tool for item in example.evidence} <= TOOL_NAMES


# Each answer is judged with only the question, the answer and the evidence. The
# explanation of the planted flaw never reaches the judge, so it can't give the
# answer away, and each verdict is paired with the one expected.
def test_check_example_sends_only_question_answer_and_evidence():
    example = load_examples()[0]
    claude = FakeClaude(judgment(), judgment("unsupported"))

    results = check_example(claude.client(), example)

    assert [(kind, expected, result.verdict) for kind, expected, result in results] == [
        ("supported", "pass", "pass"),
        ("flawed", "fail", "fail"),
    ]
    answers = [example.supported_answer, example.flawed_answer]
    for request, answer in zip(claude.requests, answers, strict=True):
        prompt = request["messages"][0]["content"]
        assert example.question in prompt and answer in prompt
        assert all(json.dumps(item.output) in prompt for item in example.evidence)
        assert example.flaw not in prompt
