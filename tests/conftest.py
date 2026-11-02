import pytest

from app.ai import QuestionAnalysis, QuestionCategory, SupportReply, get_llm_service
from app.db.models import TicketPriority
from app.main import app


class FakeLLMService:
    def __init__(self):
        self.questions = []
        self.reply_calls = []
        self.error = None
        self.analysis = QuestionAnalysis(
            category=QuestionCategory.billing,
            priority=TicketPriority.high,
            summary="The customer was charged twice.",
            suggested_response="We're looking into this for you.",
            human_review_recommended=True,
        )
        self.reply = SupportReply(
            reply="You were charged $49.00 on November 1.",
            analysis=self.analysis,
        )

    async def analyze_question(self, question: str) -> QuestionAnalysis:
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.analysis

    async def generate_reply(
        self, message: str, session, customer_id: str
    ) -> SupportReply:
        self.reply_calls.append((message, customer_id))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture
def fake_llm():
    service = FakeLLMService()
    app.dependency_overrides[get_llm_service] = lambda: service
    yield service
    app.dependency_overrides.clear()
