import pytest

from app.ai import QuestionAnalysis, QuestionCategory, get_llm_service
from app.db.models import TicketPriority
from app.main import app


class FakeLLMService:
    def __init__(self):
        self.questions = []
        self.error = None
        self.analysis = QuestionAnalysis(
            category=QuestionCategory.billing,
            priority=TicketPriority.high,
            summary="The customer was charged twice.",
            suggested_response="We're looking into this for you.",
            human_review_recommended=True,
        )

    async def analyze_question(self, question: str) -> QuestionAnalysis:
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.analysis


@pytest.fixture
def fake_llm():
    service = FakeLLMService()
    app.dependency_overrides[get_llm_service] = lambda: service
    yield service
    app.dependency_overrides.clear()
