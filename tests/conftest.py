import pytest
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from app.ai import (
    GeneratedReply,
    QuestionAnalysis,
    QuestionCategory,
    SupportReply,
    get_llm_service,
)
from app.db.models import TicketPriority
from app.main import app


class FakeLLMService:
    def __init__(self):
        self.questions = []
        self.reply_calls = []
        self.histories = []
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
        self, message: str, history, session, customer_id: str
    ) -> GeneratedReply:
        self.reply_calls.append((message, customer_id))
        self.histories.append(history)
        if self.error:
            raise self.error
        return GeneratedReply(
            reply=self.reply,
            llm_messages=[
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.reply.reply}],
                },
            ],
        )


@pytest.fixture
def fake_llm():
    service = FakeLLMService()
    app.dependency_overrides[get_llm_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


@pytest.fixture
def engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    SQLModel.metadata.create_all(engine)
    return engine
