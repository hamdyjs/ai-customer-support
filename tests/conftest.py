import numpy as np
import pytest
from qdrant_client import QdrantClient
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from app import retrieval
from app.ai import (
    GeneratedReply,
    QuestionAnalysis,
    QuestionCategory,
    SupportReply,
    get_llm_service,
)
from app.db.models import TicketPriority
from app.main import app
from scripts.ingest import (
    EMBEDDING_DIMENSIONS,
    Chunk,
    EmbeddedChunk,
    store_knowledge_base,
)

PASSAGES = [
    Chunk(
        id="cancellation:5",
        document_id="cancellation",
        title="Canceling Your Subscription",
        source="cancellation.md",
        section_heading="Refunds after canceling",
        chunk_index=5,
        text=(
            "Canceling Your Subscription\nRefunds after canceling\n\n"
            "Canceling doesn't refund the current period."
        ),
    ),
    Chunk(
        id="failed-payments:0",
        document_id="failed-payments",
        title="Failed Payments",
        source="failed-payments.md",
        section_heading=None,
        chunk_index=0,
        text="Failed Payments\n\nWe retry a failed payment three times.",
    ),
]
# Every query embeds to this vector, which scores 0.8 against the first passage and
# 0.6 against the second, so the first always ranks higher.
QUERY_VECTOR = [0.8, 0.6] + [0.0] * (EMBEDDING_DIMENSIONS - 2)


class FakeEmbeddingModel:
    def __init__(self):
        self.queries = []

    def encode(self, query):
        self.queries.append(query)
        return np.array(QUERY_VECTOR)


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
        self, message: str, history, new_session, customer_id: str
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


# Searches use an in-memory Qdrant collection holding PASSAGES and a fake embedding
# model, so they need neither Docker nor PyTorch. The model records each query.
@pytest.fixture
def knowledge_base(monkeypatch):
    client = QdrantClient(":memory:")
    store_knowledge_base(
        [
            EmbeddedChunk(
                chunk=chunk,
                embedding=[float(i == position) for i in range(EMBEDDING_DIMENSIONS)],
            )
            for position, chunk in enumerate(PASSAGES)
        ],
        client,
    )
    model = FakeEmbeddingModel()
    monkeypatch.setattr(retrieval, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: model)
    return model
