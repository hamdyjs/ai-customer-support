import types

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app import retrieval
from app.retrieval import search_knowledge_base
from scripts.ingest import (
    EMBEDDING_DIMENSIONS,
    Chunk,
    EmbeddedChunk,
    store_knowledge_base,
)


def axis(position):
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[position] = 1.0
    return vector


# The query leans 0.8 towards invoices:0 and 0.6 towards refunds:0, and is unrelated
# to the other two chunks, so the cosine scores are known exactly.
QUERY_VECTOR = [0.6, 0.0, 0.8, 0.0] + [0.0] * (EMBEDDING_DIMENSIONS - 4)
CHUNK_VECTORS = {
    "refunds:0": axis(0),
    "refunds:1": axis(1),
    "invoices:0": axis(2),
    "invoices:1": axis(3),
}


def make_embedded_chunk(chunk_id, vector):
    document_id, index = chunk_id.split(":")
    return EmbeddedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=document_id,
            title=document_id.title(),
            source=f"{document_id}.md",
            section_heading=f"Section {index}",
            chunk_index=int(index),
            text=f"{document_id.title()}\nSection {index}\n\nText {index}",
        ),
        embedding=vector,
    )


# Every query embeds to QUERY_VECTOR, so these tests don't need PyTorch or a download.
@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    model = types.SimpleNamespace(encode=lambda query: np.array(QUERY_VECTOR))
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: model)


# An in-memory collection built by the ingest script's own storage code, so it's laid
# out exactly like the real one.
@pytest.fixture
def qdrant():
    client = QdrantClient(":memory:")
    store_knowledge_base(
        [make_embedded_chunk(id, vector) for id, vector in CHUNK_VECTORS.items()],
        client,
    )
    return client


# Without a filter, the best matches come from any document, best first.
def test_unfiltered_search_ranks_all_documents(qdrant):
    results = search_knowledge_base(qdrant, "question", limit=2)

    assert [result.id for result in results] == ["invoices:0", "refunds:0"]
    assert [result.score for result in results] == pytest.approx([0.8, 0.6])


# With a document_id, only that document's chunks come back, still ranked by score.
def test_filtered_search_returns_only_that_document(qdrant):
    results = search_knowledge_base(qdrant, "question", document_id="refunds")

    assert [result.id for result in results] == ["refunds:0", "refunds:1"]
    assert [result.score for result in results] == pytest.approx([0.6, 0.0])


# The filter runs inside the search, so a document whose chunks rank below another
# document's still fills the limit instead of coming back empty.
def test_filter_applies_before_the_limit(qdrant):
    results = search_knowledge_base(qdrant, "question", limit=1, document_id="refunds")

    assert [result.id for result in results] == ["refunds:0"]


# Filtered results have the same fields and scores as the same chunks found unfiltered.
def test_filtered_results_match_unfiltered_ones(qdrant):
    unfiltered = search_knowledge_base(qdrant, "question", limit=4)
    filtered = search_knowledge_base(qdrant, "question", document_id="invoices")

    assert filtered == [
        result for result in unfiltered if result.document_id == "invoices"
    ]
    assert filtered[0].model_dump(exclude={"score"}) == {
        "id": "invoices:0",
        "document_id": "invoices",
        "title": "Invoices",
        "source": "invoices.md",
        "section_heading": "Section 0",
        "text": "Invoices\nSection 0\n\nText 0",
    }


# A document_id with no chunks gives an empty list rather than an error.
def test_filter_for_unknown_document_returns_empty_list(qdrant):
    assert search_knowledge_base(qdrant, "question", document_id="missing") == []


# An empty string is treated as a document ID, not as "no filter".
def test_empty_document_id_still_filters(qdrant):
    assert search_knowledge_base(qdrant, "question", document_id="") == []


# An empty collection gives an empty list too.
def test_search_of_empty_collection_returns_empty_list():
    client = QdrantClient(":memory:")
    store_knowledge_base([], client)

    assert search_knowledge_base(client, "question") == []
