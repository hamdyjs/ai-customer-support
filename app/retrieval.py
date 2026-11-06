"""Search the help articles that scripts/ingest.py stores in Qdrant.

The ingest script imports the settings below, so storing and searching always use the
same Qdrant collection and embedding model.
"""

import os
from functools import lru_cache

from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "knowledge_base"
# Questions must be embedded with the same model as the chunks, or their vectors
# can't be compared.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_SEARCH_LIMIT = 5


class SearchResult(BaseModel):
    id: str  # the chunk's ID: document_id:chunk_index
    document_id: str
    title: str
    source: str
    section_heading: str | None
    text: str  # the chunk's heading context followed by its content
    score: float  # cosine similarity to the query, higher is closer


class RetrievalError(Exception):
    pass


@lru_cache
def get_qdrant_client() -> QdrantClient:
    try:
        return QdrantClient(url=QDRANT_URL)
    except Exception as error:
        raise RetrievalError(
            f"Cannot create a Qdrant client for {QDRANT_URL}: {error}"
        ) from error


@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        # Imported here because it loads PyTorch, which is slow, so only the first
        # search pays for it.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as error:
        raise RetrievalError(
            f"Cannot load the {EMBEDDING_MODEL} model: {error}"
        ) from error


def search_knowledge_base(
    client: QdrantClient,
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    document_id: str | None = None,
) -> list[SearchResult]:
    """Return the chunks closest in meaning to the query, best match first.

    With a document_id, only that article's chunks are searched.
    """
    if not query.strip():
        return []

    # Qdrant applies the filter during the search, so the results are the best matches
    # within the document, rather than whatever is left of the global top results.
    query_filter = None
    if document_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                )
            ]
        )

    model = get_embedding_model()
    try:
        vector = model.encode(query).tolist()
    except Exception as error:
        raise RetrievalError(
            f"Cannot embed the query with {EMBEDDING_MODEL}: {error}"
        ) from error

    try:
        response = client.query_points(
            COLLECTION_NAME, query=vector, query_filter=query_filter, limit=limit
        )
    except Exception as error:
        if isinstance(error, UnexpectedResponse) and error.status_code == 404:
            message = (
                f"The {COLLECTION_NAME} collection doesn't exist. Create it with: "
                "uv run python -m scripts.ingest"
            )
        else:
            message = f"Cannot search the {COLLECTION_NAME} Qdrant collection: {error}"
        raise RetrievalError(message) from error

    return [
        SearchResult.model_validate({**(point.payload or {}), "score": point.score})
        for point in response.points
    ]
