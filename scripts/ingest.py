"""Chunk and embed the articles in knowledge_base/ and store them in Qdrant.

Each run replaces the knowledge_base collection and prints a sample of the chunks.

This runs on its own and doesn't import or need the FastAPI app, but Qdrant must be
running. The first run downloads the embedding model from Hugging Face, then uses a
local cache.

Run it from the repo root with:

    uv run python -m scripts.ingest
"""

import math
import os
import re
import sys
import uuid
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from tokenizers import Encoding, Tokenizer

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "knowledge_base"
# Point IDs are made from chunk IDs within this namespace, so a chunk gets the same ID
# on every run. Any fixed UUID works, as long as it stays the same.
POINT_ID_NAMESPACE = uuid.UUID("c3830e34-8645-48f4-baa0-d72730938742")

TITLE_PATTERN = re.compile(r"#\s+(\S.*?)\s*")
HEADING_PATTERN = re.compile(r"(#{1,6})\s+(\S.*?)\s*")
# The row under a table's header, like "| --- | :--: |". It carries no meaning, but the
# tokenizer counts every "-", so it can use up most of a chunk. Requiring a "|" leaves
# "---" horizontal rules alone.
TABLE_SEPARATOR_PATTERN = re.compile(r"(?=.*-)[\s|:-]*\|[\s|:-]*")
FENCE_MARKERS = ("```", "~~~")
WHITESPACE = re.compile(r"\s+")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
# The model accepts 256 tokens. This limit includes heading context and the [CLS] and
# [SEP] tokens the model adds to every input.
MAX_CHUNK_TOKENS = 200
OVERLAP_TOKENS = 30

# How many chunks the script prints for review.
SAMPLE_CHUNKS = 10

# Where to split text that's too long, from most to least preferred. Each pattern
# matches just after the separator, so the pieces join back into the original text.
SPLIT_BOUNDARIES = (
    re.compile(r"(?<=\n\n)"),  # paragraphs
    re.compile(r"(?<=\n)"),  # lines, such as list items and table rows
    re.compile(r"(?<=[.!?]\s)"),  # sentences
)


class Document(BaseModel):
    id: str
    title: str
    source: str
    content: str


class Chunk(BaseModel):
    id: str  # document_id:chunk_index
    document_id: str
    title: str
    source: str
    section_heading: str | None  # None for text that comes before the first subheading
    chunk_index: int  # position within the document, starting at 0
    text: str  # what gets embedded: heading context followed by the chunk's content


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    embedding: list[float]


class DocumentLoadError(Exception):
    pass


class DocumentChunkError(Exception):
    pass


class DocumentEmbeddingError(Exception):
    pass


class DocumentStoreError(Exception):
    pass


def extract_title(content: str) -> str | None:
    in_code_block = False
    for line in content.splitlines():
        # A "#" inside a code block is usually a comment, not a heading.
        if line.lstrip().startswith(FENCE_MARKERS):
            in_code_block = not in_code_block
        elif not in_code_block and (match := TITLE_PATTERN.fullmatch(line)):
            return match.group(1)
    return None


def load_document(path: Path) -> Document:
    try:
        # newline="" stops Python from rewriting line endings as it reads.
        content = path.read_text(encoding="utf-8", newline="")
    except (OSError, UnicodeError) as error:
        raise DocumentLoadError(f"Cannot read {path}: {error}") from error

    title = extract_title(content)
    if title is None:
        raise DocumentLoadError(f"No level-1 heading found in {path}")

    return Document(id=path.stem, title=title, source=path.name, content=content)


def load_documents(directory: Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    try:
        paths = sorted(path for path in directory.iterdir() if path.suffix == ".md")
    except OSError as error:
        raise DocumentLoadError(
            f"Cannot read knowledge base directory {directory}: {error}"
        ) from error

    return [load_document(path) for path in paths]


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    try:
        tokenizer = Tokenizer.from_pretrained(EMBEDDING_MODEL)
    except Exception as error:
        raise DocumentChunkError(
            f"Cannot load the {EMBEDDING_MODEL} tokenizer: {error}"
        ) from error

    # This model's tokenizer truncates and pads to 128 tokens by default, which would
    # make long text look like it fits.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def count_tokens(text: str) -> int:
    """Count the tokens the model will see, including [CLS] and [SEP]."""
    return len(get_tokenizer().encode(text).ids)


def tokenize_content(text: str) -> Encoding:
    return get_tokenizer().encode(text, add_special_tokens=False)


def split_sections(document: Document) -> list[tuple[str | None, str]]:
    """Return (heading path, body) pairs in document order, leaving out empty bodies."""
    sections: list[tuple[str | None, str]] = []
    headings: list[tuple[int, str]] = []
    lines: list[str] = []
    in_code_block = False

    def finish_section() -> None:
        body = "\n".join(lines).strip()
        if body:
            path = " > ".join(text for _, text in headings)
            sections.append((path or None, body))
        lines.clear()

    for line in document.content.splitlines():
        if line.lstrip().startswith(FENCE_MARKERS):
            in_code_block = not in_code_block
        elif not in_code_block and (match := HEADING_PATTERN.fullmatch(line)):
            finish_section()
            level, text = len(match.group(1)), match.group(2)
            headings = [heading for heading in headings if heading[0] < level]
            # Every chunk already gets the title as context.
            if not (level == 1 and text == document.title):
                headings.append((level, text))
            continue
        elif not in_code_block and TABLE_SEPARATOR_PATTERN.fullmatch(line):
            continue
        lines.append(line)
    finish_section()

    return sections


def split_on_tokens(text: str, limit: int) -> list[str]:
    """Split text with no usable boundaries, preferring to break between words."""
    pieces = []
    while len(offsets := tokenize_content(text).offsets) > limit:
        end = offsets[limit - 1][1]
        space = max(text.rfind(" ", 0, end + 1), text.rfind("\n", 0, end + 1))
        if space > 0:
            end = space + 1
        pieces.append(text[:end])
        text = text[end:]
    pieces.append(text)
    return pieces


def split_to_fit(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most limit tokens that join back into the original."""
    if len(tokenize_content(text).ids) <= limit:
        return [text]

    for boundary in SPLIT_BOUNDARIES:
        parts = [part for part in boundary.split(text) if part]
        if len(parts) > 1:
            return [piece for part in parts for piece in split_to_fit(part, limit)]

    # Leave room for the overlap, which wouldn't otherwise fit beside a full piece.
    return split_on_tokens(text, limit - OVERLAP_TOKENS)


def overlap_tail(text: str, tokens: int) -> str:
    """Return roughly the last few tokens of text, starting at a word if possible."""
    offsets = tokenize_content(text).offsets
    if tokens <= 0 or not offsets:
        return ""

    start = offsets[-tokens:][0][0]
    in_word = start > 0 and not text[start - 1].isspace()
    if in_word and (match := WHITESPACE.search(text, start)):
        start = match.end()
    return text[start:]


def start_with_overlap(previous: str, piece: str, limit: int) -> str:
    """Put as much overlap as fits, up to OVERLAP_TOKENS, in front of the next piece."""
    room = limit - len(tokenize_content(piece).ids)
    # Joining text can change how a split word tokenizes, so check the real total.
    for tokens in range(min(OVERLAP_TOKENS, room), 0, -1):
        candidate = overlap_tail(previous, tokens) + piece
        if len(tokenize_content(candidate).ids) <= limit:
            return candidate
    return piece


def pack_pieces(pieces: list[str], limit: int) -> list[str]:
    """Combine pieces into chunks of at most limit tokens, overlapping between chunks."""
    chunks = []
    current = ""
    for piece in pieces:
        if len(tokenize_content(current + piece).ids) <= limit:
            current += piece
        else:
            chunks.append(current)
            current = start_with_overlap(current, piece, limit)
    chunks.append(current)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Split documents into heading-aware chunks that fit the embedding model."""
    chunks = []
    for document in documents:
        chunk_index = 0
        for heading, body in split_sections(document):
            context = (
                f"{document.title}\n{heading}\n\n"
                if heading
                else f"{document.title}\n\n"
            )
            limit = MAX_CHUNK_TOKENS - count_tokens(context)
            if limit < 2 * OVERLAP_TOKENS:
                raise DocumentChunkError(
                    f"Headings in {document.source} leave only {limit} of "
                    f"{MAX_CHUNK_TOKENS} tokens for content: {context.strip()!r}"
                )

            for text in pack_pieces(split_to_fit(body, limit), limit):
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{chunk_index}",
                        document_id=document.id,
                        title=document.title,
                        source=document.source,
                        section_heading=heading,
                        chunk_index=chunk_index,
                        text=context + text,
                    )
                )
                chunk_index += 1

    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        # Imported here because it loads PyTorch, which is slow and isn't needed when
        # there's nothing to embed.
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as error:
        raise DocumentEmbeddingError(
            f"Cannot load the {EMBEDDING_MODEL} model: {error}"
        ) from error


def generate_embeddings(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Embed each chunk's text and pair it with its vector, skipping blank chunks."""
    chunks = [chunk for chunk in chunks if chunk.text.strip()]
    if not chunks:
        return []

    model = get_embedding_model()
    try:
        vectors = model.encode(
            [chunk.text for chunk in chunks], show_progress_bar=False
        )
    except Exception as error:
        raise DocumentEmbeddingError(
            f"Cannot generate embeddings with {EMBEDDING_MODEL}: {error}"
        ) from error

    # Pairing only by position, so a wrong count would attach vectors to the wrong chunks.
    if len(vectors) != len(chunks):
        raise DocumentEmbeddingError(
            f"Expected {len(chunks)} embeddings, but the model returned {len(vectors)}"
        )

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise DocumentEmbeddingError(
                f"Expected {EMBEDDING_DIMENSIONS} dimensions for chunk "
                f"{chunk.chunk_index} of {chunk.source}, but got {len(vector)}"
            )
        embedding = vector.tolist()
        # Qdrant can't compare NaN or infinity, so searches against this chunk would break.
        if not all(math.isfinite(value) for value in embedding):
            raise DocumentEmbeddingError(
                f"Found non-finite values in the embedding for chunk "
                f"{chunk.chunk_index} of {chunk.source}"
            )
        embedded_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

    return embedded_chunks


def point_id(chunk_id: str) -> str:
    """Turn a chunk ID into a UUID, since Qdrant only accepts integers and UUIDs."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def get_qdrant_client() -> QdrantClient:
    try:
        return QdrantClient(url=QDRANT_URL)
    except Exception as error:
        raise DocumentStoreError(
            f"Cannot create a Qdrant client for {QDRANT_URL}: {error}"
        ) from error


def store_knowledge_base(
    embedded_chunks: list[EmbeddedChunk],
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Replace the collection with a new one holding only the given chunks."""
    points = [
        models.PointStruct(
            id=point_id(embedded.chunk.id),
            vector=embedded.embedding,
            payload=embedded.chunk.model_dump(),
        )
        for embedded in embedded_chunks
    ]

    try:
        # Starting from an empty collection means chunks from deleted or shortened
        # articles can't linger. Until the upload finishes, searches find nothing.
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE
            ),
            # Searches must embed queries with the same model, so record which one
            # made these vectors.
            metadata={"embedding_model": EMBEDDING_MODEL},
        )
        # Qdrant rejects an upload with no points.
        if points:
            client.upsert(collection_name, points=points, wait=True)
    except Exception as error:
        raise DocumentStoreError(
            f"Cannot store chunks in the {collection_name} Qdrant collection: {error}"
        ) from error


def main(directory: Path = KNOWLEDGE_BASE_DIR) -> int:
    try:
        documents = load_documents(directory)
        chunks = chunk_documents(documents)
        embedded_chunks = generate_embeddings(chunks)
        # Stored last, so a failure in any earlier step leaves the collection as it was.
        store_knowledge_base(embedded_chunks, get_qdrant_client())
    except (
        DocumentLoadError,
        DocumentChunkError,
        DocumentEmbeddingError,
        DocumentStoreError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # An even spread across the articles, so the sample isn't all from the first one.
    step = max(1, len(embedded_chunks) // SAMPLE_CHUNKS)
    for embedded in embedded_chunks[::step][:SAMPLE_CHUNKS]:
        text = embedded.chunk.text
        print(
            f"--- {count_tokens(text)} tokens, "
            f"{len(embedded.embedding)} dimensions ---\n{text}\n"
        )
    print(
        f"Stored {len(embedded_chunks)} chunks in the {COLLECTION_NAME} collection "
        f"at {QDRANT_URL}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
