"""Load the Markdown articles in knowledge_base/ and list what was found.

This runs on its own and doesn't import or need the FastAPI app.

Run it from the repo root with:

    uv run python -m scripts.ingest
"""

import re
import sys
from pathlib import Path

from pydantic import BaseModel

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"

TITLE_PATTERN = re.compile(r"#\s+(\S.*?)\s*")
FENCE_MARKERS = ("```", "~~~")


class Document(BaseModel):
    id: str
    title: str
    source: str
    content: str


class DocumentLoadError(Exception):
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


def main(directory: Path = KNOWLEDGE_BASE_DIR) -> int:
    try:
        documents = load_documents(directory)
    except DocumentLoadError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for document in documents:
        print(f"{document.id}: {document.title} ({len(document.content)} characters)")
    print(f"Documents loaded from {directory}: {len(documents)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
