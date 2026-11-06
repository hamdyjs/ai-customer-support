"""Search the knowledge base from the command line, to see what retrieval finds.

Qdrant must be running, with the chunks stored by scripts/ingest.py. Pass one or more
queries in quotes, from the repo root:

    uv run python -m scripts.search "I was charged twice" "How do I cancel?"

Add --limit to change how many results each query shows, and --full to print each
result's whole text instead of a short preview. Add --document-id to search only one
article, using its file name in knowledge_base/ without the .md:

    uv run python -m scripts.search "I was charged twice" --document-id cancellation
"""

import argparse
import sys
import textwrap

from app.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    RetrievalError,
    SearchResult,
    get_qdrant_client,
    search_knowledge_base,
)

PREVIEW_CHARACTERS = 120
INDENT = " " * 9  # lines the text up under the chunk ID, after the score


def format_result(result: SearchResult, full: bool) -> str:
    heading = result.section_heading or "(introduction)"
    # The text starts with the title and heading, so leave those out of the preview.
    body = result.text.split("\n\n", 1)[-1]
    if full:
        body = textwrap.indent(body, INDENT)
    else:
        body = INDENT + textwrap.shorten(body, PREVIEW_CHARACTERS, placeholder="...")
    return f"  {result.score:.3f}  {result.id}  {heading}\n{body}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the knowledge base in Qdrant.")
    parser.add_argument(
        "queries", nargs="+", metavar="query", help="text to search for"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"results per query (default: {DEFAULT_SEARCH_LIMIT})",
    )
    parser.add_argument(
        "--full", action="store_true", help="print each result's whole text"
    )
    parser.add_argument(
        "--document-id",
        help="only search this article, by its file name without .md, "
        "such as cancellation",
    )
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        client = get_qdrant_client()
        for query in args.queries:
            results = search_knowledge_base(
                client, query, args.limit, document_id=args.document_id
            )
            if args.document_id is None:
                print(f"\n{query!r}")
            else:
                print(f"\n{query!r} in {args.document_id}")
            if not results:
                print("  No results")
            for result in results:
                print(format_result(result, args.full))
    except RetrievalError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
