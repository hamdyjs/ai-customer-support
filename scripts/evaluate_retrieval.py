"""Measure how well search_knowledge_base finds the right help articles.

For each question in eval/retrieval_cases.json, this searches the knowledge base, ranks
the articles by their best-matching chunk, and keeps the first five different
articles. It then checks where the labeled relevant articles appear, and reports
Hit@1, Hit@3, Hit@5 and Mean Reciprocal Rank overall, by category and per question.
A question with several relevant articles counts as a hit if any one of them appears.

Qdrant must be running, with the chunks stored by scripts/ingest.py. This doesn't use
FastAPI or Claude. Run it from the repo root with:

    uv run python -m scripts.evaluate_retrieval
"""

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from qdrant_client import QdrantClient

from app.retrieval import RetrievalError, get_qdrant_client, search_knowledge_base

CASES_FILE = Path(__file__).resolve().parents[1] / "eval" / "retrieval_cases.json"

TOP_K = 5  # how many different documents each question is judged on
HIT_CUTOFFS = (1, 3, 5)
CATEGORIES = ("direct", "paraphrased", "ambiguous", "multi-document")


class RetrievalCase(BaseModel):
    id: str
    question: str
    relevant_document_ids: list[str] = Field(min_length=1)
    category: Literal["direct", "paraphrased", "ambiguous", "multi-document"]


@dataclass
class CaseResult:
    case: RetrievalCase
    ranking: list[str]  # different document IDs, best first, at most TOP_K
    chunks_needed: int  # how many chunks it took to find those documents


@dataclass
class Scores:
    count: int
    hit_rates: dict[int, float]  # cutoff -> share of questions with a hit
    mean_reciprocal_rank: float


class EvaluationError(Exception):
    pass


def load_cases(path: Path = CASES_FILE) -> list[RetrievalCase]:
    try:
        cases = TypeAdapter(list[RetrievalCase]).validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise EvaluationError(f"Cannot load {path}: {error}") from error
    if not cases:
        raise EvaluationError(f"{path} has no cases")
    return cases


def unique_documents(document_ids: Iterable[str], limit: int) -> list[str]:
    """Keep each document's first appearance, in order, up to limit documents."""
    ranking: list[str] = []
    for document_id in document_ids:
        if document_id not in ranking:
            ranking.append(document_id)
            if len(ranking) == limit:
                break
    return ranking


def first_relevant_rank(ranking: list[str], relevant: Iterable[str]) -> int | None:
    """Return the 1-based rank of the first relevant document, or None if none is."""
    relevant = set(relevant)
    for rank, document_id in enumerate(ranking, start=1):
        if document_id in relevant:
            return rank
    return None


def hit_at(ranking: list[str], relevant: Iterable[str], cutoff: int) -> bool:
    rank = first_relevant_rank(ranking, relevant)
    return rank is not None and rank <= cutoff


def reciprocal_rank(ranking: list[str], relevant: Iterable[str]) -> float:
    rank = first_relevant_rank(ranking, relevant)
    return 0.0 if rank is None else 1 / rank


def score(results: list[CaseResult]) -> Scores:
    """Average each metric over the results."""
    count = len(results)
    hit_rates = {
        cutoff: sum(
            hit_at(result.ranking, result.case.relevant_document_ids, cutoff)
            for result in results
        )
        / count
        for cutoff in HIT_CUTOFFS
    }
    mean_reciprocal_rank = (
        sum(
            reciprocal_rank(result.ranking, result.case.relevant_document_ids)
            for result in results
        )
        / count
    )
    return Scores(count, hit_rates, mean_reciprocal_rank)


def rank_documents(
    client: QdrantClient, question: str, top_k: int = TOP_K
) -> tuple[list[str], int]:
    """Return the first top_k different documents found, and the chunks that took.

    One article often fills several of the top chunks, so this asks for more chunks
    until top_k different documents appear, or the collection has no more to give.
    """
    limit = top_k
    while True:
        results = search_knowledge_base(client, question, limit=limit)
        document_ids = [result.document_id for result in results]
        ranking = unique_documents(document_ids, top_k)
        if len(ranking) == top_k or len(results) < limit:
            chunks_needed = document_ids.index(ranking[-1]) + 1 if ranking else 0
            return ranking, chunks_needed
        limit *= 2


def evaluate(client: QdrantClient, cases: list[RetrievalCase]) -> list[CaseResult]:
    results = []
    for case in cases:
        ranking, chunks_needed = rank_documents(client, case.question)
        results.append(CaseResult(case, ranking, chunks_needed))
    return results


def format_scores(label: str, scores: Scores) -> str:
    values = [scores.hit_rates[cutoff] for cutoff in HIT_CUTOFFS]
    values.append(scores.mean_reciprocal_rank)
    return f"  {label:<22}" + "".join(f"{value:>8.2f}" for value in values)


def print_report(results: list[CaseResult]) -> None:
    print(
        f"Retrieval evaluation: {len(results)} questions, judged on the first "
        f"{TOP_K} different documents for each\n"
    )
    header = "".join(f"{f'Hit@{cutoff}':>8}" for cutoff in HIT_CUTOFFS) + f"{'MRR':>8}"
    print(f"  {'':<22}{header}")
    print(format_scores(f"All ({len(results)})", score(results)))
    for category in CATEGORIES:
        in_category = [result for result in results if result.case.category == category]
        if in_category:
            print(format_scores(f"{category} ({len(in_category)})", score(in_category)))

    print("\nPer question (* marks a relevant document)")
    for result in results:
        relevant = set(result.case.relevant_document_ids)
        rank = first_relevant_rank(result.ranking, relevant)
        found = len(relevant.intersection(result.ranking))
        print(
            f"  {result.case.id}  {result.case.category:<16}"
            f"{'rank ' + str(rank) if rank else 'MISS':<8}"
            f"RR {reciprocal_rank(result.ranking, relevant):.2f}  "
            f"found {found}/{len(relevant)}  "
            f"from {result.chunks_needed} chunks"
        )
        documents = [
            f"{position}. {document_id}{'*' if document_id in relevant else ''}"
            for position, document_id in enumerate(result.ranking, start=1)
        ]
        print("         " + "  ".join(documents))

    misses = [
        result
        for result in results
        if first_relevant_rank(result.ranking, result.case.relevant_document_ids)
        is None
    ]
    print(f"\nMisses: no relevant document in the first {TOP_K} different documents")
    if not misses:
        print("  None")
    for result in misses:
        print(f"  {result.case.id} ({result.case.category}) {result.case.question}")
        print(f"         expected: {', '.join(result.case.relevant_document_ids)}")
        print(f"         got:      {', '.join(result.ranking) or 'nothing'}")

    short = [result for result in results if len(result.ranking) < TOP_K]
    print(f"\nQuestions with fewer than {TOP_K} different documents available")
    if not short:
        print("  None")
    for result in short:
        print(f"  {result.case.id}: only {len(result.ranking)}")


def main(cases_path: Path = CASES_FILE) -> int:
    try:
        cases = load_cases(cases_path)
        results = evaluate(get_qdrant_client(), cases)
    except (EvaluationError, RetrievalError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
