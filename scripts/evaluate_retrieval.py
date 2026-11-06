"""Measure how well search_knowledge_base finds the right help articles and chunks.

Document level: for each question in eval/retrieval_cases.json, this searches the
knowledge base, ranks the articles by their best-matching chunk, and keeps the first
five different articles. It then checks where the labeled relevant articles appear,
and reports Hit@1, Hit@3, Hit@5 and Mean Reciprocal Rank overall, by category and per
question. A question with several relevant articles counts as a hit if any one of
them appears.

Chunk level: for the questions that also have answer_chunk_ids, this takes the top
five chunks exactly as search returns them, without merging chunks from the same
article. A chunk is relevant only if it's labeled as answering the question, so chunks
that just mention the topic or link to the answer count as irrelevant. It reports
Hit@1, Hit@3, Hit@5, Precision@3, Precision@5 and MRR, and calls out questions where
an irrelevant chunk ranks above a relevant one.

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

TOP_K = 5  # how many different documents, or chunks, each question is judged on
HIT_CUTOFFS = (1, 3, 5)
PRECISION_CUTOFFS = (3, 5)
CATEGORIES = ("direct", "paraphrased", "ambiguous", "multi-document")


class RetrievalCase(BaseModel):
    id: str
    question: str
    relevant_document_ids: list[str] = Field(min_length=1)
    category: Literal["direct", "paraphrased", "ambiguous", "multi-document"]
    # Only some questions have chunk labels. Answer chunks are the relevant ones.
    answer_chunk_ids: list[str] | None = Field(default=None, min_length=1)
    mention_chunk_ids: list[str] = Field(default_factory=list)


@dataclass
class CaseResult:
    case: RetrievalCase
    ranking: list[str]  # different document IDs, best first, at most TOP_K
    chunks_needed: int  # how many chunks it took to find those documents


@dataclass
class ChunkCaseResult:
    case: RetrievalCase
    ranking: list[str]  # chunk IDs as search returned them, best first, at most TOP_K
    similarities: list[float]  # each chunk's cosine similarity to the question

    @property
    def relevant(self) -> list[str]:
        return self.case.answer_chunk_ids or []


@dataclass
class Scores:
    count: int
    hit_rates: dict[int, float]  # cutoff -> share of questions with a hit
    mean_reciprocal_rank: float


@dataclass
class ChunkScores(Scores):
    precisions: dict[int, float]  # cutoff -> average share of relevant chunks


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


# These work the same on a ranking of document IDs or of chunk IDs.


def first_relevant_rank(ranking: list[str], relevant: Iterable[str]) -> int | None:
    """Return the 1-based rank of the first relevant ID, or None if none is."""
    relevant = set(relevant)
    for rank, item_id in enumerate(ranking, start=1):
        if item_id in relevant:
            return rank
    return None


def hit_at(ranking: list[str], relevant: Iterable[str], cutoff: int) -> bool:
    rank = first_relevant_rank(ranking, relevant)
    return rank is not None and rank <= cutoff


def reciprocal_rank(ranking: list[str], relevant: Iterable[str]) -> float:
    rank = first_relevant_rank(ranking, relevant)
    return 0.0 if rank is None else 1 / rank


def precision_at(ranking: list[str], relevant: Iterable[str], cutoff: int) -> float:
    """Return the share of the top cutoff places taken by relevant IDs.

    This always divides by cutoff, so a ranking shorter than cutoff can't score higher
    for having fewer places to fill.
    """
    relevant = set(relevant)
    return sum(item_id in relevant for item_id in ranking[:cutoff]) / cutoff


def irrelevant_above_relevant(ranking: list[str], relevant: Iterable[str]) -> list[str]:
    """Return the irrelevant IDs that rank above at least one relevant ID."""
    relevant = set(relevant)
    last_relevant = max(
        (index for index, item_id in enumerate(ranking) if item_id in relevant),
        default=0,
    )
    return [item_id for item_id in ranking[:last_relevant] if item_id not in relevant]


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


def score_chunks(results: list[ChunkCaseResult]) -> ChunkScores:
    """Average each chunk-level metric over the results."""
    count = len(results)
    hit_rates = {
        cutoff: sum(hit_at(r.ranking, r.relevant, cutoff) for r in results) / count
        for cutoff in HIT_CUTOFFS
    }
    precisions = {
        cutoff: sum(precision_at(r.ranking, r.relevant, cutoff) for r in results)
        / count
        for cutoff in PRECISION_CUTOFFS
    }
    mean_reciprocal_rank = (
        sum(reciprocal_rank(r.ranking, r.relevant) for r in results) / count
    )
    return ChunkScores(count, hit_rates, mean_reciprocal_rank, precisions)


def best_possible_precisions(results: list[ChunkCaseResult]) -> dict[int, float]:
    """Return the highest precision each cutoff could reach with these labels.

    A question with one answer chunk can't score above 1/cutoff, however good search is.
    """
    return {
        cutoff: sum(min(len(r.relevant), cutoff) / cutoff for r in results)
        / len(results)
        for cutoff in PRECISION_CUTOFFS
    }


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


def evaluate_chunks(
    client: QdrantClient, cases: list[RetrievalCase]
) -> list[ChunkCaseResult]:
    """Search for each question with chunk labels and keep the top TOP_K chunks."""
    results = []
    for case in cases:
        if case.answer_chunk_ids is None:
            continue
        chunks = search_knowledge_base(client, case.question, limit=TOP_K)
        results.append(
            ChunkCaseResult(
                case,
                ranking=[chunk.id for chunk in chunks],
                similarities=[chunk.score for chunk in chunks],
            )
        )
    return results


def format_scores(label: str, scores: Scores) -> str:
    values = [scores.hit_rates[cutoff] for cutoff in HIT_CUTOFFS]
    if isinstance(scores, ChunkScores):
        values += [scores.precisions[cutoff] for cutoff in PRECISION_CUTOFFS]
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


def chunk_label(result: ChunkCaseResult, chunk_id: str) -> str:
    if chunk_id in result.relevant:
        return "ANSWER"
    if chunk_id in result.case.mention_chunk_ids:
        return "mention"
    return "-"


def print_chunk_report(results: list[ChunkCaseResult]) -> None:
    print(
        f"Chunk-level evaluation: {len(results)} questions with labeled chunks, "
        f"judged on the top {TOP_K} chunks as search returns them\n"
    )
    header = (
        "".join(f"{f'Hit@{cutoff}':>8}" for cutoff in HIT_CUTOFFS)
        + "".join(f"{f'P@{cutoff}':>8}" for cutoff in PRECISION_CUTOFFS)
        + f"{'MRR':>8}"
    )
    print(f"  {'':<22}{header}")
    print(format_scores(f"All ({len(results)})", score_chunks(results)))
    for category in CATEGORIES:
        in_category = [result for result in results if result.case.category == category]
        if in_category:
            scores = score_chunks(in_category)
            print(format_scores(f"{category} ({len(in_category)})", scores))
    best = ChunkScores(
        len(results),
        {cutoff: 1.0 for cutoff in HIT_CUTOFFS},
        1.0,
        best_possible_precisions(results),
    )
    print(format_scores("Best possible", best))
    print(
        "  (Precision can't reach 1.00 when a question has fewer answer chunks than "
        "places.)"
    )

    print(
        "\nPer question (ANSWER: labeled answer, mention: mentions the topic or links "
        "to the answer, -: other)"
    )
    for result in results:
        rank = first_relevant_rank(result.ranking, result.relevant)
        found = len(set(result.relevant).intersection(result.ranking))
        print(
            f"  {result.case.id}  {result.case.category:<16}"
            f"{'rank ' + str(rank) if rank else 'MISS':<8}"
            f"RR {reciprocal_rank(result.ranking, result.relevant):.2f}  "
            + "  ".join(
                f"P@{cutoff} {precision_at(result.ranking, result.relevant, cutoff):.2f}"
                for cutoff in PRECISION_CUTOFFS
            )
            + f"  found {found}/{len(result.relevant)} answer chunks"
        )
        for position, (chunk_id, similarity) in enumerate(
            zip(result.ranking, result.similarities), start=1
        ):
            print(
                f"         {position}. {chunk_label(result, chunk_id):<8}"
                f"{similarity:.3f}  {chunk_id}"
            )

    print("\nRelevant chunks ranked below irrelevant ones")
    outranked = [
        result
        for result in results
        if irrelevant_above_relevant(result.ranking, result.relevant)
    ]
    if not outranked:
        print("  None")
    for result in outranked:
        above = [
            f"{result.ranking.index(chunk_id) + 1}. {chunk_id} "
            f"({chunk_label(result, chunk_id)})"
            for chunk_id in irrelevant_above_relevant(result.ranking, result.relevant)
        ]
        answers = [
            str(position)
            for position, chunk_id in enumerate(result.ranking, start=1)
            if chunk_id in result.relevant
        ]
        print(
            f"  {result.case.id}  answer chunks at {', '.join(answers)}, "
            "with these irrelevant chunks above them:"
        )
        print("         " + "; ".join(above))

    misses = [
        result
        for result in results
        if first_relevant_rank(result.ranking, result.relevant) is None
    ]
    print(f"\nMisses: no answer chunk in the top {TOP_K}")
    if not misses:
        print("  None")
    for result in misses:
        print(f"  {result.case.id} ({result.case.category}) {result.case.question}")
        print(f"         expected one of: {', '.join(result.relevant)}")

    multi = [result for result in results if result.case.category == "multi-document"]
    print("\nQuestions that need several pieces of information: answer chunks found")
    if not multi:
        print("  None")
    for result in multi:
        retrieved = [chunk for chunk in result.relevant if chunk in result.ranking]
        missing = [chunk for chunk in result.relevant if chunk not in retrieved]
        print(f"  {result.case.id}  {len(retrieved)} of {len(result.relevant)}")
        print(f"         found:   {', '.join(retrieved) or 'none'}")
        print(f"         missing: {', '.join(missing) or 'none'}")


def main(cases_path: Path = CASES_FILE) -> int:
    try:
        cases = load_cases(cases_path)
        client = get_qdrant_client()
        results = evaluate(client, cases)
        chunk_results = evaluate_chunks(client, cases)
    except (EvaluationError, RetrievalError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_report(results)
    if chunk_results:
        print("\n" + "=" * 88 + "\n")
        print_chunk_report(chunk_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
