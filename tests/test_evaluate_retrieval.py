from pathlib import Path

import pytest

from app.retrieval import SearchResult
from scripts import evaluate_retrieval
from scripts.evaluate_retrieval import (
    CaseResult,
    EvaluationError,
    RetrievalCase,
    first_relevant_rank,
    hit_at,
    load_cases,
    print_report,
    rank_documents,
    reciprocal_rank,
    score,
    unique_documents,
)

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"


def make_result(ranking, relevant, category="direct", id="case"):
    case = RetrievalCase(
        id=id,
        question="A question",
        relevant_document_ids=relevant,
        category=category,
    )
    return CaseResult(case=case, ranking=ranking, chunks_needed=len(ranking))


def make_search_result(document_id, index):
    return SearchResult(
        id=f"{document_id}:{index}",
        document_id=document_id,
        title=document_id.title(),
        source=f"{document_id}.md",
        section_heading=None,
        text="Text",
        score=1.0 - index / 100,
    )


# Stands in for search_knowledge_base, returning the given chunks' documents in order
# and recording each limit it was asked for.
@pytest.fixture
def fake_search(monkeypatch):
    def install(document_ids):
        chunks = [
            make_search_result(document_id, index)
            for index, document_id in enumerate(document_ids)
        ]
        limits = []

        def search(client, question, limit):
            limits.append(limit)
            return chunks[:limit]

        monkeypatch.setattr(evaluate_retrieval, "search_knowledge_base", search)
        return limits

    return install


# Later chunks from an already-ranked document are dropped, keeping first-seen order.
def test_unique_documents_keep_first_appearance_order():
    chunks = ["refunds", "refunds", "invoices", "refunds", "cancellation"]

    assert unique_documents(chunks, 5) == ["refunds", "invoices", "cancellation"]


# Ranking stops once it has enough different documents.
def test_unique_documents_stop_at_the_limit():
    assert unique_documents(["a", "b", "a", "c", "d", "e", "f"], 3) == ["a", "b", "c"]


# A relevant document at rank 3 misses Hit@1 but counts for Hit@3 and Hit@5.
def test_hit_at_each_cutoff():
    ranking = ["a", "b", "relevant", "c", "d"]

    assert first_relevant_rank(ranking, ["relevant"]) == 3
    assert not hit_at(ranking, ["relevant"], 1)
    assert hit_at(ranking, ["relevant"], 3)
    assert hit_at(ranking, ["relevant"], 5)


# The reciprocal rank is one over the first relevant document's rank.
@pytest.mark.parametrize("rank, expected", [(1, 1.0), (2, 0.5), (4, 0.25), (5, 0.2)])
def test_reciprocal_rank(rank, expected):
    ranking = ["a", "b", "c", "d", "e"]
    ranking[rank - 1] = "relevant"

    assert reciprocal_rank(ranking, ["relevant"]) == pytest.approx(expected)


# No relevant document gives a miss at every cutoff and a reciprocal rank of 0.
def test_no_relevant_document_scores_zero():
    ranking = ["a", "b", "c", "d", "e"]

    assert first_relevant_rank(ranking, ["relevant"]) is None
    assert not hit_at(ranking, ["relevant"], 5)
    assert reciprocal_rank(ranking, ["relevant"]) == 0.0


# With several relevant documents, any one is a hit and the earliest sets the rank.
def test_multi_document_case_uses_first_relevant_document():
    ranking = ["a", "second", "b", "first", "c"]
    relevant = ["first", "second"]

    assert hit_at(ranking, relevant, 3)
    assert reciprocal_rank(ranking, relevant) == pytest.approx(0.5)


# Repeated chunks from one article don't push the next article down the ranking.
def test_duplicate_chunks_count_once_for_rank():
    chunks = ["a", "a", "a", "b", "relevant"]

    ranking = unique_documents(chunks, 5)

    assert reciprocal_rank(ranking, ["relevant"]) == pytest.approx(1 / 3)
    assert hit_at(ranking, ["relevant"], 3)


# Scores are averaged over the questions: here ranks 1 and 3 and one miss.
def test_score_averages_over_questions():
    results = [
        make_result(["relevant", "a", "b", "c", "d"], ["relevant"]),
        make_result(["a", "b", "relevant", "c", "d"], ["relevant"]),
        make_result(["a", "b", "c", "d", "e"], ["relevant"]),
    ]

    scores = score(results)

    assert scores.count == 3
    assert scores.hit_rates == pytest.approx({1: 1 / 3, 3: 2 / 3, 5: 2 / 3})
    assert scores.mean_reciprocal_rank == pytest.approx((1 + 1 / 3 + 0) / 3)


# When the top chunks repeat one article, more chunks are fetched until five
# different documents appear.
def test_rank_documents_fetches_more_chunks_for_five_documents(fake_search):
    limits = fake_search(["a"] * 6 + ["b", "a", "c", "d", "e", "f"])

    ranking, chunks_needed = rank_documents(None, "question")

    assert ranking == ["a", "b", "c", "d", "e"]
    assert chunks_needed == 11
    assert limits == [5, 10, 20]


# A collection with fewer than five documents returns what it has, instead of looping.
def test_rank_documents_stops_when_chunks_run_out(fake_search):
    limits = fake_search(["a", "a", "b", "c", "b", "a", "c"])

    ranking, chunks_needed = rank_documents(None, "question")

    assert ranking == ["a", "b", "c"]
    assert chunks_needed == 4
    assert limits == [5, 10]


# Five different documents in the first five chunks need only one search.
def test_rank_documents_searches_once_when_enough(fake_search):
    limits = fake_search(["a", "b", "c", "d", "e", "a"])

    assert rank_documents(None, "question") == (["a", "b", "c", "d", "e"], 5)
    assert limits == [5]


# The report calls out every question with no relevant document in its top five.
def test_report_highlights_misses(capsys):
    results = [
        make_result(["relevant", "a", "b", "c", "d"], ["relevant"], id="rc-hit"),
        make_result(["a", "b", "c", "d", "e"], ["relevant"], id="rc-miss"),
    ]

    print_report(results)

    misses = capsys.readouterr().out.split("Misses:")[1]
    assert "rc-miss" in misses
    assert "rc-hit" not in misses


# The report says which questions had fewer than five documents to judge.
def test_report_flags_short_rankings(capsys):
    print_report([make_result(["relevant", "a"], ["relevant"], id="rc-short")])

    assert "rc-short: only 2" in capsys.readouterr().out


# The real dataset loads, its IDs are unique, and every label is a real article.
def test_dataset_labels_existing_articles():
    cases = load_cases()
    articles = {path.stem for path in KNOWLEDGE_BASE_DIR.glob("*.md")}

    assert len({case.id for case in cases}) == len(cases)
    for case in cases:
        assert set(case.relevant_document_ids) <= articles, case.id


# A malformed dataset is a clear error rather than a traceback.
def test_invalid_dataset_raises_clear_error(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"id": "x", "question": "q", "relevant_document_ids": ["a"], '
        '"category": "unknown"}]',
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="Cannot load"):
        load_cases(path)
