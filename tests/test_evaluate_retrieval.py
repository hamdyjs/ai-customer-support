from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from app.retrieval import COLLECTION_NAME, EMBEDDING_MODEL, SearchResult
from scripts import evaluate_retrieval
from scripts.evaluate_retrieval import (
    CaseResult,
    ChunkCaseResult,
    EvaluationError,
    RetrievalCase,
    best_possible_precisions,
    check_collection,
    check_labels,
    current_chunks,
    evaluate_chunks,
    first_relevant_rank,
    hit_at,
    irrelevant_above_relevant,
    load_cases,
    precision_at,
    print_chunk_report,
    print_report,
    rank_documents,
    reciprocal_rank,
    score,
    score_chunks,
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


def make_chunk_result(
    ranking, answers, mentions=(), category="direct", id="case", similarities=None
):
    case = RetrievalCase(
        id=id,
        question="A question",
        relevant_document_ids=sorted({chunk.split(":")[0] for chunk in answers}),
        category=category,
        answer_chunk_ids=list(answers),
        mention_chunk_ids=list(mentions),
    )
    if similarities is None:
        similarities = [1.0 - index / 10 for index in range(len(ranking))]
    return ChunkCaseResult(case=case, ranking=ranking, similarities=similarities)


# Precision is the share of the top places that hold relevant chunks.
def test_precision_at_counts_relevant_chunks_in_the_top_places():
    ranking = ["a:1", "x:0", "a:2", "y:0", "a:3"]
    relevant = ["a:1", "a:2", "a:3"]

    assert precision_at(ranking, relevant, 3) == pytest.approx(2 / 3)
    assert precision_at(ranking, relevant, 5) == pytest.approx(3 / 5)


# A relevant chunk just past the cutoff doesn't count towards it.
def test_precision_at_ignores_chunks_past_the_cutoff():
    ranking = ["x:0", "y:0", "z:0", "a:1", "b:0"]

    assert precision_at(ranking, ["a:1"], 3) == 0.0
    assert precision_at(ranking, ["a:1"], 5) == pytest.approx(0.2)


# A short ranking is still divided by the cutoff, so it can't look more precise.
def test_precision_at_divides_by_cutoff_for_short_rankings():
    assert precision_at(["a:1", "a:2"], ["a:1", "a:2"], 5) == pytest.approx(0.4)


# Two chunks from the same article both count, since chunks aren't merged by article.
def test_chunks_from_one_document_count_separately():
    ranking = ["a:1", "a:2", "x:0"]

    assert precision_at(ranking, ["a:1", "a:2"], 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranking, ["a:2"]) == pytest.approx(0.5)


# Irrelevant chunks above any relevant one are reported, but not those below them all.
def test_irrelevant_above_relevant_lists_chunks_that_outrank_answers():
    ranking = ["pointer:0", "a:1", "x:0", "a:2", "y:0"]

    assert irrelevant_above_relevant(ranking, ["a:1", "a:2"]) == ["pointer:0", "x:0"]


# Nothing is reported when the answers come first, or when there are no answers.
@pytest.mark.parametrize(
    "ranking, relevant",
    [
        (["a:1", "a:2", "x:0", "y:0", "z:0"], ["a:1", "a:2"]),
        (["x:0", "y:0", "z:0", "w:0", "v:0"], ["a:1"]),
    ],
)
def test_irrelevant_above_relevant_is_empty_without_outranking(ranking, relevant):
    assert irrelevant_above_relevant(ranking, relevant) == []


# Chunk scores average hits, precision and MRR over the questions.
def test_score_chunks_averages_over_questions():
    results = [
        make_chunk_result(["a:1", "a:2", "x:0", "y:0", "z:0"], ["a:1", "a:2"]),
        make_chunk_result(["x:0", "y:0", "a:1", "z:0", "w:0"], ["a:1"]),
        make_chunk_result(["x:0", "y:0", "z:0", "w:0", "v:0"], ["a:1"]),
    ]

    scores = score_chunks(results)

    assert scores.count == 3
    assert scores.hit_rates == pytest.approx({1: 1 / 3, 3: 2 / 3, 5: 2 / 3})
    assert scores.precisions == pytest.approx(
        {3: (2 / 3 + 1 / 3 + 0) / 3, 5: (2 / 5 + 1 / 5 + 0) / 3}
    )
    assert scores.mean_reciprocal_rank == pytest.approx((1 + 1 / 3 + 0) / 3)


# With one answer chunk, the best possible Precision@5 is 1/5, not 1.
def test_best_possible_precision_is_capped_by_answer_count():
    results = [
        make_chunk_result(["a:1"], ["a:1"]),
        make_chunk_result(["a:1"], ["a:1", "a:2", "a:3", "a:4", "a:5", "a:6"]),
    ]

    assert best_possible_precisions(results) == pytest.approx(
        {3: (1 / 3 + 1) / 2, 5: (1 / 5 + 1) / 2}
    )


# Chunk evaluation searches once for the top five chunks, keeps chunks from the same
# article, and skips questions without chunk labels.
def test_evaluate_chunks_uses_top_five_chunks_as_returned(fake_search):
    limits = fake_search(["a", "a", "b", "a", "c", "d", "e"])
    labeled = make_chunk_result([], ["a:1"]).case
    unlabeled = make_result([], ["a"]).case

    results = evaluate_chunks(None, [labeled, unlabeled])

    assert len(results) == 1
    assert results[0].ranking == ["a:0", "a:1", "b:2", "a:3", "c:4"]
    assert limits == [5]


# The report marks each chunk as an answer, a mention or other.
def test_chunk_report_labels_each_chunk(capsys):
    result = make_chunk_result(
        ["pointer:0", "a:1", "x:0"], ["a:1"], mentions=["pointer:0"], id="rc-x"
    )

    print_chunk_report([result])

    output = capsys.readouterr().out
    assert "1. mention " in output
    assert "2. ANSWER  " in output
    assert "3. -       " in output


# Questions where an irrelevant chunk outranks an answer are called out by name.
def test_chunk_report_highlights_answers_below_irrelevant_chunks(capsys):
    results = [
        make_chunk_result(["a:1", "x:0"], ["a:1"], id="rc-first"),
        make_chunk_result(
            ["pointer:0", "a:1"], ["a:1"], mentions=["pointer:0"], id="rc-below"
        ),
    ]

    print_chunk_report(results)

    section = capsys.readouterr().out.split("ranked below irrelevant ones")[1]
    section = section.split("Misses:")[0]
    assert "rc-below" in section
    assert "1. pointer:0 (mention)" in section
    assert "rc-first" not in section


# Multi-document questions report which answer chunks were found and which weren't.
def test_chunk_report_counts_answer_chunks_for_multi_document_cases(capsys):
    result = make_chunk_result(
        ["a:1", "x:0", "y:0"], ["a:1", "b:1"], category="multi-document", id="rc-m"
    )

    print_chunk_report([result])

    section = capsys.readouterr().out.split("several pieces of information")[1]
    assert "rc-m  1 of 2" in section
    assert "found:   a:1" in section
    assert "missing: b:1" in section


# Questions without chunk labels load as before, with no answer chunks.
def test_cases_without_chunk_labels_load():
    case = RetrievalCase.model_validate(
        {
            "id": "x",
            "question": "q",
            "relevant_document_ids": ["a"],
            "category": "direct",
        }
    )

    assert case.answer_chunk_ids is None
    assert case.mention_chunk_ids == []


CHUNKS = {
    "a:0": "Article A\n\nSee [the steps](a.md#steps).",
    "a:1": "Article A\nSteps\n\nSelect **Start**. It works   right away.",
    "a:2": "Article A\nSteps\n\nworks right away. Something else entirely.",
    "b:0": "Article B\n\nUnrelated text.",
}


def make_labeled_case(evidence, mentions=(), documents=("a",)):
    return RetrievalCase(
        id="case",
        question="A question",
        relevant_document_ids=list(documents),
        category="direct",
        answer_chunk_ids=list(evidence),
        answer_evidence=evidence,
        mention_chunk_ids=list(mentions),
    )


# A consistent label passes, even when the quote's spacing differs from the chunk's.
def test_check_labels_accepts_consistent_labels():
    case = make_labeled_case({"a:1": "Select **Start**."}, mentions=["a:0"])

    assert check_labels([case], CHUNKS) == []


# Each kind of labeling mistake is reported.
@pytest.mark.parametrize(
    "case, problem",
    [
        (
            make_labeled_case({"a:1": "Select **Start**."}, mentions=["a:9"]),
            "a:9 isn't a current chunk",
        ),
        (
            make_labeled_case({"a:1": "Select **Start**."}, mentions=["a:1"]),
            "a:1 is both an answer and a mention",
        ),
        (
            make_labeled_case({"a:1": "Click **Go**."}),
            "a:1 doesn't contain",
        ),
        (
            # The quote also appears in the next chunk's overlap, which isn't labeled.
            make_labeled_case({"a:1": "works right away."}),
            "a:2 contains the answer",
        ),
        (
            # A mention can't hold the answer either.
            make_labeled_case({"a:1": "works right away."}, mentions=["a:2"]),
            "a:2 contains the answer",
        ),
        (
            make_labeled_case({"a:1": "Select **Start**."}, documents=["a", "b"]),
            "the relevant documents are ['a', 'b']",
        ),
    ],
)
def test_check_labels_reports_mistakes(case, problem):
    assert any(problem in found for found in check_labels([case], CHUNKS))


# Every answer chunk needs a quote, and every quote needs an answer chunk.
def test_check_labels_requires_evidence_for_each_answer():
    case = make_labeled_case({"a:1": "Select **Start**."})
    case.answer_chunk_ids = ["a:1", "a:2"]

    assert any(
        "must quote each answer chunk" in p for p in check_labels([case], CHUNKS)
    )


# Questions without chunk labels aren't checked.
def test_check_labels_skips_unlabeled_cases():
    assert check_labels([make_result([], ["a"]).case], CHUNKS) == []


def make_collection(chunks, model=EMBEDDING_MODEL):
    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION_NAME,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        metadata={"embedding_model": model},
    )
    client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=index, vector=[1.0, 0.0], payload={"id": chunk_id, "text": text}
            )
            for index, (chunk_id, text) in enumerate(chunks.items())
        ],
    )
    return client


# A collection holding exactly the current chunks passes.
def test_check_collection_accepts_matching_chunks():
    check_collection(make_collection(CHUNKS), CHUNKS)


# Stale or different data stops the run, saying to re-run the ingest script.
@pytest.mark.parametrize(
    "stored, model, problem",
    [
        (
            {**CHUNKS, "a:1": "Old text."},
            EMBEDDING_MODEL,
            "1 chunks missing or changed",
        ),
        ({k: v for k, v in CHUNKS.items() if k != "b:0"}, EMBEDDING_MODEL, "1 chunks"),
        ({**CHUNKS, "c:0": "Removed article."}, EMBEDDING_MODEL, "1 extra"),
        (CHUNKS, "another-model", "model 'another-model'"),
    ],
)
def test_check_collection_rejects_mismatches(stored, model, problem):
    with pytest.raises(EvaluationError, match="scripts.ingest") as error:
        check_collection(make_collection(stored, model), CHUNKS)

    assert problem in str(error.value)


# A missing collection is a clear error rather than a traceback.
def test_check_collection_reports_missing_collection():
    with pytest.raises(EvaluationError, match="Cannot read"):
        check_collection(QdrantClient(":memory:"), CHUNKS)


# Every question has chunk labels, and they pass every check against the current
# articles: real chunk IDs, quotes found only in answer chunks, and answer chunks
# that cover exactly the question's relevant articles.
def test_dataset_chunk_labels_pass_every_check():
    cases = load_cases()

    assert all(case.answer_chunk_ids is not None for case in cases)
    assert check_labels(cases, current_chunks()) == []
