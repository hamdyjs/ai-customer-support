import itertools
import json
import sys
import types

import numpy as np
import pytest

from scripts import ingest
from scripts.ingest import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MAX_CHUNK_TOKENS,
    OVERLAP_TOKENS,
    SAMPLE_CHUNKS,
    Chunk,
    Document,
    DocumentChunkError,
    DocumentEmbeddingError,
    DocumentLoadError,
    DocumentStoreError,
    EmbeddedChunk,
    KnowledgeBase,
    chunk_documents,
    count_tokens,
    generate_embeddings,
    get_embedding_model,
    load_documents,
    main,
    save_knowledge_base,
)


# Stands in for the real model, so these tests don't need PyTorch or a large download.
# Each vector is filled with its text's position, to show which text it belongs to.
class FakeModel:
    def __init__(self):
        self.texts = []

    def encode(self, texts, show_progress_bar=True):
        self.texts.extend(texts)
        return np.array(
            [[float(index)] * EMBEDDING_DIMENSIONS for index in range(len(texts))]
        )


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(ingest, "get_embedding_model", lambda: model)
    return model


# Fails the test if anything tries to load the model.
@pytest.fixture
def model_must_not_load(monkeypatch):
    def fail():
        pytest.fail("The embedding model should not have been loaded")

    monkeypatch.setattr(ingest, "get_embedding_model", fail)


# Each field is filled in from the file's name, heading and text.
def test_loads_document_fields(tmp_path):
    content = "# Refund Policy\n\nSome text.\n"
    (tmp_path / "refund-policy.md").write_text(content, encoding="utf-8")

    assert load_documents(tmp_path) == [
        Document(
            id="refund-policy",
            title="Refund Policy",
            source="refund-policy.md",
            content=content,
        )
    ]


# Documents come back sorted by filename, whatever order they were created in.
def test_loads_documents_in_filename_order(tmp_path):
    for name in ["c", "a", "b"]:
        (tmp_path / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")

    assert [document.id for document in load_documents(tmp_path)] == ["a", "b", "c"]


# Files without the .md extension aren't articles, so they're left out.
def test_ignores_other_file_types(tmp_path):
    (tmp_path / "article.md").write_text("# Article\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("# Notes\n", encoding="utf-8")

    assert [document.id for document in load_documents(tmp_path)] == ["article"]


# An empty directory is a valid result, not an error.
def test_empty_directory_returns_no_documents(tmp_path):
    assert load_documents(tmp_path) == []


# The title can come after other text, and only the first level-1 heading counts.
def test_title_is_first_level_1_heading(tmp_path):
    content = "Intro\n\n## Subheading\n\n# The Title\n\n# Another\n"
    (tmp_path / "article.md").write_text(content, encoding="utf-8")

    assert load_documents(tmp_path)[0].title == "The Title"


# A "#" comment inside a code block must not be mistaken for the title.
def test_title_ignores_code_blocks(tmp_path):
    content = "```bash\n# a comment\n```\n\n# The Title\n"
    (tmp_path / "article.md").write_text(content, encoding="utf-8")

    assert load_documents(tmp_path)[0].title == "The Title"


# Content is unchanged, including Markdown syntax, blank lines and CRLF line endings.
def test_preserves_original_content(tmp_path):
    content = "# Title\r\n\r\n**Bold** | `code`\r\n\r\n\r\n  - item  \r\n"
    (tmp_path / "article.md").write_bytes(content.encode("utf-8"))

    document = load_documents(tmp_path)[0]

    assert document.title == "Title"
    assert document.content == content


# A missing directory gives a clear error that names the path.
def test_missing_directory_raises_clear_error(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(
        DocumentLoadError, match="Cannot read knowledge base directory"
    ) as error:
        load_documents(missing)

    assert str(missing) in str(error.value)


# A file in place of the directory gets the same clear error.
def test_path_that_is_a_file_raises_clear_error(tmp_path):
    path = tmp_path / "article.md"
    path.write_text("# Title\n", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Cannot read knowledge base directory"):
        load_documents(path)


# A file that can't be decoded as UTF-8 is reported by path.
def test_non_utf8_file_raises_clear_error(tmp_path):
    path = tmp_path / "broken.md"
    path.write_bytes(b"# Title\n\xff\xfe\n")

    with pytest.raises(DocumentLoadError, match="Cannot read") as error:
        load_documents(tmp_path)

    assert str(path) in str(error.value)


# A file the OS can't read is reported by path. A directory named like an article
# fails on read for any user, unlike file permissions, which root ignores.
def test_unreadable_file_raises_clear_error(tmp_path):
    path = tmp_path / "broken.md"
    path.mkdir()

    with pytest.raises(DocumentLoadError, match="Cannot read") as error:
        load_documents(tmp_path)

    assert str(path) in str(error.value)


# A file with no level-1 heading is reported by path. The cases are: only a subheading,
# a heading with no text or only spaces, a heading only in a code block, a "#" with no
# space after it, and an empty file.
@pytest.mark.parametrize(
    "content",
    ["## Subheading\n", "# \n", "#    \n", "```\n# comment\n```\n", "#NoSpace\n", ""],
)
def test_missing_title_raises_clear_error(tmp_path, content):
    path = tmp_path / "untitled.md"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="No level-1 heading found") as error:
        load_documents(tmp_path)

    assert str(path) in str(error.value)


# The script prints chunks for review, without the old list of files and titles.
def test_main_prints_chunks(tmp_path, capsys, fake_model):
    (tmp_path / "article.md").write_text("# Article\n\nSome text.\n", encoding="utf-8")

    assert main(tmp_path, tmp_path / "output.json") == 0

    output = capsys.readouterr().out
    assert "Article\n\nSome text." in output
    assert f"{EMBEDDING_DIMENSIONS} dimensions" in output
    assert "article:" not in output
    assert "Documents loaded" not in output


# The sample is capped and spread across all the chunks, not just the first ones.
def test_main_prints_an_even_sample(tmp_path, capsys, fake_model):
    for number in range(100):
        (tmp_path / f"{number:03}.md").write_text(
            f"# Article {number:03}\n\nText.\n", encoding="utf-8"
        )

    assert main(tmp_path, tmp_path / "output.json") == 0

    output = capsys.readouterr().out
    assert output.count("--- ") == SAMPLE_CHUNKS
    assert "Article 000" in output
    assert "Article 090" in output


# An embedding failure is reported the same way as a loading failure.
def test_main_reports_embedding_errors(tmp_path, capsys, monkeypatch):
    def fail():
        raise DocumentEmbeddingError("Cannot load the model")

    monkeypatch.setattr(ingest, "get_embedding_model", fail)
    (tmp_path / "article.md").write_text("# Article\n\nSome text.\n", encoding="utf-8")

    assert main(tmp_path) == 1

    assert "Cannot load the model" in capsys.readouterr().err


# On failure the script prints the error to stderr and exits with status 1.
def test_main_reports_errors(tmp_path, capsys):
    assert main(tmp_path / "missing") == 1

    assert "Cannot read knowledge base directory" in capsys.readouterr().err


def make_document(content, id="article", title="Article"):
    return Document(id=id, title=title, source=f"{id}.md", content=content)


# The chunk's text without the title and heading lines in front of it.
def body_of(chunk):
    return chunk.text.split("\n\n", 1)[1]


# Unique words, so tests can check that each one survives chunking, in order.
def numbered_words(count):
    return [f"item{number}x" for number in range(count)]


def assert_all_words_in_order(chunks, words):
    found = [word for chunk in chunks for word in body_of(chunk).split()]
    # Overlap repeats words, so remove repeats by keeping the first of each.
    assert list(dict.fromkeys(found)) == words


# Short sections each become one chunk carrying their metadata and heading context.
def test_chunks_normal_sections():
    content = "# Article\n\nIntro text.\n\n## Billing\n\nBilling text.\n\n## Access\n\nAccess text.\n"

    chunks = chunk_documents([make_document(content)])

    assert [chunk.model_dump() for chunk in chunks] == [
        {
            "id": "article:0",
            "document_id": "article",
            "title": "Article",
            "source": "article.md",
            "section_heading": None,
            "chunk_index": 0,
            "text": "Article\n\nIntro text.",
        },
        {
            "id": "article:1",
            "document_id": "article",
            "title": "Article",
            "source": "article.md",
            "section_heading": "Billing",
            "chunk_index": 1,
            "text": "Article\nBilling\n\nBilling text.",
        },
        {
            "id": "article:2",
            "document_id": "article",
            "title": "Article",
            "source": "article.md",
            "section_heading": "Access",
            "chunk_index": 2,
            "text": "Article\nAccess\n\nAccess text.",
        },
    ]


# A subheading keeps its parents as context, and a new section drops ones that ended.
def test_section_heading_includes_parent_headings():
    content = "# Article\n## Plans\n### Annual\nA.\n### Monthly\nB.\n## Other\nC.\n"

    chunks = chunk_documents([make_document(content)])

    assert [chunk.section_heading for chunk in chunks] == [
        "Plans > Annual",
        "Plans > Monthly",
        "Other",
    ]


# A "#" comment in a code block stays in the section instead of starting a new one.
def test_code_block_comment_is_not_a_heading():
    content = "# Article\n## Setup\n```bash\n# a comment\nrun\n```\n"

    chunks = chunk_documents([make_document(content)])

    assert len(chunks) == 1
    assert chunks[0].section_heading == "Setup"
    assert body_of(chunks[0]) == "```bash\n# a comment\nrun\n```"


# Table separator rows are left out of the chunk, while the header and data rows stay.
@pytest.mark.parametrize(
    "separator",
    [
        "| --- | --- |",
        "|---|---|",
        "--- | ---",
        "| :--- | ---: |",
        "| ------- | :-----: |",
    ],
)
def test_removes_table_separator_rows(separator):
    content = f"# Article\n## Plans\n| Plan | Price |\n{separator}\n| Pro | $49 |\n"

    chunks = chunk_documents([make_document(content)])

    assert body_of(chunks[0]) == "| Plan | Price |\n| Pro | $49 |"


# Removing separator rows doesn't touch the Document's original content.
def test_removing_table_separator_keeps_document_content():
    content = "# Article\n| Plan |\n| --- |\n| Pro |\n"
    document = make_document(content)

    chunk_documents([document])

    assert document.content == content


# Lines that only look similar are real content: a horizontal rule, a row of text
# containing a hyphen, and a separator shown as an example inside a code block.
@pytest.mark.parametrize(
    "line",
    ["---", "| Pro-rated | $5 |", "```\n| --- | --- |\n```"],
)
def test_keeps_lines_that_are_not_table_separators(line):
    chunks = chunk_documents([make_document(f"# Article\n## Section\nText.\n{line}\n")])

    assert body_of(chunks[0]) == f"Text.\n{line}"


# Windows line endings don't stop headings from being recognised.
def test_handles_crlf_headings():
    chunks = chunk_documents([make_document("# Article\r\n## Billing\r\nText.\r\n")])

    assert chunks[0].section_heading == "Billing"
    assert body_of(chunks[0]) == "Text."


# IDs combine the document and chunk position, so they stay unique across documents.
def test_chunk_ids_combine_document_and_position():
    documents = [
        make_document("# B\n## One\nText.\n## Two\nText.\n", id="b", title="B"),
        make_document("# A\n## One\nText.\n", id="a", title="A"),
    ]

    chunks = chunk_documents(documents)

    assert [chunk.id for chunk in chunks] == ["b:0", "b:1", "a:0"]


# Documents stay in order, and the chunk position restarts for each one.
def test_keeps_document_order_and_position():
    documents = [
        make_document("# B\n## One\nText.\n## Two\nText.\n", id="b", title="B"),
        make_document("# A\n## One\nText.\n", id="a", title="A"),
    ]

    chunks = chunk_documents(documents)

    assert [(chunk.document_id, chunk.chunk_index) for chunk in chunks] == [
        ("b", 0),
        ("b", 1),
        ("a", 0),
    ]


# An oversized section breaks between paragraphs that fit, keeping each one whole.
def test_oversized_section_keeps_paragraphs_whole():
    paragraphs = [
        " ".join(numbered_words(40)[:40]) + f" end{number}." for number in range(6)
    ]
    content = "# Article\n## Long\n\n" + "\n\n".join(paragraphs)

    chunks = chunk_documents([make_document(content)])

    assert len(chunks) > 1
    for paragraph in paragraphs:
        assert any(paragraph in body_of(chunk) for chunk in chunks)
    assert all(chunk.section_heading == "Long" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


# An oversized list breaks between items, so no item is cut in half.
def test_oversized_list_breaks_between_lines():
    items = [f"- Step {word} does one thing" for word in numbered_words(100)]
    content = "# Article\n## Steps\n" + "\n".join(items)

    chunks = chunk_documents([make_document(content)])

    assert len(chunks) > 1
    lines = [line for chunk in chunks[1:] for line in body_of(chunk).splitlines()[1:]]
    assert set(lines) <= set(items)


# An oversized paragraph breaks at the end of a sentence, and every word survives.
def test_oversized_paragraph_breaks_between_sentences():
    words = numbered_words(300)
    sentences = [
        " ".join(words[start : start + 10]) + "." for start in range(0, 300, 10)
    ]
    content = "# Article\n" + " ".join(sentences)

    chunks = chunk_documents([make_document(content)])

    assert len(chunks) > 1
    assert all(body_of(chunk).endswith(".") for chunk in chunks)
    found = [word.rstrip(".") for chunk in chunks for word in body_of(chunk).split()]
    assert list(dict.fromkeys(found)) == words


# Consecutive chunks of a split section share about 30 tokens of text.
def test_oversized_section_overlaps_chunks():
    words = numbered_words(500)
    chunks = chunk_documents([make_document("# Article\n" + " ".join(words))])

    assert len(chunks) > 2
    for previous, following in itertools.pairwise(chunks):
        previous_words = body_of(previous).split()
        first_word = body_of(following).split()[0]
        shared = previous_words[previous_words.index(first_word) :]
        assert body_of(following).split()[: len(shared)] == shared
        # -2 leaves out [CLS] and [SEP]. Starting on a whole word can lose a few tokens.
        assert (
            OVERLAP_TOKENS - 6 <= count_tokens(" ".join(shared)) - 2 <= OVERLAP_TOKENS
        )


# Sections are separate topics, so short ones don't overlap with each other.
def test_short_sections_do_not_overlap():
    chunks = chunk_documents(
        [make_document("# Article\n## A\nFirst.\n## B\nSecond.\n")]
    )

    assert [body_of(chunk) for chunk in chunks] == ["First.", "Second."]


# Text with no spaces still gets split, and joining the chunks restores all of it.
def test_splits_long_text_without_spaces():
    text = "-".join(numbered_words(400))

    chunks = chunk_documents([make_document("# Article\n" + text)])

    assert len(chunks) > 1
    assert all(count_tokens(chunk.text) <= MAX_CHUNK_TOKENS for chunk in chunks)
    joined = body_of(chunks[0])
    for chunk in chunks[1:]:
        body = body_of(chunk)
        # Find where this chunk's new text starts, after what it shares with the last.
        overlap = next(
            size for size in range(len(body), -1, -1) if joined.endswith(body[:size])
        )
        joined += body[overlap:]
    assert joined == text


# Documents with nothing to embed produce no chunks, not empty ones.
@pytest.mark.parametrize(
    "content",
    ["", "   \n\n  \n", "# Article\n", "# Article\n\n## Empty\n\n### Also empty\n\n"],
)
def test_empty_content_produces_no_chunks(content):
    assert chunk_documents([make_document(content)]) == []


# No documents at all is also fine.
def test_no_documents_produces_no_chunks():
    assert chunk_documents([]) == []


# A heading with no text directly under it is skipped, but its child sections aren't.
def test_empty_section_between_others_is_skipped():
    chunks = chunk_documents([make_document("# Article\n## Empty\n\n## Full\nText.\n")])

    assert [chunk.section_heading for chunk in chunks] == ["Full"]
    assert [chunk.chunk_index for chunk in chunks] == [0]


# The count must not stop at the tokenizer's default of 128 tokens, and must include
# [CLS] and [SEP]. Otherwise long text would appear to fit.
def test_count_tokens_is_not_truncated():
    assert count_tokens("word " * 1000) == 1002


# No chunk goes over the limit, including heading context and special tokens, for
# each way a section can be too long. Every word must still be present.
@pytest.mark.parametrize(
    "separator",
    [" ", "\n", "\n\n", ". ", "-", "/"],
    ids=["spaces", "lines", "paragraphs", "sentences", "hyphens", "slashes"],
)
def test_chunks_stay_within_token_limit(separator):
    words = numbered_words(1500)
    heading = "A fairly long section heading with several extra words in it"
    content = (
        f"# A Long Article Title\n## {heading}\n### Another Subheading\n"
        + separator.join(words)
    )

    chunks = chunk_documents([make_document(content, title="A Long Article Title")])

    assert len(chunks) > 1
    # A limit of 200 could pass by accident if most chunks were tiny.
    assert max(count_tokens(chunk.text) for chunk in chunks) > MAX_CHUNK_TOKENS * 0.8
    assert all(count_tokens(chunk.text) <= MAX_CHUNK_TOKENS for chunk in chunks)
    assert all(chunk.text.startswith("A Long Article Title\n") for chunk in chunks)

    body = "\n".join(body_of(chunk) for chunk in chunks)
    positions = [body.find(word) for word in words]
    assert -1 not in positions


# A single word too long for the tokenizer's vocabulary is kept, not dropped.
def test_keeps_extremely_long_single_word():
    word = "a" * 5000

    chunks = chunk_documents([make_document("# Article\n" + word)])

    assert len(chunks) == 1
    assert body_of(chunks[0]) == word


# Headings that leave too little room for content are an error, not cut-off text.
def test_oversized_heading_context_raises_clear_error():
    title = "word " * 190

    with pytest.raises(DocumentChunkError, match="article.md"):
        chunk_documents([make_document("Some text.", title=title)])


def make_chunk(text, chunk_index=0):
    return Chunk(
        id=f"article:{chunk_index}",
        document_id="article",
        title="Article",
        source="article.md",
        section_heading="Billing",
        chunk_index=chunk_index,
        text=text,
    )


# Each chunk stays unchanged and gets its own vector, in the original order.
def test_pairs_each_chunk_with_its_embedding(fake_model):
    chunks = [make_chunk(f"Text {index}", chunk_index=index) for index in range(3)]

    embedded = generate_embeddings(chunks)

    assert fake_model.texts == ["Text 0", "Text 1", "Text 2"]
    assert embedded == [
        EmbeddedChunk(chunk=chunk, embedding=[float(index)] * EMBEDDING_DIMENSIONS)
        for index, chunk in enumerate(chunks)
    ]


# Embeddings are plain floats, not NumPy values, so they're ready to store later.
def test_embeddings_are_plain_floats(fake_model):
    embedded = generate_embeddings([make_chunk("Text")])

    assert type(embedded[0].embedding[0]) is float


# No chunks means nothing to embed, so the slow model load is skipped.
def test_no_chunks_does_not_load_model(model_must_not_load):
    assert generate_embeddings([]) == []


# Chunks with only whitespace have nothing to embed either.
def test_all_blank_chunks_do_not_load_model(model_must_not_load):
    assert generate_embeddings([make_chunk(""), make_chunk("  \n\t")]) == []


# Blank chunks are left out, and the others still get the right vectors.
def test_skips_blank_chunks(fake_model):
    chunks = [make_chunk("First"), make_chunk("   "), make_chunk("Second")]

    embedded = generate_embeddings(chunks)

    assert fake_model.texts == ["First", "Second"]
    assert [item.chunk.text for item in embedded] == ["First", "Second"]
    assert [item.embedding[0] for item in embedded] == [0.0, 1.0]


# A model with the wrong output size fails, and the error names the chunk.
def test_wrong_dimensions_raise_clear_error(monkeypatch):
    model = types.SimpleNamespace(
        encode=lambda texts, show_progress_bar: np.zeros((len(texts), 10))
    )
    monkeypatch.setattr(ingest, "get_embedding_model", lambda: model)

    with pytest.raises(DocumentEmbeddingError, match="chunk 4 of article.md.*got 10"):
        generate_embeddings([make_chunk("Text", chunk_index=4)])


# Every vector gets checked, not just the first.
def test_single_wrong_sized_vector_raises_clear_error(monkeypatch):
    vectors = [np.zeros(EMBEDDING_DIMENSIONS), np.zeros(EMBEDDING_DIMENSIONS - 1)]
    model = types.SimpleNamespace(encode=lambda texts, show_progress_bar: vectors)
    monkeypatch.setattr(ingest, "get_embedding_model", lambda: model)
    chunks = [make_chunk("First", chunk_index=0), make_chunk("Second", chunk_index=1)]

    with pytest.raises(DocumentEmbeddingError, match="chunk 1 of article.md"):
        generate_embeddings(chunks)


# Too few vectors would attach them to the wrong chunks, so it's an error.
def test_wrong_number_of_vectors_raises_clear_error(monkeypatch):
    model = types.SimpleNamespace(
        encode=lambda texts, show_progress_bar: np.zeros((1, EMBEDDING_DIMENSIONS))
    )
    monkeypatch.setattr(ingest, "get_embedding_model", lambda: model)

    with pytest.raises(
        DocumentEmbeddingError, match="Expected 2 embeddings.*returned 1"
    ):
        generate_embeddings([make_chunk("First"), make_chunk("Second")])


# A model that can't load gives a clear error, such as when offline with no cache.
# The library is replaced here, so this test doesn't load PyTorch.
def test_model_load_failure_raises_clear_error(monkeypatch):
    def fail(name):
        raise OSError("no internet")

    library = types.SimpleNamespace(SentenceTransformer=fail)
    monkeypatch.setitem(sys.modules, "sentence_transformers", library)
    get_embedding_model.cache_clear()

    with pytest.raises(DocumentEmbeddingError, match="Cannot load.*no internet"):
        get_embedding_model()


def make_embedded_chunks(count=2):
    return [
        EmbeddedChunk(
            chunk=make_chunk(f"Text {index}", chunk_index=index),
            embedding=[index + 0.1] * EMBEDDING_DIMENSIONS,
        )
        for index in range(count)
    ]


# The file records which model made the vectors and reads back exactly as saved.
def test_saved_file_has_metadata_and_reads_back(tmp_path):
    path = tmp_path / "knowledge_base.json"
    embedded_chunks = make_embedded_chunks()

    save_knowledge_base(embedded_chunks, path)

    assert json.loads(path.read_text(encoding="utf-8"))["metadata"] == {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunk_count": 2,
    }
    saved = KnowledgeBase.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved.chunks == embedded_chunks


# The data directory doesn't exist on a fresh checkout, so it gets created.
def test_save_creates_missing_directories(tmp_path):
    path = tmp_path / "new" / "folder" / "knowledge_base.json"

    save_knowledge_base(make_embedded_chunks(), path)

    assert path.exists()


# A new save replaces the old contents entirely, rather than adding to them.
def test_save_replaces_previous_file(tmp_path):
    path = tmp_path / "knowledge_base.json"
    save_knowledge_base(make_embedded_chunks(3), path)

    save_knowledge_base(make_embedded_chunks(1), path)

    saved = KnowledgeBase.model_validate_json(path.read_text(encoding="utf-8"))
    assert saved.metadata.chunk_count == 1
    assert len(saved.chunks) == 1


# Only the finished file remains after a successful save.
def test_save_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "knowledge_base.json"

    save_knowledge_base(make_embedded_chunks(), path)

    assert list(tmp_path.iterdir()) == [path]


# If the final swap fails, the old file survives and the temporary file is removed.
def test_failed_save_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "knowledge_base.json"
    path.write_text("previous", encoding="utf-8")

    def fail(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(type(path), "replace", fail)

    with pytest.raises(DocumentStoreError, match="disk full") as error:
        save_knowledge_base(make_embedded_chunks(), path)

    assert str(path) in str(error.value)
    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [path]


# A directory that can't be created gives the same clear error.
def test_save_to_impossible_directory_raises_clear_error(tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(DocumentStoreError, match="Cannot save"):
        save_knowledge_base(make_embedded_chunks(), blocker / "knowledge_base.json")


# The script writes the file and says where it went.
def test_main_saves_knowledge_base(tmp_path, capsys, fake_model):
    (tmp_path / "article.md").write_text("# Article\n\nSome text.\n", encoding="utf-8")
    output = tmp_path / "output.json"

    assert main(tmp_path, output) == 0

    saved = KnowledgeBase.model_validate_json(output.read_text(encoding="utf-8"))
    assert [item.chunk.id for item in saved.chunks] == ["article:0"]
    assert f"Saved 1 chunks to {output}" in capsys.readouterr().out


# A failed run exits before saving, so the last good file stays in place.
def test_main_keeps_previous_file_when_embedding_fails(tmp_path, monkeypatch):
    def fail():
        raise DocumentEmbeddingError("Cannot load the model")

    monkeypatch.setattr(ingest, "get_embedding_model", fail)
    (tmp_path / "article.md").write_text("# Article\n\nSome text.\n", encoding="utf-8")
    output = tmp_path / "output.json"
    output.write_text("previous", encoding="utf-8")

    assert main(tmp_path, output) == 1

    assert output.read_text(encoding="utf-8") == "previous"


# A storage failure is reported the same way as the script's other failures.
def test_main_reports_save_errors(tmp_path, capsys, fake_model):
    (tmp_path / "article.md").write_text("# Article\n\nSome text.\n", encoding="utf-8")
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory", encoding="utf-8")

    assert main(tmp_path, blocker / "output.json") == 1

    assert "Cannot save" in capsys.readouterr().err
