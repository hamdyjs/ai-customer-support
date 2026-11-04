import itertools

import pytest

from scripts.ingest import (
    MAX_CHUNK_TOKENS,
    OVERLAP_TOKENS,
    SAMPLE_CHUNKS,
    Document,
    DocumentChunkError,
    DocumentLoadError,
    chunk_documents,
    count_tokens,
    load_documents,
    main,
)


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
def test_main_prints_chunks(tmp_path, capsys):
    (tmp_path / "article.md").write_text("# Article\n\nSome text.\n", encoding="utf-8")

    assert main(tmp_path) == 0

    output = capsys.readouterr().out
    assert "Article\n\nSome text." in output
    assert "article:" not in output
    assert "Documents loaded" not in output


# The sample is capped and spread across all the chunks, not just the first ones.
def test_main_prints_an_even_sample(tmp_path, capsys):
    for number in range(100):
        (tmp_path / f"{number:03}.md").write_text(
            f"# Article {number:03}\n\nText.\n", encoding="utf-8"
        )

    assert main(tmp_path) == 0

    output = capsys.readouterr().out
    assert output.count("--- ") == SAMPLE_CHUNKS
    assert "Article 000" in output
    assert "Article 090" in output


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
            "document_id": "article",
            "source": "article.md",
            "section_heading": None,
            "chunk_index": 0,
            "text": "Article\n\nIntro text.",
        },
        {
            "document_id": "article",
            "source": "article.md",
            "section_heading": "Billing",
            "chunk_index": 1,
            "text": "Article\nBilling\n\nBilling text.",
        },
        {
            "document_id": "article",
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
