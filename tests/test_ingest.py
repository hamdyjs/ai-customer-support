import pytest

from scripts.ingest import (
    Document,
    DocumentLoadError,
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


# The script lists what it loaded and exits successfully.
def test_main_prints_summary(tmp_path, capsys):
    (tmp_path / "article.md").write_text("# Article\n", encoding="utf-8")

    assert main(tmp_path) == 0

    output = capsys.readouterr().out
    assert "article: Article" in output
    assert "Documents loaded" in output


# On failure the script prints the error to stderr and exits with status 1.
def test_main_reports_errors(tmp_path, capsys):
    assert main(tmp_path / "missing") == 1

    assert "Cannot read knowledge base directory" in capsys.readouterr().err
