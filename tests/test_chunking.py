"""Tests for get_text_chunks() in app.py."""

import pytest

from app import get_text_chunks


def test_returns_parallel_lists():
    texts = [("Hello world " * 50, "doc.pdf")]
    chunks, metas = get_text_chunks(texts)
    assert len(chunks) == len(metas)
    assert len(chunks) > 0


def test_source_metadata_key():
    texts = [("Sample text " * 50, "report.pdf")]
    chunks, metas = get_text_chunks(texts)
    assert all(m["source"] == "report.pdf" for m in metas)


def test_multiple_files_tracked_separately():
    texts = [
        ("Content from alpha " * 40, "alpha.pdf"),
        ("Content from beta " * 40, "beta.pdf"),
    ]
    chunks, metas = get_text_chunks(texts)
    sources = {m["source"] for m in metas}
    assert "alpha.pdf" in sources
    assert "beta.pdf" in sources


def test_chunk_size_respected():
    # Build text with explicit newlines so CharacterTextSplitter (separator="\n")
    # can actually split it; each line is ~20 chars.
    line = "word " * 4  # ~20 chars
    long_text = (line + "\n") * 200
    texts = [(long_text, "big.pdf")]
    chunk_size = 100
    chunk_overlap = 10
    chunks, _ = get_text_chunks(texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # All chunks must fit within the configured size boundary
    assert all(len(c) <= chunk_size + chunk_overlap for c in chunks)


def test_custom_chunk_size_produces_more_chunks():
    text = "X " * 500
    texts = [(text, "test.pdf")]
    chunks_small, _ = get_text_chunks(texts, chunk_size=100, chunk_overlap=0)
    chunks_large, _ = get_text_chunks(texts, chunk_size=500, chunk_overlap=0)
    assert len(chunks_small) >= len(chunks_large)


def test_empty_input_returns_empty_lists():
    chunks, metas = get_text_chunks([])
    assert chunks == []
    assert metas == []


def test_overlap_greater_than_chunk_size_raises():
    texts = [("A " * 300, "overlap_test.pdf")]
    with pytest.raises(ValueError):
        # LangChain raises ValueError when chunk_overlap strictly exceeds chunk_size
        get_text_chunks(texts, chunk_size=50, chunk_overlap=51)
