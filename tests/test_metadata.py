"""Tests for save_index_metadata() and load_index_metadata() in app.py."""

import json
from datetime import datetime

import app


def test_save_and_load_roundtrip(tmp_path):
    app.save_index_metadata(["doc1.pdf", "doc2.pdf"], chunk_count=42, index_path=str(tmp_path))
    result = app.load_index_metadata(str(tmp_path))

    assert result is not None
    assert result["files"] == ["doc1.pdf", "doc2.pdf"]
    assert result["chunks"] == 42
    assert "timestamp" in result


def test_load_returns_none_when_file_missing(tmp_path):
    # Point to an empty directory — no metadata.json present
    assert app.load_index_metadata(str(tmp_path)) is None


def test_load_returns_none_on_corrupt_json(tmp_path):
    (tmp_path / "metadata.json").write_text("not valid json {{{", encoding="utf-8")
    assert app.load_index_metadata(str(tmp_path)) is None


def test_save_overwrites_existing(tmp_path):
    app.save_index_metadata(["old.pdf"], chunk_count=10, index_path=str(tmp_path))
    app.save_index_metadata(["new.pdf"], chunk_count=99, index_path=str(tmp_path))

    with open(tmp_path / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)

    assert data["files"] == ["new.pdf"]
    assert data["chunks"] == 99


def test_timestamp_is_iso_format(tmp_path):
    app.save_index_metadata(["x.pdf"], chunk_count=1, index_path=str(tmp_path))
    result = app.load_index_metadata(str(tmp_path))

    # Should not raise — confirms it is a valid ISO 8601 timestamp
    ts = datetime.fromisoformat(result["timestamp"])
    assert ts.tzinfo is not None  # UTC timezone present
