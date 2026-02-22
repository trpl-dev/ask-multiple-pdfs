"""Tests for save_index_metadata(), load_index_metadata(), HMAC helpers, and path guard."""

import json
from datetime import datetime

import pytest

import app  # noqa: E402 (app.py imports streamlit at module level)

# ---------------------------------------------------------------------------
# Existing metadata round-trip tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _get_hmac_secret
# ---------------------------------------------------------------------------


def test_get_hmac_secret_returns_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("FAISS_HMAC_SECRET", raising=False)
    assert app._get_hmac_secret() is None


def test_get_hmac_secret_returns_bytes_when_env_set(monkeypatch):
    monkeypatch.setenv("FAISS_HMAC_SECRET", "mysecret")
    result = app._get_hmac_secret()
    assert result == b"mysecret"


# ---------------------------------------------------------------------------
# _compute_index_hmac
# ---------------------------------------------------------------------------


def _write_fake_artifacts(path):
    """Write stub FAISS artifact files into *path* for testing."""
    (path / "index.faiss").write_bytes(b"faiss-data")
    (path / "index.pkl").write_bytes(b"pickle-data")


def test_compute_hmac_returns_none_without_secret(monkeypatch, tmp_path):
    monkeypatch.delenv("FAISS_HMAC_SECRET", raising=False)
    _write_fake_artifacts(tmp_path)
    assert app._compute_index_hmac(str(tmp_path)) is None


def test_compute_hmac_returns_hex_string_with_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_HMAC_SECRET", "testsecret")
    _write_fake_artifacts(tmp_path)
    result = app._compute_index_hmac(str(tmp_path))
    assert result is not None
    # HMAC-SHA256 hex digest is always 64 hex characters
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_compute_hmac_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_HMAC_SECRET", "testsecret")
    _write_fake_artifacts(tmp_path)
    assert app._compute_index_hmac(str(tmp_path)) == app._compute_index_hmac(str(tmp_path))


def test_compute_hmac_differs_after_artifact_change(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_HMAC_SECRET", "testsecret")
    _write_fake_artifacts(tmp_path)
    digest_before = app._compute_index_hmac(str(tmp_path))

    # Tamper with the pickle artifact
    (tmp_path / "index.pkl").write_bytes(b"tampered-data")
    digest_after = app._compute_index_hmac(str(tmp_path))

    assert digest_before != digest_after


def test_compute_hmac_returns_none_if_artifact_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_HMAC_SECRET", "testsecret")
    # Only write one of the two required artifacts
    (tmp_path / "index.faiss").write_bytes(b"faiss-data")
    # index.pkl is absent
    assert app._compute_index_hmac(str(tmp_path)) is None


def test_compute_hmac_changes_with_different_secrets(monkeypatch, tmp_path):
    _write_fake_artifacts(tmp_path)

    monkeypatch.setenv("FAISS_HMAC_SECRET", "secret-A")
    digest_a = app._compute_index_hmac(str(tmp_path))

    monkeypatch.setenv("FAISS_HMAC_SECRET", "secret-B")
    digest_b = app._compute_index_hmac(str(tmp_path))

    assert digest_a != digest_b


# ---------------------------------------------------------------------------
# save_index_metadata — HMAC field inclusion
# ---------------------------------------------------------------------------


def test_save_metadata_includes_hmac_when_secret_set(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_HMAC_SECRET", "testsecret")
    _write_fake_artifacts(tmp_path)

    app.save_index_metadata(["a.pdf"], chunk_count=5, index_path=str(tmp_path))
    result = app.load_index_metadata(str(tmp_path))

    assert result is not None
    assert "hmac" in result
    assert len(result["hmac"]) == 64


def test_save_metadata_omits_hmac_when_secret_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("FAISS_HMAC_SECRET", raising=False)

    app.save_index_metadata(["b.pdf"], chunk_count=3, index_path=str(tmp_path))
    result = app.load_index_metadata(str(tmp_path))

    assert result is not None
    assert "hmac" not in result


def test_save_metadata_hmac_matches_recomputed_value(monkeypatch, tmp_path):
    monkeypatch.setenv("FAISS_HMAC_SECRET", "testsecret")
    _write_fake_artifacts(tmp_path)

    app.save_index_metadata(["c.pdf"], chunk_count=7, index_path=str(tmp_path))
    stored = app.load_index_metadata(str(tmp_path))["hmac"]
    recomputed = app._compute_index_hmac(str(tmp_path))

    assert stored == recomputed


# ---------------------------------------------------------------------------
# _assert_within_base_dir
# ---------------------------------------------------------------------------


def test_path_guard_allows_valid_subdir(tmp_path):
    subdir = tmp_path / "slots" / "default"
    subdir.mkdir(parents=True)
    # Should not raise
    app._assert_within_base_dir(str(subdir), str(tmp_path / "slots"))


def test_path_guard_rejects_traversal(tmp_path):
    base = tmp_path / "slots"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="resolves outside"):
        app._assert_within_base_dir(str(outside), str(base))


def test_path_guard_rejects_symlink_traversal(tmp_path):
    base = tmp_path / "slots"
    base.mkdir()
    slot = base / "myslot"
    slot.mkdir()

    target = tmp_path / "secret"
    target.mkdir()

    link = slot / "escape"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="resolves outside"):
        app._assert_within_base_dir(str(link), str(base))


def test_path_guard_allows_base_dir_itself(tmp_path):
    base = tmp_path / "slots"
    base.mkdir()
    # Passing the base dir itself should not raise
    app._assert_within_base_dir(str(base), str(base))
