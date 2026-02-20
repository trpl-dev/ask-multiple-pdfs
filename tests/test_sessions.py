"""Tests for session persistence helpers in app.py."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import app  # noqa: E402

# ---------------------------------------------------------------------------
# _serialize_messages / _deserialize_messages
# ---------------------------------------------------------------------------


def test_serialize_deserialize_messages_roundtrip():
    original = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="What is RAG?"),
        AIMessage(content="Retrieval-Augmented Generation."),
    ]
    serialized = app._serialize_messages(original)
    assert serialized == [
        {"type": "human", "content": "Hello"},
        {"type": "ai", "content": "Hi there!"},
        {"type": "human", "content": "What is RAG?"},
        {"type": "ai", "content": "Retrieval-Augmented Generation."},
    ]
    restored = app._deserialize_messages(serialized)
    assert len(restored) == 4
    assert isinstance(restored[0], HumanMessage)
    assert isinstance(restored[1], AIMessage)
    assert restored[2].content == "What is RAG?"
    assert restored[3].content == "Retrieval-Augmented Generation."


def test_deserialize_unknown_type_becomes_ai_message():
    data = [{"type": "system", "content": "You are helpful."}]
    result = app._deserialize_messages(data)
    assert len(result) == 1
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "You are helpful."


def test_deserialize_empty_list():
    assert app._deserialize_messages([]) == []


# ---------------------------------------------------------------------------
# _serialize_sources / _deserialize_sources
# ---------------------------------------------------------------------------


def test_serialize_deserialize_sources_roundtrip():
    turn1 = [Document(page_content="chunk 1", metadata={"source": "a.pdf", "page": 1})]
    turn2 = [
        Document(page_content="chunk 2", metadata={"source": "b.pdf", "page": 3}),
        Document(page_content="chunk 3", metadata={"source": "b.pdf", "page": 4}),
    ]
    sources = [turn1, turn2]
    serialized = app._serialize_sources(sources)
    restored = app._deserialize_sources(serialized)
    assert len(restored) == 2
    assert restored[0][0].page_content == "chunk 1"
    assert restored[0][0].metadata == {"source": "a.pdf", "page": 1}
    assert len(restored[1]) == 2
    assert restored[1][1].metadata["page"] == 4


# ---------------------------------------------------------------------------
# save_session / load_session / list_sessions / delete_session
# ---------------------------------------------------------------------------


def test_save_and_load_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    chat = [HumanMessage(content="q"), AIMessage(content="a")]
    sources = [[Document(page_content="src", metadata={"source": "doc.pdf", "page": 2})]]

    app.save_session("my-session", chat, sources)
    hist, srcs = app.load_session("my-session")

    assert hist is not None
    assert len(hist) == 2
    assert isinstance(hist[0], HumanMessage)
    assert hist[0].content == "q"
    assert isinstance(hist[1], AIMessage)
    assert hist[1].content == "a"
    assert srcs[0][0].metadata["page"] == 2


def test_load_session_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    hist, srcs = app.load_session("nonexistent")
    assert hist is None
    assert srcs is None


def test_load_session_corrupt_json_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    (tmp_path / "bad.json").write_text("{{{not json", encoding="utf-8")
    hist, srcs = app.load_session("bad")
    assert hist is None
    assert srcs is None


def test_list_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    app.save_session("alpha", [HumanMessage(content="x"), AIMessage(content="y")], [])
    app.save_session("beta", [HumanMessage(content="x"), AIMessage(content="y")], [])
    sessions = app.list_sessions()
    assert sessions == ["alpha", "beta"]


def test_list_sessions_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    assert app.list_sessions() == []


def test_delete_session(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    app.save_session("to-delete", [HumanMessage(content="x"), AIMessage(content="y")], [])
    assert "to-delete" in app.list_sessions()
    app.delete_session("to-delete")
    assert "to-delete" not in app.list_sessions()


def test_delete_nonexistent_session_is_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    # Should not raise
    app.delete_session("ghost")


# ---------------------------------------------------------------------------
# _safe_name
# ---------------------------------------------------------------------------


def test_safe_name_accepts_valid_names():
    assert app._safe_name("my-session") == "my-session"
    assert app._safe_name("  project-alpha  ") == "project-alpha"
    assert app._safe_name("Doc 2024") == "Doc 2024"
    assert app._safe_name("a") == "a"


def test_safe_name_rejects_path_traversal():
    with pytest.raises(ValueError):
        app._safe_name("../etc/passwd")


def test_safe_name_rejects_empty_string():
    with pytest.raises(ValueError):
        app._safe_name("")
    with pytest.raises(ValueError):
        app._safe_name("   ")


def test_safe_name_rejects_leading_dot():
    with pytest.raises(ValueError):
        app._safe_name(".hidden")


def test_save_session_rejects_unsafe_name(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SESSIONS_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        app.save_session(
            "../evil",
            [HumanMessage(content="x"), AIMessage(content="y")],
            [],
        )


# ---------------------------------------------------------------------------
# _truncate_history
# ---------------------------------------------------------------------------


def test_truncate_history_no_op_when_short():
    history = [HumanMessage(content="q"), AIMessage(content="a")] * 5  # 10 messages, 5 turns
    result = app._truncate_history(history, max_turns=10)
    assert result is history  # unchanged — same object returned


def test_truncate_history_trims_to_max_turns():
    history = []
    for i in range(15):
        history.append(HumanMessage(content=f"q{i}"))
        history.append(AIMessage(content=f"a{i}"))
    # 30 messages, 15 turns → truncate to 10 turns = 20 messages
    result = app._truncate_history(history, max_turns=10)
    assert len(result) == 20
    # Should be the *last* 20 messages
    assert result[0].content == "q5"
    assert result[-1].content == "a14"
