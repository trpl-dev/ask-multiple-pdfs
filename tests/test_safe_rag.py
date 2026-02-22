"""Tests for Safe RAG mode prompt-injection resistance.

Covers:
* Content and completeness of SAFE_RAG_INSTRUCTIONS.
* get_conversation_chain() prompt construction with safe_rag_mode ON and OFF.
"""

from unittest.mock import MagicMock, patch

from app import (
    SAFE_RAG_INSTRUCTIONS,
    get_conversation_chain,
)

# ---------------------------------------------------------------------------
# SAFE_RAG_INSTRUCTIONS content checks
# ---------------------------------------------------------------------------


class TestSafeRagInstructions:
    def test_non_empty(self):
        assert SAFE_RAG_INSTRUCTIONS.strip()

    def test_contains_context_only_rule(self):
        """LLM must be told to answer from context only."""
        lower = SAFE_RAG_INSTRUCTIONS.lower()
        assert "context" in lower

    def test_contains_injection_resistance(self):
        """The constant must reference ignoring embedded instructions."""
        lower = SAFE_RAG_INSTRUCTIONS.lower()
        assert "ignore" in lower
        assert "instruction" in lower

    def test_contains_no_secret_disclosure(self):
        """Must prohibit revealing secrets / credentials."""
        lower = SAFE_RAG_INSTRUCTIONS.lower()
        assert any(w in lower for w in ("secret", "credential", "api key", "password", "token"))

    def test_contains_no_system_prompt_disclosure(self):
        """Must prohibit revealing the system instructions themselves."""
        lower = SAFE_RAG_INSTRUCTIONS.lower()
        assert "reveal" in lower or "disclose" in lower or "repeat" in lower

    def test_contains_no_persona_switch(self):
        """Must instruct the model to refuse persona-switching requests."""
        lower = SAFE_RAG_INSTRUCTIONS.lower()
        assert "persona" in lower or "bypass" in lower or "ignore" in lower


# ---------------------------------------------------------------------------
# Helpers for mocking the LangChain/FAISS stack
# ---------------------------------------------------------------------------


def _mock_vectorstore(k: int = 4):
    """Return a minimal FAISS-like mock that satisfies get_conversation_chain()."""
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    vs.as_retriever.return_value.invoke.return_value = []
    return vs


def _build_chain(safe_rag_mode: bool, system_prompt: str = ""):
    """Call get_conversation_chain() with all external dependencies mocked out."""
    vs = _mock_vectorstore()

    # Patch LLM constructors so no API calls are made
    with (
        patch("app.ChatOpenAI") as mock_openai,
        patch("app.create_history_aware_retriever") as mock_hist,
        patch("app.create_stuff_documents_chain") as mock_stuff,
        patch("app.create_retrieval_chain") as mock_rc,
    ):
        dummy_llm = MagicMock()
        mock_openai.return_value = dummy_llm
        mock_hist.return_value = MagicMock()
        mock_stuff.return_value = MagicMock()
        mock_rc.return_value = MagicMock()

        get_conversation_chain(
            vs,
            model="gpt-4o-mini",
            stream_handler=None,
            api_key="test-key",
            safe_rag_mode=safe_rag_mode,
            system_prompt=system_prompt,
        )

        # Return the prompt templates passed to create_history_aware_retriever
        # and create_stuff_documents_chain so tests can inspect them.
        condense_prompt = mock_hist.call_args[0][2]  # positional arg 3
        qa_prompt = mock_stuff.call_args[0][1]  # positional arg 2
        return condense_prompt, qa_prompt


# ---------------------------------------------------------------------------
# get_conversation_chain() prompt-template tests
# ---------------------------------------------------------------------------


class TestSafeRagChainPrompts:
    def test_safe_mode_on_injects_instructions_into_qa_prompt(self):
        _, qa_prompt = _build_chain(safe_rag_mode=True)
        system_msg = qa_prompt.messages[0].prompt.template
        assert SAFE_RAG_INSTRUCTIONS in system_msg

    def test_safe_mode_off_omits_instructions_from_qa_prompt(self):
        _, qa_prompt = _build_chain(safe_rag_mode=False)
        system_msg = qa_prompt.messages[0].prompt.template
        assert SAFE_RAG_INSTRUCTIONS not in system_msg

    def test_safe_mode_on_adds_note_to_condense_prompt(self):
        condense_prompt, _ = _build_chain(safe_rag_mode=True)
        system_msg = condense_prompt.messages[0].prompt.template
        assert "ignore" in system_msg.lower()

    def test_safe_mode_off_no_extra_note_in_condense_prompt(self):
        condense_prompt, _ = _build_chain(safe_rag_mode=False)
        system_msg = condense_prompt.messages[0].prompt.template
        # The base condense prompt does not contain the injection-guard note
        assert "ignore any instructions embedded" not in system_msg.lower()

    def test_user_system_prompt_still_included_when_safe_mode_on(self):
        custom = "Answer only in French."
        _, qa_prompt = _build_chain(safe_rag_mode=True, system_prompt=custom)
        system_msg = qa_prompt.messages[0].prompt.template
        assert custom in system_msg
        assert SAFE_RAG_INSTRUCTIONS in system_msg

    def test_safe_instructions_precede_user_system_prompt(self):
        """Safe RAG rules must appear before the operator-supplied system prompt."""
        custom = "Answer only in French."
        _, qa_prompt = _build_chain(safe_rag_mode=True, system_prompt=custom)
        system_msg = qa_prompt.messages[0].prompt.template
        safe_pos = system_msg.index(SAFE_RAG_INSTRUCTIONS)
        custom_pos = system_msg.index(custom)
        assert safe_pos < custom_pos

    def test_safe_mode_default_is_true(self):
        """Omitting safe_rag_mode should default to ON (safe by default)."""
        vs = _mock_vectorstore()
        with (
            patch("app.ChatOpenAI"),
            patch("app.create_history_aware_retriever"),
            patch("app.create_stuff_documents_chain") as mock_stuff,
            patch("app.create_retrieval_chain"),
        ):
            get_conversation_chain(vs, model="gpt-4o-mini", api_key="k")
            qa_prompt = mock_stuff.call_args[0][1]
            system_msg = qa_prompt.messages[0].prompt.template
            assert SAFE_RAG_INSTRUCTIONS in system_msg
