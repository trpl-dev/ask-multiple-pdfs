"""Tests for HybridRetriever, FilteredRetriever, and RerankingRetriever in app.py."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app import (
    FilteredRetriever,
    HybridRetriever,
    RerankingRetriever,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_MANAGER = MagicMock()


def _docs(*contents: str, source: str = "a.pdf") -> list[Document]:
    return [Document(page_content=c, metadata={"source": source}) for c in contents]


def _mock_base(docs: list[Document]) -> MagicMock:
    base = MagicMock()
    base.invoke.return_value = docs
    return base


# ---------------------------------------------------------------------------
# FilteredRetriever
# ---------------------------------------------------------------------------


class TestFilteredRetriever:
    def test_keeps_only_allowed_sources(self):
        docs = [
            Document(page_content="a", metadata={"source": "a.pdf"}),
            Document(page_content="b", metadata={"source": "b.pdf"}),
            Document(page_content="c", metadata={"source": "a.pdf"}),
        ]
        retriever = FilteredRetriever(
            base_retriever=_mock_base(docs), allowed_sources=["a.pdf"], top_k=10
        )
        result = retriever._get_relevant_documents("q", run_manager=_RUN_MANAGER)
        assert all(d.metadata["source"] == "a.pdf" for d in result)
        assert len(result) == 2

    def test_respects_top_k(self):
        docs = [Document(page_content=f"d{i}", metadata={"source": "a.pdf"}) for i in range(10)]
        retriever = FilteredRetriever(
            base_retriever=_mock_base(docs), allowed_sources=["a.pdf"], top_k=3
        )
        result = retriever._get_relevant_documents("q", run_manager=_RUN_MANAGER)
        assert len(result) == 3

    def test_returns_empty_when_no_source_matches(self):
        docs = _docs("x", "y", source="b.pdf")
        retriever = FilteredRetriever(
            base_retriever=_mock_base(docs), allowed_sources=["a.pdf"], top_k=5
        )
        result = retriever._get_relevant_documents("q", run_manager=_RUN_MANAGER)
        assert result == []

    def test_empty_candidate_list(self):
        retriever = FilteredRetriever(
            base_retriever=_mock_base([]), allowed_sources=["a.pdf"], top_k=5
        )
        result = retriever._get_relevant_documents("q", run_manager=_RUN_MANAGER)
        assert result == []

    def test_multiple_allowed_sources(self):
        docs = [
            Document(page_content="a", metadata={"source": "a.pdf"}),
            Document(page_content="b", metadata={"source": "b.pdf"}),
            Document(page_content="c", metadata={"source": "c.pdf"}),
        ]
        retriever = FilteredRetriever(
            base_retriever=_mock_base(docs),
            allowed_sources=["a.pdf", "b.pdf"],
            top_k=10,
        )
        result = retriever._get_relevant_documents("q", run_manager=_RUN_MANAGER)
        sources = {d.metadata["source"] for d in result}
        assert sources == {"a.pdf", "b.pdf"}
        assert len(result) == 2


# ---------------------------------------------------------------------------
# RerankingRetriever
# ---------------------------------------------------------------------------


class TestRerankingRetriever:
    def test_reorders_by_cross_encoder_score(self):
        docs = [
            Document(page_content="worse", metadata={}),
            Document(page_content="better", metadata={}),
        ]
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.1, 0.9]  # "better" gets higher score

        with patch("app._load_cross_encoder", return_value=mock_encoder):
            retriever = RerankingRetriever(base_retriever=_mock_base(docs), top_k=2, fetch_k=10)
            result = retriever._get_relevant_documents("test", run_manager=_RUN_MANAGER)

        assert result[0].page_content == "better"
        assert result[1].page_content == "worse"

    def test_respects_top_k(self):
        docs = [Document(page_content=f"doc{i}", metadata={}) for i in range(5)]
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.5, 0.4, 0.3, 0.2, 0.1]

        with patch("app._load_cross_encoder", return_value=mock_encoder):
            retriever = RerankingRetriever(base_retriever=_mock_base(docs), top_k=2, fetch_k=10)
            result = retriever._get_relevant_documents("test", run_manager=_RUN_MANAGER)

        assert len(result) == 2

    def test_falls_back_to_original_order_on_encoder_error(self):
        docs = [Document(page_content=f"doc{i}", metadata={}) for i in range(3)]
        with patch("app._load_cross_encoder", side_effect=RuntimeError("model load failed")):
            retriever = RerankingRetriever(base_retriever=_mock_base(docs), top_k=2, fetch_k=10)
            result = retriever._get_relevant_documents("query", run_manager=_RUN_MANAGER)

        # Falls back: first 2 in original order
        assert len(result) == 2
        assert result[0].page_content == "doc0"
        assert result[1].page_content == "doc1"

    def test_single_candidate_skips_reranking(self):
        docs = [Document(page_content="only one", metadata={})]
        mock_encoder = MagicMock()

        with patch("app._load_cross_encoder", return_value=mock_encoder):
            retriever = RerankingRetriever(base_retriever=_mock_base(docs), top_k=5, fetch_k=10)
            result = retriever._get_relevant_documents("query", run_manager=_RUN_MANAGER)

        mock_encoder.predict.assert_not_called()
        assert result[0].page_content == "only one"

    def test_empty_candidates(self):
        with patch("app._load_cross_encoder", return_value=MagicMock()):
            retriever = RerankingRetriever(base_retriever=_mock_base([]), top_k=5, fetch_k=10)
            result = retriever._get_relevant_documents("query", run_manager=_RUN_MANAGER)
        assert result == []


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    def _make_vectorstore(self, docs: list[Document]) -> MagicMock:
        vs = MagicMock()
        vs.similarity_search.return_value = docs
        vs.max_marginal_relevance_search.return_value = docs
        return vs

    def test_falls_back_to_vector_when_bm25_unavailable(self):
        vector_docs = _docs("vec1", "vec2")
        vs = self._make_vectorstore(vector_docs)
        corpus = ["vec1", "vec2", "other"]

        with patch("app._build_bm25", return_value=None):
            retriever = HybridRetriever(
                vectorstore=vs,
                corpus=corpus,
                corpus_metadatas=[{}, {}, {}],
                top_k=2,
            )
            result = retriever._get_relevant_documents("query", run_manager=_RUN_MANAGER)

        # Only vector results returned
        assert len(result) <= 2
        assert all(d.page_content in ("vec1", "vec2") for d in result)

    def test_rrf_promotes_doc_appearing_in_both_branches(self):
        shared_doc = Document(page_content="shared", metadata={})
        vec_only = Document(page_content="vec_only", metadata={})
        vs = self._make_vectorstore([shared_doc, vec_only])

        # BM25 index that scores "shared" highest
        bm25_index = MagicMock()
        bm25_index.get_scores.return_value = [0.9, 0.0, 0.0]  # first corpus entry wins

        corpus = ["shared", "bm25_only", "other"]
        corpus_meta = [{}, {}, {}]

        with patch("app._build_bm25", return_value=bm25_index):
            retriever = HybridRetriever(
                vectorstore=vs,
                corpus=corpus,
                corpus_metadatas=corpus_meta,
                top_k=3,
            )
            result = retriever._get_relevant_documents("shared", run_manager=_RUN_MANAGER)

        # "shared" appears in both branches → should rank first via RRF
        contents = [d.page_content for d in result]
        assert "shared" in contents
        assert contents.index("shared") == 0

    def test_uses_mmr_when_retrieval_mode_set(self):
        docs = _docs("doc1")
        vs = self._make_vectorstore(docs)

        with patch("app._build_bm25", return_value=None):
            retriever = HybridRetriever(
                vectorstore=vs,
                corpus=["doc1"],
                corpus_metadatas=[{}],
                top_k=1,
                retrieval_mode="MMR",
            )
            retriever._get_relevant_documents("query", run_manager=_RUN_MANAGER)

        vs.max_marginal_relevance_search.assert_called_once()
        vs.similarity_search.assert_not_called()

    def test_uses_similarity_by_default(self):
        docs = _docs("doc1")
        vs = self._make_vectorstore(docs)

        with patch("app._build_bm25", return_value=None):
            retriever = HybridRetriever(
                vectorstore=vs,
                corpus=["doc1"],
                corpus_metadatas=[{}],
                top_k=1,
            )
            retriever._get_relevant_documents("query", run_manager=_RUN_MANAGER)

        vs.similarity_search.assert_called_once()
        vs.max_marginal_relevance_search.assert_not_called()

    def test_top_k_limits_results(self):
        docs = [Document(page_content=f"doc{i}", metadata={}) for i in range(10)]
        vs = self._make_vectorstore(docs)

        with patch("app._build_bm25", return_value=None):
            retriever = HybridRetriever(
                vectorstore=vs,
                corpus=[f"doc{i}" for i in range(10)],
                corpus_metadatas=[{}] * 10,
                top_k=3,
            )
            result = retriever._get_relevant_documents("query", run_manager=_RUN_MANAGER)

        assert len(result) <= 3
