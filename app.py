import concurrent.futures
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import ConfigDict
from pypdf import PdfReader
from sentence_transformers import CrossEncoder

FAISS_INDEXES_DIR = "faiss_indexes"
DEFAULT_SLOT = "default"
SESSIONS_DIR = "sessions"
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
CHUNK_STRATEGY_CHAR = "Character (fast)"
CHUNK_STRATEGY_SEMANTIC = "Semantic (accurate)"
CHUNK_STRATEGIES = [CHUNK_STRATEGY_CHAR, CHUNK_STRATEGY_SEMANTIC]
RETRIEVAL_MODES = ["Similarity", "MMR"]
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RETRIEVAL_K = 4
MAX_HISTORY_TURNS = 20  # keep at most this many human/AI turn pairs in context
MAX_QUESTION_LENGTH = 5_000  # character cap for a single user question

# Cost tracker: approximate USD per 1 000 tokens (input / output) for OpenAI models.
# Used only for display; actual billing may differ.
OPENAI_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-3.5-turbo": (0.00050, 0.00150),
    "gpt-4o": (0.00250, 0.01000),
}

PROVIDER_OPENAI = "OpenAI"
PROVIDER_OLLAMA = "Ollama (local)"
PROVIDER_CLAUDE = "Claude (Anthropic)"
PROVIDERS = [PROVIDER_OPENAI, PROVIDER_CLAUDE, PROVIDER_OLLAMA]
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2"
OLLAMA_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
AVAILABLE_CLAUDE_MODELS = ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"]
DEFAULT_CLAUDE_MODEL = AVAILABLE_CLAUDE_MODELS[0]

# Approximate USD per 1 000 tokens (input / output) for Claude models.
# Used only for display; actual billing may differ.
CLAUDE_COST_PER_1K: dict[str, tuple[float, float]] = {
    "claude-opus-4-5": (0.01500, 0.07500),
    "claude-sonnet-4-5": (0.00300, 0.01500),
    "claude-haiku-4-5": (0.00080, 0.00400),
}

# ---------------------------------------------------------------------------
# Structured logger — writes WARNING+ to stderr; caller configures handlers
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Only safe characters for user-supplied slot and session names.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][\w\-. ]*$")


def _safe_name(name: str, label: str = "Name") -> str:
    """Validate a user-supplied slot or session name against path traversal.

    Only allows names that start with an alphanumeric character and contain
    only letters, digits, hyphens, underscores, dots, or spaces.
    Raises ValueError if the name is empty or contains unsafe characters.
    """
    name = name.strip()
    if not name:
        raise ValueError(f"{label} must not be empty.")
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"{label} '{name}' contains invalid characters. "
            "Use only letters, digits, hyphens, underscores, spaces, or dots."
        )
    return name


def slot_path(slot: str) -> str:
    """Return the filesystem path for the given index slot directory."""
    return os.path.join(FAISS_INDEXES_DIR, slot)


def list_index_slots() -> list[str]:
    """Return sorted list of existing index slot names."""
    if not os.path.exists(FAISS_INDEXES_DIR):
        return []
    return sorted(d for d in os.listdir(FAISS_INDEXES_DIR) if os.path.isdir(slot_path(d)))


def save_index_metadata(filenames: list[str], chunk_count: int, index_path: str) -> None:
    """Persist index provenance to metadata.json inside the given index directory."""
    meta = {
        "files": filenames,
        "chunks": chunk_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.join(index_path, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_index_metadata(index_path: str) -> dict | None:
    """Return the saved index metadata dict for the given path, or None."""
    meta_path = os.path.join(index_path, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("Failed to load index metadata from '%s': %s", meta_path, e)
        return None


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def _serialize_messages(chat_history: list[BaseMessage]) -> list[dict]:
    return [
        {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
        for msg in chat_history
    ]


def _deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    result = []
    for d in data:
        if d.get("type") == "human":
            result.append(HumanMessage(content=d["content"]))
        else:
            result.append(AIMessage(content=d["content"]))
    return result


def _serialize_sources(sources: list[list[Document]]) -> list[list[dict]]:
    return [
        [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in turn_sources]
        for turn_sources in sources
    ]


def _deserialize_sources(raw: list[list[dict]]) -> list[list[Document]]:
    return [
        [Document(page_content=s["page_content"], metadata=s.get("metadata", {})) for s in turn]
        for turn in raw
    ]


def list_sessions() -> list[str]:
    """Return sorted list of saved session names (without .json extension)."""
    if not os.path.exists(SESSIONS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SESSIONS_DIR) if f.endswith(".json"))


def save_session(name: str, chat_history: list[BaseMessage], sources: list[list[Document]]) -> None:
    """Persist the current conversation to sessions/<name>.json."""
    name = _safe_name(name, "Session name")
    os.makedirs(SESSIONS_DIR, mode=0o700, exist_ok=True)
    path = os.path.join(SESSIONS_DIR, f"{name}.json")
    data = {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": _serialize_messages(chat_history),
        "sources": _serialize_sources(sources),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_session(
    name: str,
) -> tuple[list[BaseMessage], list[list[Document]]] | tuple[None, None]:
    """Load a saved session; returns (chat_history, sources) or (None, None)."""
    try:
        name = _safe_name(name, "Session name")
    except ValueError:
        return None, None
    path = os.path.join(SESSIONS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return _deserialize_messages(data.get("messages", [])), _deserialize_sources(
            data.get("sources", [])
        )
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
        logger.warning("Failed to load session '%s': %s", name, e)
        return None, None


def delete_session(name: str) -> None:
    """Delete a saved session file."""
    try:
        name = _safe_name(name, "Session name")
    except ValueError:
        return
    path = os.path.join(SESSIONS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)


def _clear_conversation() -> None:
    """Reset conversation state: chat history, sources, feedback, and suggested questions."""
    st.session_state.chat_history = []
    st.session_state.sources = []
    st.session_state.feedback = []
    st.session_state.suggested_questions = []


def get_api_key() -> str | None:
    """Return the active OpenAI API key.

    Priority: key entered in the sidebar UI > OPENAI_API_KEY env var.
    Returns None if neither is set so callers can show a user-friendly warning.
    """
    ui_key = st.session_state.get("api_key_input", "").strip()
    if ui_key:
        return ui_key
    return os.environ.get("OPENAI_API_KEY") or None


def get_claude_api_key() -> str | None:
    """Return the active Anthropic (Claude) API key.

    Priority: key entered in the sidebar UI > ANTHROPIC_API_KEY env var.
    Returns None if neither is set so callers can show a user-friendly warning.
    """
    ui_key = st.session_state.get("claude_api_key_input", "").strip()
    if ui_key:
        return ui_key
    return os.environ.get("ANTHROPIC_API_KEY") or None


# ---------------------------------------------------------------------------
# Streaming callback — writes LLM tokens into a Streamlit placeholder
# ---------------------------------------------------------------------------


class StreamHandler(BaseCallbackHandler):
    """Streams LLM tokens into a st.empty() placeholder as they arrive."""

    def __init__(self, container: Any) -> None:
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.text += token
        # Show a blinking-cursor effect while streaming
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        # Remove cursor when the LLM finishes
        self.container.markdown(self.text)


# ---------------------------------------------------------------------------
# RAG pipeline helpers
# ---------------------------------------------------------------------------


def _extract_single_pdf(pdf: Any) -> tuple[list[tuple[str, str, int]], list[str]]:
    """Extract text from one PDF without touching Streamlit — safe to call from threads.

    Returns a tuple of (pages, warnings) where pages is a list of
    (page_text, filename, page_number) tuples and warnings is a list of
    human-readable warning strings to display in the UI.
    """
    pages: list[tuple[str, str, int]] = []
    warnings: list[str] = []
    try:
        pdf_reader = PdfReader(pdf)
        total_pages = len(pdf_reader.pages)
        extracted = 0
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append((page_text, pdf.name, page_num))
                extracted += 1
        if total_pages > 0 and extracted == 0:
            warnings.append(
                f"'{pdf.name}' appears to be a scanned PDF (image-only) — no text could be "
                "extracted. Use an OCR tool (e.g. Adobe Acrobat, Tesseract) to convert it first."
            )
    except Exception as e:
        logger.warning("Could not read '%s': %s", pdf.name, e)
        warnings.append(
            f"Could not read '{pdf.name}'. "
            "The file may be corrupt, password-protected, or not a valid PDF."
        )
    return pages, warnings


def get_pdf_text(pdf_docs: list[Any]) -> list[tuple[str, str, int]]:
    """Extract text from each PDF; returns list of (page_text, filename, page_number) tuples.

    Each tuple represents one page so that chunk metadata can carry a page number.
    Pages with no extractable text are silently skipped. Files that fail to open
    or are scanned PDFs emit st.warning() and are excluded entirely.
    """
    results = []
    for pdf in pdf_docs:
        pages, warnings = _extract_single_pdf(pdf)
        results.extend(pages)
        for w in warnings:
            st.warning(w)
    return results


def get_text_chunks(
    texts_with_meta: list[tuple[str, str, int]],
    strategy: str = CHUNK_STRATEGY_CHAR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    semantic_threshold: int = 95,
    api_key: str | None = None,
    provider: str = PROVIDER_OPENAI,
    ollama_embedding_model: str = OLLAMA_DEFAULT_EMBEDDING_MODEL,
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL,
) -> tuple[list[str], list[dict]]:
    """Split each page's text into chunks; returns (chunks, metadatas).

    Args:
        texts_with_meta: list of (text, filename, page_number) tuples from get_pdf_text()
        strategy: CHUNK_STRATEGY_CHAR or CHUNK_STRATEGY_SEMANTIC
        chunk_size: Character strategy — target character count per chunk
        chunk_overlap: Character strategy — overlapping characters between chunks
        semantic_threshold: Semantic strategy — percentile threshold (0–100)
            for splitting; lower → more (smaller) chunks
        api_key: OpenAI API key forwarded to SemanticChunker embeddings
        provider: PROVIDER_OPENAI or PROVIDER_OLLAMA
        ollama_embedding_model: Ollama embedding model name for semantic chunking
        ollama_base_url: Ollama server base URL for semantic chunking

    Returns:
        Tuple of (list[str], list[dict]) — the text chunks and their FAISS
        metadata (each dict carries "source" and "page" keys).
    """
    if strategy == CHUNK_STRATEGY_SEMANTIC:
        embeddings = _get_embeddings(provider, api_key, ollama_embedding_model, ollama_base_url)
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=semantic_threshold,
        )
        all_docs = splitter.create_documents(
            [text for text, _, _ in texts_with_meta],
            metadatas=[{"source": name, "page": page} for _, name, page in texts_with_meta],
        )
        return [d.page_content for d in all_docs], [d.metadata for d in all_docs]

    # Default: Character splitting
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    all_chunks = []
    all_meta = []
    for text, filename, page_num in texts_with_meta:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        all_meta.extend([{"source": filename, "page": page_num}] * len(chunks))
    return all_chunks, all_meta


def _get_embeddings(
    provider: str,
    api_key: str | None,
    ollama_embedding_model: str,
    ollama_base_url: str,
) -> Any:
    """Return the appropriate embeddings object for the given provider."""
    if provider == PROVIDER_OLLAMA:
        return OllamaEmbeddings(model=ollama_embedding_model, base_url=ollama_base_url)
    if provider == PROVIDER_CLAUDE:
        # Anthropic does not offer an embedding API; use a local sentence-transformers model.
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415

        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    kwargs = {"api_key": api_key} if api_key else {}
    return OpenAIEmbeddings(**kwargs)


def get_vectorstore(
    text_chunks: list[str],
    metadatas: list[dict] | None = None,
    api_key: str | None = None,
    provider: str = PROVIDER_OPENAI,
    ollama_embedding_model: str = OLLAMA_DEFAULT_EMBEDDING_MODEL,
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL,
) -> FAISS:
    embeddings = _get_embeddings(provider, api_key, ollama_embedding_model, ollama_base_url)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings, metadatas=metadatas or [])


def load_vectorstore(
    index_path: str,
    api_key: str | None = None,
    provider: str = PROVIDER_OPENAI,
    ollama_embedding_model: str = OLLAMA_DEFAULT_EMBEDDING_MODEL,
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL,
) -> FAISS | None:
    """Load a previously saved FAISS index from disk. Returns None on failure."""
    if not os.path.exists(index_path):
        return None
    try:
        embeddings = _get_embeddings(provider, api_key, ollama_embedding_model, ollama_base_url)
        # allow_dangerous_deserialization is required by FAISS.load_local() for pickle-based
        # indexes. Indexes are written only by this app (vectorstore.save_local()); loading
        # third-party index files from untrusted sources would be unsafe.
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.warning("Failed to load FAISS index from '%s': %s", index_path, e)
        st.warning(
            "Could not load the saved index. "
            "It may be corrupt or incompatible with the current embedding model. "
            "Re-process your documents to rebuild it."
        )
        return None


def get_conversation_chain(
    vectorstore: FAISS,
    model: str = DEFAULT_MODEL,
    stream_handler: StreamHandler | None = None,
    api_key: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    retrieval_mode: str = "Similarity",
    system_prompt: str = "",
    reranker_enabled: bool = False,
    provider: str = PROVIDER_OPENAI,
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL,
    doc_filter: list[str] | None = None,
    hybrid_enabled: bool = False,
    text_chunks: list[str] | None = None,
    chunk_metadatas: list[dict] | None = None,
) -> Runnable:
    """Build an LCEL conversational RAG chain (LangChain 0.3+).

    Uses a non-streaming LLM for question condensation via
    create_history_aware_retriever and a streaming-enabled LLM (with the
    given StreamHandler) for final answer generation via
    create_retrieval_chain so that only answer tokens appear in the UI.
    Chat history is passed explicitly on each invocation — no memory object
    is required.
    """
    callbacks = [stream_handler] if stream_handler else []
    if provider == PROVIDER_OLLAMA:
        answer_llm = ChatOllama(
            model=model,
            base_url=ollama_base_url,
            temperature=temperature,
            streaming=True,
            callbacks=callbacks,
            timeout=120,  # local models can be slow on first load
        )
        condense_llm = ChatOllama(
            model=model, base_url=ollama_base_url, temperature=0, timeout=60
        )
    elif provider == PROVIDER_CLAUDE:
        claude_kwargs = {"api_key": api_key} if api_key else {}
        answer_llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=True,
            callbacks=callbacks,
            timeout=60,
            **claude_kwargs,
        )
        condense_llm = ChatAnthropic(model=model, temperature=0, timeout=30, **claude_kwargs)
    else:
        kwargs = {"api_key": api_key} if api_key else {}
        answer_llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            callbacks=callbacks,
            timeout=60,
            **kwargs,
        )
        condense_llm = ChatOpenAI(model=model, temperature=0, timeout=30, **kwargs)

    # --- Retriever (with optional hybrid search, doc filter, and re-ranking) ---
    fetch_k = max(retrieval_k * 4, 20)
    # Downstream wrappers (filter / reranker) need extra candidates from the base retriever.
    needs_extra = reranker_enabled or bool(doc_filter)

    if hybrid_enabled and text_chunks:
        base_ret: Any = HybridRetriever(
            vectorstore=vectorstore,
            corpus=text_chunks,
            corpus_metadatas=chunk_metadatas or [],
            top_k=fetch_k if needs_extra else retrieval_k,
            retrieval_mode=retrieval_mode,
        )
    elif retrieval_mode == "MMR":
        base_ret = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k if needs_extra else retrieval_k, "fetch_k": fetch_k},
        )
    else:
        base_ret = vectorstore.as_retriever(
            search_kwargs={"k": fetch_k if needs_extra else retrieval_k}
        )

    if doc_filter:
        base_ret = FilteredRetriever(
            base_retriever=base_ret, allowed_sources=doc_filter, top_k=retrieval_k
        )

    retriever = (
        RerankingRetriever(base_retriever=base_ret, top_k=retrieval_k, fetch_k=fetch_k)
        if reranker_enabled
        else base_ret
    )

    # --- History-aware retriever: rewrites the question given chat history ---
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the chat history and the latest user question, which may reference "
                "prior context, formulate a standalone question that can be understood "
                "without the history. Do NOT answer it — only reformulate if needed, "
                "otherwise return it as-is.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        condense_llm, retriever, contextualize_prompt
    )

    # --- QA chain: answer using retrieved context ---
    prefix = f"{system_prompt.strip()}\n\n" if system_prompt.strip() else ""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{prefix}Use the following context to answer the question.\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(answer_llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking retriever (opt-in; requires sentence-transformers)
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource
def _load_cross_encoder(model_name: str) -> Any:
    """Load and cache a CrossEncoder model for the lifetime of the server process.

    The model weights (~50 MB) are downloaded once on first use and reused
    across all Streamlit reruns and chat turns.
    """
    return CrossEncoder(model_name)


class RerankingRetriever(BaseRetriever):
    """Wraps any retriever and re-ranks its candidates with a cross-encoder.

    Fetches up to ``fetch_k`` documents from the base retriever, scores them
    with a cross-encoder, and returns the ``top_k`` highest-scoring ones.
    Falls back to the original ordering when sentence-transformers is not
    installed or an error occurs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any
    top_k: int = DEFAULT_RETRIEVAL_K
    fetch_k: int = DEFAULT_RETRIEVAL_K * 4
    model_name: str = RERANKER_MODEL

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        if len(candidates) <= 1:
            return candidates[: self.top_k]
        try:
            encoder = _load_cross_encoder(self.model_name)
            scores = encoder.predict([(query, d.page_content) for d in candidates])
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[: self.top_k]]
        except Exception as e:
            logger.warning("Re-ranking failed, returning original order: %s", e)
            return candidates[: self.top_k]


# ---------------------------------------------------------------------------
# Hybrid retriever — BM25 keyword search fused with FAISS via RRF
# ---------------------------------------------------------------------------


@st.cache_resource
def _build_bm25(corpus_key: int, corpus: tuple[str, ...]) -> Any:
    """Build and cache a BM25Okapi index for the given text corpus.

    The *corpus_key* argument is a cheap hash used as the cache key so that a
    new index is built automatically whenever the processed documents change.
    Returns None when rank-bm25 is not installed.
    """
    try:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415
    except ImportError:
        return None
    tokenized = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized)


class HybridRetriever(BaseRetriever):
    """Combines BM25 keyword search with FAISS vector search via Reciprocal Rank Fusion.

    Both retrievers independently rank up to *fetch_k* candidate documents.
    Their ranked lists are merged with the RRF formula
    ``score = Σ 1 / (rrf_k + rank)`` so that a chunk ranking highly in either
    list rises to the top even when it is absent from the other list.
    Falls back to pure vector search when rank-bm25 is not installed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    corpus: list[str]
    corpus_metadatas: list[dict]
    top_k: int = DEFAULT_RETRIEVAL_K
    rrf_k: int = 60  # standard RRF constant — dampens rank-1 dominance
    retrieval_mode: str = "Similarity"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        fetch_k = max(self.top_k * 4, 20)

        # --- Vector branch ---
        if self.retrieval_mode == "MMR":
            vector_docs = self.vectorstore.max_marginal_relevance_search(
                query, k=fetch_k, fetch_k=fetch_k * 2
            )
        else:
            vector_docs = self.vectorstore.similarity_search(query, k=fetch_k)

        # --- BM25 branch ---
        bm25_docs: list[Document] = []
        bm25_index = _build_bm25(id(self.corpus), tuple(self.corpus))
        if bm25_index is not None:
            scores = bm25_index.get_scores(query.lower().split())
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                :fetch_k
            ]
            bm25_docs = [
                Document(page_content=self.corpus[i], metadata=self.corpus_metadatas[i])
                for i in top_indices
            ]

        # --- RRF fusion ---
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        ranked = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in ranked[: self.top_k]]


# ---------------------------------------------------------------------------
# Filtered retriever — restricts results to a chosen subset of source files
# ---------------------------------------------------------------------------


class FilteredRetriever(BaseRetriever):
    """Post-filters any base retriever to only return chunks from allowed sources.

    Fetches a larger candidate set from the wrapped retriever then applies the
    source filter, ensuring the final list has up to *top_k* documents even
    when many candidates come from excluded files.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any
    allowed_sources: list[str]
    top_k: int = DEFAULT_RETRIEVAL_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        filtered = [d for d in candidates if d.metadata.get("source") in self.allowed_sources]
        return filtered[: self.top_k]


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------


def generate_suggested_questions(
    text_chunks: list[str],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    provider: str = PROVIDER_OPENAI,
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL,
) -> list[str]:
    """Ask the LLM to propose 3-5 questions that users could ask about the documents.

    Takes a sample of the first ~3000 characters from the combined chunks and
    returns a list of question strings. Returns an empty list on any error so
    the caller can degrade gracefully.
    """
    sample = " ".join(text_chunks)[:3000].strip()
    if not sample:
        return []
    prompt = (
        "Based on the following document excerpt, suggest exactly 5 specific and useful "
        "questions a user might ask. Return ONLY the questions, one per line, no numbering "
        "or bullet points.\n\nExcerpt:\n" + sample
    )
    try:
        if provider == PROVIDER_OLLAMA:
            llm = ChatOllama(model=model, base_url=ollama_base_url, temperature=0.3)
        else:
            kwargs = {"api_key": api_key} if api_key else {}
            llm = ChatOpenAI(model=model, temperature=0.3, **kwargs)
        result = llm.invoke(prompt)
        lines = [ln.strip() for ln in result.content.splitlines() if ln.strip()]
        return lines[:5]
    except Exception as e:
        logger.warning("Failed to generate suggested questions: %s", e)
        return []


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_sources(sources: list[Document]) -> None:
    """Render source documents inside an expander below a bot message."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        seen = set()
        for doc in sources:
            filename = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            page_label = f" · p. {page}" if page else ""
            preview = doc.page_content[:250].replace("\n", " ").strip()
            key = (filename, page, preview[:50])
            if key in seen:
                continue
            seen.add(key)
            st.markdown(f"**{filename}**{page_label}  \n{preview}…")


def format_conversation_as_markdown(
    chat_history: list[BaseMessage], sources: list[list[Document]]
) -> str:
    """Serialize the conversation to a Markdown string for download."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Chat Export — {date_str}", ""]
    bot_turn_idx = 0
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            lines.append(f"**You:** {message.content}")
            lines.append("")
        else:
            lines.append(f"**Bot:** {message.content}")
            lines.append("")
            srcs = sources[bot_turn_idx] if bot_turn_idx < len(sources) else []
            if srcs:
                seen = set()
                src_parts = []
                for doc in srcs:
                    name = doc.metadata.get("source", "unknown")
                    if name not in seen:
                        seen.add(name)
                        src_parts.append(name)
                lines.append(f"> *Sources: {', '.join(src_parts)}*")
                lines.append("")
            bot_turn_idx += 1
    return "\n".join(lines)


def render_chat_history() -> None:
    """Render all previous turns from session state using st.chat_message()."""
    history = st.session_state.chat_history or []
    feedback = st.session_state.get("feedback", [])
    bot_turn_idx = 0
    for i, message in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        with st.chat_message(role):
            st.write(message.content)
            if role == "assistant":
                srcs = (
                    st.session_state.sources[bot_turn_idx]
                    if bot_turn_idx < len(st.session_state.sources)
                    else []
                )
                _render_sources(srcs)
                # Answer quality feedback buttons
                current_rating = feedback[bot_turn_idx] if bot_turn_idx < len(feedback) else None
                up_col, down_col, _ = st.columns([1, 1, 8])
                if up_col.button(
                    "👍" if current_rating != "up" else "✅",
                    key=f"feedback_up_{bot_turn_idx}",
                    help="This answer was helpful",
                ):
                    if bot_turn_idx < len(st.session_state.feedback):
                        st.session_state.feedback[bot_turn_idx] = (
                            None if current_rating == "up" else "up"
                        )
                    st.rerun()
                if down_col.button(
                    "👎" if current_rating != "down" else "✅",
                    key=f"feedback_down_{bot_turn_idx}",
                    help="This answer was not helpful",
                ):
                    if bot_turn_idx < len(st.session_state.feedback):
                        st.session_state.feedback[bot_turn_idx] = (
                            None if current_rating == "down" else "down"
                        )
                    st.rerun()
                bot_turn_idx += 1


def _truncate_history(history: list[BaseMessage], max_turns: int) -> list[BaseMessage]:
    """Return the last *max_turns* human/AI pairs from *history*.

    Prevents unbounded context growth from crashing the LLM API call.
    A "turn" is one human message + one AI message, so we keep at most
    ``max_turns * 2`` messages.
    """
    max_messages = max_turns * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def handle_userinput(user_question: str) -> None:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process your PDF documents first.")
        return
    if len(user_question) > MAX_QUESTION_LENGTH:
        st.warning(
            f"Your question is too long ({len(user_question):,} characters). "
            f"Please limit it to {MAX_QUESTION_LENGTH:,} characters."
        )
        return

    # Show the new user message
    with st.chat_message("user"):
        st.write(user_question)

    # Stream the bot response into a placeholder inside the chat bubble
    with st.chat_message("assistant"):
        stream_container = st.empty()
        # Show a waiting indicator until the first token arrives
        stream_container.markdown("*Thinking…*")
        stream_handler = StreamHandler(stream_container)

        try:
            provider = st.session_state.provider
            if provider == PROVIDER_OLLAMA:
                active_model = st.session_state.ollama_model
                active_api_key = None
            elif provider == PROVIDER_CLAUDE:
                active_model = st.session_state.claude_model
                active_api_key = get_claude_api_key()
            else:
                active_model = st.session_state.model
                active_api_key = get_api_key()
            chain = get_conversation_chain(
                st.session_state.vectorstore,
                active_model,
                stream_handler,
                api_key=active_api_key,
                temperature=st.session_state.temperature,
                retrieval_k=st.session_state.retrieval_k,
                retrieval_mode=st.session_state.retrieval_mode,
                system_prompt=st.session_state.system_prompt,
                reranker_enabled=st.session_state.reranker_enabled,
                provider=provider,
                ollama_base_url=st.session_state.ollama_base_url,
                doc_filter=st.session_state.doc_filter or None,
                hybrid_enabled=st.session_state.hybrid_enabled,
                text_chunks=st.session_state.text_chunks
                if st.session_state.hybrid_enabled
                else None,
                chunk_metadatas=st.session_state.chunk_metadatas
                if st.session_state.hybrid_enabled
                else None,
            )
            payload = {
                "input": user_question,
                "chat_history": _truncate_history(st.session_state.chat_history, MAX_HISTORY_TURNS),
            }
            if provider == PROVIDER_OPENAI:
                from langchain_community.callbacks import get_openai_callback  # noqa: PLC0415

                with get_openai_callback() as cb:
                    response = chain.invoke(payload)
                tracker = st.session_state.cost_tracker
                tracker["turns"] += 1
                tracker["prompt_tokens"] += cb.prompt_tokens
                tracker["completion_tokens"] += cb.completion_tokens
                tracker["total_cost"] += cb.total_cost
            elif provider == PROVIDER_CLAUDE:
                response = chain.invoke(payload)
                # Estimate cost from CLAUDE_COST_PER_1K using token counts from response metadata.
                tracker = st.session_state.cost_tracker
                tracker["turns"] += 1
                usage = response.get("answer_metadata", {}) if isinstance(response, dict) else {}
                prompt_tokens = usage.get("input_tokens", 0)
                completion_tokens = usage.get("output_tokens", 0)
                tracker["prompt_tokens"] += prompt_tokens
                tracker["completion_tokens"] += completion_tokens
                in_cost, out_cost = CLAUDE_COST_PER_1K.get(
                    st.session_state.claude_model, (0.0, 0.0)
                )
                tracker["total_cost"] += (prompt_tokens / 1000) * in_cost + (
                    completion_tokens / 1000
                ) * out_cost
            else:
                response = chain.invoke(payload)

            answer = response["answer"]
            new_sources = response.get("context", [])
            # Append the new turn to the shared history list
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.session_state.sources.append(new_sources)
            st.session_state.feedback.append(None)  # no rating yet
        except Exception as e:
            logger.error("Error generating response: %s", e, exc_info=True)
            stream_container.empty()
            err_lower = str(e).lower()
            if any(k in err_lower for k in ("api key", "authentication", "unauthorized", "401")):
                st.error(
                    "Authentication failed. Check your API key in the sidebar or .env file."
                )
            elif any(k in err_lower for k in ("rate limit", "429", "quota")):
                st.error(
                    "Rate limit reached. Wait a moment before sending another message."
                )
            elif ("context" in err_lower and "length" in err_lower) or (
                "maximum" in err_lower and "token" in err_lower
            ):
                st.error(
                    "The conversation exceeded the model's context limit. "
                    "Start a new conversation to continue."
                )
            elif any(k in err_lower for k in ("connection", "timeout", "network", "unreachable")):
                st.error(
                    "Connection error. Check your network connection and try again."
                )
            else:
                st.error(
                    "An error occurred while generating a response. "
                    "Try again or check your settings."
                )
            return

        # Replace streaming placeholder with final answer
        stream_container.markdown(answer)
        _render_sources(new_sources)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # --- Session state initialisation ---
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "chunk_strategy" not in st.session_state:
        st.session_state.chunk_strategy = CHUNK_STRATEGY_CHAR
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200
    if "semantic_threshold" not in st.session_state:
        st.session_state.semantic_threshold = 95
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = DEFAULT_RETRIEVAL_K
    if "retrieval_mode" not in st.session_state:
        st.session_state.retrieval_mode = RETRIEVAL_MODES[0]
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    if "reranker_enabled" not in st.session_state:
        st.session_state.reranker_enabled = False
    if "active_slot" not in st.session_state:
        st.session_state.active_slot = DEFAULT_SLOT
    if "provider" not in st.session_state:
        st.session_state.provider = PROVIDER_OPENAI
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = OLLAMA_DEFAULT_MODEL
    if "ollama_embedding_model" not in st.session_state:
        st.session_state.ollama_embedding_model = OLLAMA_DEFAULT_EMBEDDING_MODEL
    if "ollama_base_url" not in st.session_state:
        st.session_state.ollama_base_url = OLLAMA_DEFAULT_BASE_URL
    if "claude_model" not in st.session_state:
        st.session_state.claude_model = DEFAULT_CLAUDE_MODEL
    # Hybrid search + per-document filter
    if "hybrid_enabled" not in st.session_state:
        st.session_state.hybrid_enabled = False
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = []
    if "chunk_metadatas" not in st.session_state:
        st.session_state.chunk_metadatas = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "doc_filter" not in st.session_state:
        st.session_state.doc_filter = []
    # Cost tracker (OpenAI only)
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = {
            "turns": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
        }

    # Auto-load the active slot's FAISS index on startup / slot switch.
    active_path = slot_path(st.session_state.active_slot)
    if st.session_state.vectorstore is None and os.path.exists(active_path):
        vs = load_vectorstore(
            active_path,
            api_key=get_api_key(),
            provider=st.session_state.provider,
            ollama_embedding_model=st.session_state.ollama_embedding_model,
            ollama_base_url=st.session_state.ollama_base_url,
        )
        if vs is not None:
            st.session_state.vectorstore = vs

    st.header("Chat with multiple PDFs :books:")
    st.caption(f"Slot: **{st.session_state.active_slot}**")
    render_chat_history()

    # Welcome screen when no index is loaded and conversation is empty
    if st.session_state.vectorstore is None and not st.session_state.chat_history:
        if st.session_state.provider == PROVIDER_OLLAMA:
            step1 = "Make sure your Ollama server is running (`ollama serve`)."
        else:
            step1 = "Enter your OpenAI API key in the sidebar."
        st.info(
            "**Get started in 3 steps:**\n\n"
            f"1. {step1}\n"
            "2. Upload one or more PDF files under **Your documents**.\n"
            "3. Click **Process** — then ask anything below!"
        )

    # Show suggested questions as one-click buttons (max 3 per row) when the chat is still empty
    if not st.session_state.chat_history and st.session_state.suggested_questions:
        st.caption("Suggested questions — click to ask:")
        questions = st.session_state.suggested_questions
        for row_start in range(0, len(questions), 3):
            batch = questions[row_start : row_start + 3]
            cols = st.columns(len(batch))
            for col, q in zip(cols, batch):
                if col.button(q, use_container_width=True):
                    handle_userinput(q)
                    st.rerun()

    if user_question := st.chat_input("Ask a question about your documents:"):
        handle_userinput(user_question)

    # --- Sidebar ---
    with st.sidebar:
        selected_provider = st.radio(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(st.session_state.provider),
            horizontal=True,
            help=(
                "OpenAI: cloud-based models (API key required).  \n"
                "Claude: Anthropic models (API key required).  \n"
                "Ollama: locally-running models (no API key needed, "
                "Ollama must be running on your machine)."
            ),
        )
        if selected_provider != st.session_state.provider:
            st.session_state.provider = selected_provider
            st.session_state.vectorstore = None
            _clear_conversation()
            st.toast(f"Switched to {selected_provider} — index and history cleared.", icon="ℹ️")

        st.divider()

        if st.session_state.provider == PROVIDER_OPENAI:
            st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-… (leave blank to use .env)",
                key="api_key_input",
            )
            if not get_api_key():
                st.warning(
                    "No API key found. Enter one above or set OPENAI_API_KEY in your .env file."
                )
            selected_model = st.selectbox(
                "OpenAI model",
                AVAILABLE_MODELS,
                index=AVAILABLE_MODELS.index(st.session_state.model),
            )
            if selected_model != st.session_state.model:
                st.session_state.model = selected_model
                _clear_conversation()
                st.toast(f"Switched to {selected_model} — chat history cleared.", icon="ℹ️")
        elif st.session_state.provider == PROVIDER_CLAUDE:
            st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-… (leave blank to use .env)",
                key="claude_api_key_input",
            )
            if not get_claude_api_key():
                st.warning(
                    "No API key found. Enter one above or set ANTHROPIC_API_KEY in your .env file."
                )
            selected_claude_model = st.selectbox(
                "Claude model",
                AVAILABLE_CLAUDE_MODELS,
                index=AVAILABLE_CLAUDE_MODELS.index(st.session_state.claude_model),
            )
            if selected_claude_model != st.session_state.claude_model:
                st.session_state.claude_model = selected_claude_model
                _clear_conversation()
                st.toast(
                    f"Switched to {selected_claude_model} — chat history cleared.", icon="ℹ️"
                )
            st.caption(
                "Embeddings use a local `all-MiniLM-L6-v2` model (no additional API key needed)."
            )
        else:
            st.caption(
                "**Ollama** — models run locally.  \n"
                "Start the server with `ollama serve` and pull models with `ollama pull <model>`."
            )
            new_base_url = st.text_input(
                "Ollama base URL",
                value=st.session_state.ollama_base_url,
                placeholder=OLLAMA_DEFAULT_BASE_URL,
                help="URL of the Ollama server (default: http://localhost:11434).",
            )
            if new_base_url != st.session_state.ollama_base_url:
                st.session_state.ollama_base_url = new_base_url
            new_ollama_model = st.text_input(
                "Chat model",
                value=st.session_state.ollama_model,
                placeholder="e.g. llama3.2, mistral, qwen2.5, gemma2",
                help="Name of the Ollama model to use for chat (must be pulled first).",
            )
            if new_ollama_model and new_ollama_model != st.session_state.ollama_model:
                st.session_state.ollama_model = new_ollama_model
                _clear_conversation()
                st.toast(f"Switched to {new_ollama_model} — chat history cleared.", icon="ℹ️")
            new_embedding_model = st.text_input(
                "Embedding model",
                value=st.session_state.ollama_embedding_model,
                placeholder="e.g. nomic-embed-text, mxbai-embed-large, all-minilm",
                help=(
                    "Ollama embedding model for building the vector index.  \n"
                    "Changing this requires re-processing your documents."
                ),
            )
            if (
                new_embedding_model
                and new_embedding_model != st.session_state.ollama_embedding_model
            ):
                st.session_state.ollama_embedding_model = new_embedding_model
                st.session_state.vectorstore = None
                st.toast(
                    f"Embedding model changed to {new_embedding_model} — "
                    "please re-process your documents.",
                    icon="ℹ️",
                )

        with st.expander("LLM & Retrieval", expanded=False):
            st.session_state.system_prompt = st.text_area(
                "System prompt",
                value=st.session_state.system_prompt,
                placeholder="e.g. Answer only in German. Be concise.",
                help=(
                    "Optional instructions prepended to every QA prompt. "
                    "Use to set language, tone, or domain constraints."
                ),
                height=80,
            )
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.05,
                help="Higher = more creative answers; lower = more deterministic.",
            )
            st.session_state.retrieval_k = st.slider(
                "Retrieved chunks (k)",
                min_value=1,
                max_value=10,
                value=st.session_state.retrieval_k,
                step=1,
                help="Number of document chunks passed to the LLM per question.",
            )
            st.session_state.retrieval_mode = st.radio(
                "Retrieval mode",
                RETRIEVAL_MODES,
                index=RETRIEVAL_MODES.index(st.session_state.retrieval_mode),
                horizontal=True,
                help=(
                    "Similarity: closest chunks by cosine distance.  \n"
                    "MMR: balances relevance with diversity to avoid repetitive chunks."
                ),
            )
            st.session_state.reranker_enabled = st.toggle(
                "Cross-encoder re-ranking",
                value=st.session_state.reranker_enabled,
                help=(
                    "Re-rank retrieved chunks with a cross-encoder before answering.  \n"
                    "First use downloads ~50 MB model weights."
                ),
            )
            st.session_state.hybrid_enabled = st.toggle(
                "Hybrid search (BM25 + Vector)",
                value=st.session_state.hybrid_enabled,
                help=(
                    "Fuse keyword-based BM25 search with FAISS vector search using  \n"
                    "Reciprocal Rank Fusion (RRF).  \n"
                    "Requires `rank-bm25` (`pip install rank-bm25`).  \n"
                    "Improves recall for exact-term queries that pure vector search misses."
                ),
            )
            if st.session_state.indexed_files and len(st.session_state.indexed_files) > 1:
                st.session_state.doc_filter = st.multiselect(
                    "Filter by document",
                    options=st.session_state.indexed_files,
                    default=st.session_state.doc_filter,
                    help=(
                        "Restrict retrieval to the selected files only.  \n"
                        "Leave empty to search across all indexed documents."
                    ),
                )
            elif st.session_state.indexed_files:
                st.caption(f"Indexed: **{st.session_state.indexed_files[0]}**")

        # Context window indicator — warn when history truncation is active
        total_turns = len(st.session_state.chat_history) // 2
        if total_turns > MAX_HISTORY_TURNS:
            st.caption(
                f":warning: Context limited to the last {MAX_HISTORY_TURNS} of "
                f"{total_turns} turns to stay within the model's context window."
            )

        # Cost tracker (OpenAI and Claude — Ollama is free / local)
        if st.session_state.provider in (PROVIDER_OPENAI, PROVIDER_CLAUDE):
            tracker = st.session_state.cost_tracker
            if tracker["turns"] > 0:
                with st.expander("Cost tracker", expanded=False):
                    st.metric("Session cost (USD)", f"${tracker['total_cost']:.4f}")
                    st.caption(
                        f"Turns: {tracker['turns']} · "
                        f"Prompt tokens: {tracker['prompt_tokens']:,} · "
                        f"Completion tokens: {tracker['completion_tokens']:,}"
                    )
                    if st.session_state.provider == PROVIDER_CLAUDE:
                        st.caption("Cost is estimated from public pricing; actual billing may differ.")
                    if st.button("Reset cost tracker", use_container_width=True):
                        st.session_state.cost_tracker = {
                            "turns": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_cost": 0.0,
                        }
                        st.rerun()

        st.divider()
        if st.session_state.chat_history:
            md_export = format_conversation_as_markdown(
                st.session_state.chat_history, st.session_state.sources
            )
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            st.download_button(
                "Export conversation (.md)",
                data=md_export,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True,
            )
            if st.button("New conversation", use_container_width=True):
                _clear_conversation()
                st.rerun()

        st.divider()
        with st.expander("Sessions", expanded=False):
            saved = list_sessions()
            new_name = st.text_input(
                "Session name",
                placeholder="my-session",
                key="session_name_input",
            )
            if st.button("Save session", use_container_width=True, disabled=not new_name):
                if st.session_state.chat_history:
                    try:
                        save_session(
                            new_name.strip(),
                            st.session_state.chat_history,
                            st.session_state.sources,
                        )
                        st.success(f"Saved as '{new_name.strip()}'")
                    except ValueError as exc:
                        st.error(str(exc))
                else:
                    st.warning("Nothing to save — start a conversation first.")

            if saved:
                # Filter sessions by search query
                search_query = st.text_input(
                    "Search sessions",
                    placeholder="Filter by name…",
                    key="session_search_input",
                )
                visible = (
                    [s for s in saved if search_query.lower() in s.lower()]
                    if search_query
                    else saved
                )

                if visible:
                    selected_session = st.selectbox("Load session", [""] + visible)
                    load_col, del_col2 = st.columns(2)
                    if load_col.button(
                        "Load", use_container_width=True, disabled=not selected_session
                    ):
                        hist, srcs = load_session(selected_session)
                        if hist is not None:
                            st.session_state.chat_history = hist
                            st.session_state.sources = srcs
                            st.session_state.feedback = [None] * (len(hist) // 2)
                            st.session_state.suggested_questions = []
                            st.rerun()
                        else:
                            st.error("Could not load session.")
                    confirm_del_session = del_col2.checkbox(
                        "Confirm delete",
                        key="confirm_del_session",
                        help="Tick to enable session deletion.",
                    )
                    if del_col2.button(
                        "Delete",
                        use_container_width=True,
                        disabled=not selected_session or not confirm_del_session,
                    ):
                        delete_session(selected_session)
                        st.rerun()

                    # Bulk delete
                    st.caption("Bulk delete:")
                    to_delete = st.multiselect(
                        "Select sessions to delete",
                        options=visible,
                        label_visibility="collapsed",
                    )
                    if st.button(
                        f"Delete {len(to_delete)} selected",
                        disabled=not to_delete,
                        use_container_width=True,
                    ):
                        for s in to_delete:
                            delete_session(s)
                        st.rerun()
                else:
                    st.caption(f"No sessions match '{search_query}'.")
            else:
                st.caption("No saved sessions yet.")

        st.divider()
        with st.expander("Index slots", expanded=False):
            existing_slots = list_index_slots() or [DEFAULT_SLOT]
            chosen_slot = st.selectbox(
                "Active slot",
                existing_slots,
                index=existing_slots.index(st.session_state.active_slot)
                if st.session_state.active_slot in existing_slots
                else 0,
                help="Each slot stores an independent FAISS index.",
            )
            if chosen_slot != st.session_state.active_slot:
                st.session_state.active_slot = chosen_slot
                st.session_state.vectorstore = None
                _clear_conversation()
                st.rerun()

            new_slot_name = st.text_input(
                "New slot name", placeholder="e.g. project-alpha", key="new_slot_input"
            )
            cre_col, del_col = st.columns(2)
            if cre_col.button("Create", use_container_width=True, disabled=not new_slot_name):
                try:
                    name = _safe_name(new_slot_name, "Slot name")
                    os.makedirs(slot_path(name), mode=0o700, exist_ok=True)
                    st.session_state.active_slot = name
                    st.session_state.vectorstore = None
                    _clear_conversation()
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            confirm_del_slot = del_col.checkbox(
                "Confirm delete",
                key="confirm_del_slot",
                help="Tick to enable slot deletion.",
            )
            if del_col.button(
                "Delete slot",
                use_container_width=True,
                disabled=chosen_slot == DEFAULT_SLOT or not confirm_del_slot,
                help="Cannot delete the default slot. Tick 'Confirm delete' first.",
            ):
                shutil.rmtree(slot_path(chosen_slot), ignore_errors=True)
                st.session_state.active_slot = DEFAULT_SLOT
                st.session_state.vectorstore = None
                _clear_conversation()
                st.rerun()

        st.subheader("Your documents")
        if st.session_state.vectorstore is not None:
            st.success("Index loaded — ready to chat.", icon="✅")
        else:
            st.info("No index loaded. Upload PDFs and click **Process**.", icon="ℹ️")
        with st.expander("Chunking settings", expanded=False):
            selected_strategy = st.radio(
                "Strategy",
                CHUNK_STRATEGIES,
                index=CHUNK_STRATEGIES.index(st.session_state.chunk_strategy),
                help=(
                    "Character: fast rule-based splitting by character count.  \n"
                    "Semantic: uses embeddings to find natural topic boundaries "
                    "(makes API calls during processing)."
                ),
            )
            st.session_state.chunk_strategy = selected_strategy

            if selected_strategy == CHUNK_STRATEGY_CHAR:
                st.session_state.chunk_size = st.slider(
                    "Chunk size (characters)",
                    min_value=100,
                    max_value=3000,
                    value=st.session_state.chunk_size,
                    step=100,
                    help="Maximum number of characters per chunk.",
                )
                st.session_state.chunk_overlap = st.slider(
                    "Overlap (characters)",
                    min_value=0,
                    max_value=min(500, st.session_state.chunk_size - 1),
                    value=min(st.session_state.chunk_overlap, st.session_state.chunk_size - 1),
                    step=50,
                    help="Characters shared between adjacent chunks to preserve context.",
                )
            else:
                st.session_state.semantic_threshold = st.slider(
                    "Breakpoint percentile",
                    min_value=50,
                    max_value=99,
                    value=st.session_state.semantic_threshold,
                    step=1,
                    help=(
                        "Cosine-distance percentile used to detect topic boundaries.  \n"
                        "Lower → more (smaller) chunks. Higher → fewer (larger) chunks."
                    ),
                )
                if st.session_state.provider == PROVIDER_OPENAI:
                    st.caption(
                        ":warning: Semantic chunking calls the OpenAI Embeddings API "
                        "for every document during processing."
                    )
                else:
                    st.caption(
                        ":warning: Semantic chunking calls the Ollama Embeddings API "
                        "for every document during processing."
                    )

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type="pdf",
        )
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file before processing.")
            elif st.session_state.provider == PROVIDER_OPENAI and not get_api_key():
                st.error(
                    "No API key found. Enter your OpenAI API key in the sidebar "
                    "or set OPENAI_API_KEY in your .env file before processing."
                )
            else:
                try:
                    with st.status("Processing documents…", expanded=True) as proc_status:
                        # Step 1 — text extraction (parallelised per-file)
                        st.write(f"Extracting text from {len(pdf_docs)} PDF(s)…")
                        prog = st.progress(0.0)
                        texts_with_meta = []
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future_to_pdf = {
                                pool.submit(_extract_single_pdf, pdf): (idx, pdf)
                                for idx, pdf in enumerate(pdf_docs)
                            }
                            completed = 0
                            for future in concurrent.futures.as_completed(future_to_pdf):
                                idx, pdf = future_to_pdf[future]
                                try:
                                    pages, warns = future.result()
                                except Exception as exc:
                                    logger.error("Unexpected error extracting '%s': %s", pdf.name, exc)
                                    warns = [f"Unexpected error reading '{pdf.name}'."]
                                    pages = []
                                for w in warns:
                                    st.warning(w)
                                texts_with_meta.extend(pages)
                                completed += 1
                                prog.progress(completed / len(pdf_docs) * 0.35)

                        if not texts_with_meta:
                            proc_status.update(label="Extraction failed", state="error")
                            st.error(
                                "No text could be extracted from the uploaded PDFs. "
                                "Ensure the files are not scanned images or password-protected."
                            )
                        else:
                            # Step 2 — chunking
                            st.write("Chunking text…")
                            prog.progress(0.4)
                            text_chunks, metadatas = get_text_chunks(
                                texts_with_meta,
                                strategy=st.session_state.chunk_strategy,
                                chunk_size=st.session_state.chunk_size,
                                chunk_overlap=st.session_state.chunk_overlap,
                                semantic_threshold=st.session_state.semantic_threshold,
                                api_key=get_api_key(),
                                provider=st.session_state.provider,
                                ollama_embedding_model=st.session_state.ollama_embedding_model,
                                ollama_base_url=st.session_state.ollama_base_url,
                            )

                            # Step 3 — embedding + FAISS
                            st.write(f"Embedding {len(text_chunks)} chunks and building index…")
                            prog.progress(0.55)
                            vectorstore = get_vectorstore(
                                text_chunks,
                                metadatas,
                                api_key=get_api_key(),
                                provider=st.session_state.provider,
                                ollama_embedding_model=st.session_state.ollama_embedding_model,
                                ollama_base_url=st.session_state.ollama_base_url,
                            )

                            # Step 4 — persist
                            st.write("Saving index to disk…")
                            prog.progress(0.9)
                            os.makedirs(active_path, mode=0o700, exist_ok=True)
                            vectorstore.save_local(active_path)
                            save_index_metadata(
                                filenames=list(dict.fromkeys(t[1] for t in texts_with_meta)),
                                chunk_count=len(text_chunks),
                                index_path=active_path,
                            )
                            prog.progress(1.0)

                            st.session_state.vectorstore = vectorstore
                            st.session_state.text_chunks = text_chunks
                            st.session_state.chunk_metadatas = metadatas
                            st.session_state.indexed_files = list(
                                dict.fromkeys(t[1] for t in texts_with_meta)
                            )
                            st.session_state.doc_filter = []  # reset filter on new index
                            st.session_state.chat_history = []
                            st.session_state.sources = []

                            # Generate suggested questions in the background step
                            st.write("Generating suggested questions…")
                            active_model = (
                                st.session_state.ollama_model
                                if st.session_state.provider == PROVIDER_OLLAMA
                                else st.session_state.model
                            )
                            st.session_state.suggested_questions = generate_suggested_questions(
                                text_chunks,
                                api_key=get_api_key(),
                                model=active_model,
                                provider=st.session_state.provider,
                                ollama_base_url=st.session_state.ollama_base_url,
                            )
                            proc_status.update(
                                label=f"Done — {len(text_chunks)} chunks from "
                                f"{len(pdf_docs)} document(s)",
                                state="complete",
                                expanded=False,
                            )
                except Exception as e:
                    logger.error("Document processing failed: %s", e, exc_info=True)
                    st.error(f"Error processing documents: {e}")

        if os.path.exists(active_path):
            st.divider()
            meta = load_index_metadata(active_path)
            if meta:
                ts = datetime.fromisoformat(meta["timestamp"])
                age_min = int((datetime.now(timezone.utc) - ts).total_seconds() / 60)
                if age_min < 60:
                    age_str = f"{age_min} min ago"
                elif age_min < 60 * 24:
                    age_str = f"{age_min // 60} h ago"
                else:
                    age_str = f"{age_min // (60 * 24)} d ago"
                st.caption(
                    f"**Index loaded** · {meta['chunks']} chunks · {age_str}  \n"
                    + ", ".join(meta["files"])
                )
            confirm_clear = st.checkbox(
                "Confirm clear",
                key="confirm_clear_index",
                help="Tick to enable index deletion.",
            )
            if st.button(
                "Clear saved index",
                disabled=not confirm_clear,
                help="Permanently deletes the FAISS index for this slot. Tick 'Confirm clear' first.",
            ):
                shutil.rmtree(active_path)
                st.session_state.vectorstore = None
                _clear_conversation()
                st.rerun()


if __name__ == "__main__":
    main()
