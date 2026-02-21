# CLAUDE.md — AI Assistant Reference for ask-multiple-pdfs

## Project Overview

**MultiPDF Chat App** is a single-script Python web application that allows users to upload multiple PDF files and ask natural language questions about their contents via a conversational interface. It implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI, FAISS, and Streamlit.

This is an educational project accompanying a YouTube tutorial. It is not intended for production use, and pull requests are not accepted upstream.

---

## Repository Structure

```
ask-multiple-pdfs/
├── app.py                # Sole application file — all business logic and Streamlit UI
├── htmlTemplates.py      # HTML/CSS templates (legacy; no longer imported by app.py)
├── requirements.txt      # Python dependencies (unpinned ranges, modern versions)
├── pyproject.toml        # Ruff linter/formatter configuration
├── Makefile              # Convenience targets: run, install, lint, format, test, docker-*
├── Dockerfile            # Single-stage python:3.11-slim image with HEALTHCHECK
├── docker-compose.yml    # Compose file: port 8501, volume mounts for indexes & sessions
├── .dockerignore         # Excludes .env, venv, runtime data from image
├── .env.example          # Template for required environment variables
├── .python-version       # pyenv Python version pin (3.11)
├── .gitignore            # Standard Python gitignore (excludes .env, __pycache__, etc.)
├── readme.md             # User-facing documentation
├── docs/
│   └── PDF-LangChain.jpg # Architecture diagram referenced in readme.md
└── tests/
    ├── __init__.py
    ├── test_chunking.py  # 9 tests for get_text_chunks()
    ├── test_pdf.py       # 5 tests for get_pdf_text()
    ├── test_metadata.py  # 5 tests for save/load_index_metadata()
    └── test_sessions.py  # 18 tests for session persistence helpers
```

Runtime directories (gitignored, created on first use):
- `faiss_indexes/{slot}/` — one FAISS index per named slot (default slot: `default`)
- `sessions/` — saved chat sessions as JSON files

---

## Architecture: RAG Pipeline

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store → Conversational Chain → Answer
```

| Step | Function | Implementation |
|---|---|---|
| Text extraction | `get_pdf_text()` | `pypdf.PdfReader` iterates pages; per-file errors shown as `st.warning()` |
| Chunking | `get_text_chunks()` | `CharacterTextSplitter` (size=1000, overlap=200) or `SemanticChunker` |
| Embedding | `get_vectorstore()` | `OpenAIEmbeddings` (default) or `OllamaEmbeddings` |
| Vector store | `get_vectorstore()` | `FAISS.from_texts()` — saved to `faiss_indexes/{slot}/` |
| Retrieval | `get_conversation_chain()` | Similarity, MMR, or Hybrid (BM25 + Vector via RRF); optionally wrapped by `FilteredRetriever` and/or `RerankingRetriever` |
| Conversational chain | `get_conversation_chain()` | LCEL: `create_history_aware_retriever` + `create_retrieval_chain`; chat history passed explicitly |
| Streaming | `StreamHandler` | `BaseCallbackHandler` subclass; writes tokens into `st.empty()` placeholder |
| Cost tracking | `handle_userinput()` | `get_openai_callback()` context manager; accumulates tokens + cost in session state |
| UI rendering | `handle_userinput()`, `render_chat_history()`, `main()` | Native `st.chat_message()` / `st.chat_input()` |

---

## Key Files

### `app.py` (~1 350 lines)

The entire application. All functions have full type annotations (Python 3.11 native generics).

**Constants**

| Name | Value | Purpose |
|---|---|---|
| `FAISS_INDEXES_DIR` | `"faiss_indexes"` | Base directory for all index slots |
| `DEFAULT_SLOT` | `"default"` | Name of the default index slot |
| `SESSIONS_DIR` | `"sessions"` | Directory for saved chat session JSON files |
| `AVAILABLE_MODELS` | `["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]` | Selectable OpenAI models |
| `DEFAULT_MODEL` | `AVAILABLE_MODELS[0]` | |
| `CHUNK_STRATEGY_CHAR/SEMANTIC` | string literals | Chunking strategy identifiers |
| `RETRIEVAL_MODES` | `["Similarity", "MMR"]` | FAISS retrieval modes (Hybrid is a separate toggle) |
| `DEFAULT_TEMPERATURE` | `0.0` | LLM temperature |
| `DEFAULT_RETRIEVAL_K` | `4` | Default number of retrieved chunks |
| `RERANKER_MODEL` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model name |
| `MAX_QUESTION_LENGTH` | `5_000` | Character cap for a single user question; longer inputs are rejected with a warning |
| `OPENAI_COST_PER_1K` | `dict[str, tuple[float, float]]` | Approximate input/output cost per 1 000 tokens per model (display only) |

**Index slot helpers**
- **`slot_path(slot)`** — returns `faiss_indexes/{slot}`
- **`list_index_slots()`** — returns sorted list of existing slot directory names
- **`save_index_metadata(filenames, chunk_count, index_path)`** — writes `metadata.json` into the slot dir
- **`load_index_metadata(index_path)`** — reads `metadata.json`; returns `None` if missing or corrupt

**Session persistence helpers** (serialize/deserialize `BaseMessage` and `Document` objects to/from JSON)
- `_serialize_messages()` / `_deserialize_messages()`
- `_serialize_sources()` / `_deserialize_sources()`
- **`list_sessions()`**, **`save_session()`**, **`load_session()`**, **`delete_session()`** — `load_session()` and `delete_session()` validate the supplied name via `_safe_name()` to prevent path traversal; invalid names return `(None, None)` / no-op respectively

**`_clear_conversation()`** — resets `chat_history`, `sources`, and `suggested_questions` in session state. Called in every code path that clears the conversation (provider switch, model switch, slot change, "New conversation", "Clear saved index").

**`get_api_key()`** — UI input priority over `OPENAI_API_KEY` env var; returns `None` if neither set.

**`StreamHandler(BaseCallbackHandler)`** — appends `▌` cursor on each token; removes it on LLM end. Uses `st.empty()` placeholder passed at construction time.

**`get_pdf_text(pdf_docs)`** → `list[tuple[str, str, int]]` — returns `(text, filename, page_number)` tuples (one per page); skips pages with no extractable text and emits `st.warning()` for files that fail to open.

**`get_text_chunks(texts_with_meta, strategy, chunk_size, chunk_overlap, semantic_threshold, api_key)`** → `tuple[list[str], list[dict]]` — supports:
- `CHUNK_STRATEGY_CHAR`: `CharacterTextSplitter(separator="\n", ...)`
- `CHUNK_STRATEGY_SEMANTIC`: `SemanticChunker` from `langchain-experimental`

**`get_vectorstore(text_chunks, metadatas, api_key)`** → `FAISS`

**`load_vectorstore(index_path, api_key)`** → `FAISS | None` — loads from `index_path`; shows `st.warning()` on failure.

**`get_conversation_chain(vectorstore, model, stream_handler, api_key, temperature, retrieval_k, retrieval_mode, system_prompt, reranker_enabled, provider, ollama_base_url, doc_filter, hybrid_enabled, text_chunks, chunk_metadatas)`** → `Runnable`:
- LCEL pipeline: `create_history_aware_retriever` (question condensing) → `create_stuff_documents_chain` (QA) → `create_retrieval_chain`
- Separate non-streaming `condense_llm` and streaming `answer_llm`
- `ChatPromptTemplate` with `MessagesPlaceholder` for history; optional system prompt prefix
- Retriever build order: base retriever (Hybrid or FAISS) → optional `FilteredRetriever` → optional `RerankingRetriever`
- `hybrid_enabled=True` builds a `HybridRetriever`; otherwise builds a plain FAISS retriever
- `doc_filter` (non-empty list) wraps the base retriever with `FilteredRetriever`
- `reranker_enabled=True` wraps with `RerankingRetriever`
- No `memory` parameter — chat history passed explicitly via `chain.invoke({"input": ..., "chat_history": ...})`
- Response keys: `response["answer"]` (str) and `response["context"]` (list[Document])

**`_build_bm25(corpus_key, corpus)`** — `@st.cache_resource` helper that builds and caches a `BM25Okapi` index; returns `None` when `rank-bm25` is not installed.

**`HybridRetriever(BaseRetriever)`** — combines BM25 and FAISS vector search using Reciprocal Rank Fusion (RRF, k=60). Fetches `max(top_k * 4, 20)` candidates from each branch; merges with `score = Σ 1/(rrf_k + rank)`. Falls back to pure vector search when `rank-bm25` is not installed.

**`FilteredRetriever(BaseRetriever)`** — post-filters any wrapped retriever's results to only return `Document` objects whose `metadata["source"]` is in `allowed_sources`. Returns up to `top_k` matching documents.

**`RerankingRetriever(BaseRetriever)`** — Pydantic model wrapping a base retriever. Calls `base_retriever.invoke(query)`, scores pairs with `CrossEncoder`, returns top `k`. Falls back to original order on any exception.

**`generate_suggested_questions(text_chunks, api_key, model)`** → `list[str]` — LLM call (temperature 0.3) on first 3000 chars of combined chunks; returns up to 5 questions.

**`_render_sources(sources)`** — expander with deduped filename + 250-char preview per source document.

**`format_conversation_as_markdown(chat_history, sources)`** → `str` — Markdown export for download.

**`render_chat_history()`** — replays all `st.session_state.chat_history` turns using `st.chat_message()`; shows sources below each assistant turn.

**`handle_userinput(user_question)`** — guards against missing vectorstore and questions exceeding `MAX_QUESTION_LENGTH`; renders user bubble; shows a "Thinking…" placeholder while waiting for the first streaming token; streams answer into assistant bubble via fresh `StreamHandler`; appends sources to `st.session_state.sources`. Errors are classified (auth, rate-limit, context-length, network) into user-friendly messages rather than raw exceptions. When using OpenAI, wraps `chain.invoke()` with `get_openai_callback()` and accumulates token counts and cost into `st.session_state.cost_tracker`.

**`main()`** — Streamlit entry point: `load_dotenv()`, session state init, auto-load of active slot's FAISS index, chat area (history + suggested questions + `st.chat_input()`), full sidebar.

### `htmlTemplates.py`

No longer imported by `app.py` (replaced by native `st.chat_message()` in Step 2). Kept for reference. Contains `css`, `bot_template`, `user_template` with self-contained SVG avatar data URIs.

### `tests/`

| File | Tests | What it covers |
|---|---|---|
| `test_chunking.py` | 9 | `get_text_chunks()`: parallel list lengths, metadata keys, multi-file/page attribution, chunk size bounds, relative chunk count, empty input, `ValueError` on `chunk_overlap > chunk_size` |
| `test_pdf.py` | 5 | `get_pdf_text()`: single/multiple PDFs, no-text excluded, broken PDF warning, mixed good/broken |
| `test_metadata.py` | 5 | `save_index_metadata()` / `load_index_metadata()`: roundtrip, missing file → None, corrupt JSON → None, overwrite, UTC ISO timestamp |
| `test_sessions.py` | 18 | `_serialize/deserialize_messages()`, `_serialize/deserialize_sources()`, `save/load/list/delete_session()`, `_safe_name()`, `_truncate_history()` |

---

## Streamlit Session State

| Key | Type | Description |
|---|---|---|
| `vectorstore` | `FAISS \| None` | Active vector store for the current slot |
| `chat_history` | `list[BaseMessage]` | Alternating HumanMessage / AIMessage; passed explicitly to each chain invocation |
| `sources` | `list[list[Document]]` | Source docs per bot turn |
| `model` | `str` | Selected OpenAI model name |
| `temperature` | `float` | LLM temperature (0.0–1.0) |
| `retrieval_k` | `int` | Number of chunks retrieved per question |
| `retrieval_mode` | `str` | `"Similarity"` or `"MMR"` |
| `system_prompt` | `str` | Optional system-level instruction prefix |
| `reranker_enabled` | `bool` | Toggle for cross-encoder re-ranking |
| `hybrid_enabled` | `bool` | Toggle for hybrid BM25 + vector search |
| `text_chunks` | `list[str]` | Raw text chunks stored after processing (used by `HybridRetriever`) |
| `chunk_metadatas` | `list[dict]` | Metadata dicts parallel to `text_chunks` |
| `indexed_files` | `list[str]` | Filenames present in the current index (used to populate the doc-filter multiselect) |
| `doc_filter` | `list[str]` | Selected filenames to restrict retrieval to; empty = no filter |
| `cost_tracker` | `dict` | Cumulative OpenAI cost: `turns`, `prompt_tokens`, `completion_tokens`, `total_cost` (float USD) |
| `chunk_strategy` | `str` | `CHUNK_STRATEGY_CHAR` or `CHUNK_STRATEGY_SEMANTIC` |
| `chunk_size` | `int` | Character splitter: chars per chunk |
| `chunk_overlap` | `int` | Character splitter: overlapping chars |
| `semantic_threshold` | `int` | Semantic splitter: breakpoint percentile (50–99) |
| `suggested_questions` | `list[str]` | LLM-generated one-click questions |
| `active_slot` | `str` | Active FAISS index slot name (default: `"default"`) |
| `provider` | `str` | `PROVIDER_OPENAI` or `PROVIDER_OLLAMA` |
| `ollama_model` | `str` | Ollama chat model name |
| `ollama_embedding_model` | `str` | Ollama embedding model name |
| `ollama_base_url` | `str` | Ollama server base URL |

---

## Environment Setup

### Prerequisites

- Python 3.11 (pinned via `.python-version` for pyenv)
- An OpenAI API key (required for default embeddings and LLM)
- Docker + Docker Compose (optional, for containerised deployment)

### Installation

```bash
git clone <repo-url>
cd ask-multiple-pdfs
python -m venv venv && source venv/bin/activate
make install
cp .env.example .env   # then set OPENAI_API_KEY
```

### Docker

```bash
cp .env.example .env   # set OPENAI_API_KEY
make docker-up         # builds image, starts container on port 8501
make docker-down       # stop
```

---

## Running the Application

```bash
make run   # or: streamlit run app.py
```

---

## Development Commands (Makefile)

| Command | Description |
|---|---|
| `make run` | Start the Streamlit app |
| `make install` | Install all dependencies from `requirements.txt` |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |
| `make test` | Run pytest (37 unit tests) |
| `make docker-build` | Build Docker image |
| `make docker-up` | Build and start via Docker Compose (detached) |
| `make docker-down` | Stop Docker Compose services |

---

## Dependencies

| Package | Version Range | Purpose |
|---|---|---|
| `langchain` | `>=0.3.0` | Base package; pulls in shared utilities |
| `langchain-classic` | `>=1.0.0` | `create_history_aware_retriever`, `create_retrieval_chain`, `create_stuff_documents_chain` |
| `langchain-openai` | `>=0.2.0` | OpenAI embeddings and chat models |
| `langchain-community` | `>=0.3.0` | FAISS vector store, `get_openai_callback`, HuggingFace integrations |
| `langchain-text-splitters` | `>=0.3.0` | `CharacterTextSplitter` |
| `langchain-ollama` | `>=0.2.0` | Ollama chat and embedding models |
| `langchain-experimental` | `>=0.3.0` | `SemanticChunker` for semantic chunking |
| `pypdf` | `>=4.0.0` | PDF text extraction |
| `python-dotenv` | `>=1.0.0` | `.env` file loading |
| `streamlit` | `>=1.35.0` | Web UI framework |
| `faiss-cpu` | `>=1.8.0` | In-memory vector similarity search |
| `sentence-transformers` | `>=3.0.0` | Cross-encoder re-ranking |
| `rank-bm25` | `>=0.2.2` | BM25 keyword index for hybrid search |
| `pytest` | `>=8.0.0` | Unit testing |


---

## Code Conventions

- **Flat procedural structure** — Classes: `StreamHandler`, `RerankingRetriever`, `HybridRetriever`, `FilteredRetriever`. All other logic is in module-level functions in `app.py`.
- **Full type annotations** — All functions and methods use Python 3.11 native generics (`list[str]`, `X | None`, etc.).
- **Snake_case** — All functions and variables.
- **Error handling** — `try/except` blocks surface errors as `st.error()` / `st.warning()` rather than crashing.
- **Input validation** — "Process" button validates at least one file is uploaded; empty extraction result is caught with a user-friendly message.
- **No tests for UI functions** — `handle_userinput()`, `main()`, `render_chat_history()` are tested by running the app manually.
- **Ruff** — Code style enforced by ruff (see `pyproject.toml`). Run `make lint` and `make format` before committing.

---

## Linting and Formatting

Configured in `pyproject.toml`:

- **Target**: Python 3.11
- **Line length**: 100 characters
- **Rules**: `E` (pycodestyle), `F` (pyflakes), `W` (warnings), `I` (isort)
- **Formatter**: ruff format with double quotes and space indentation

```bash
make lint     # check for issues
make format   # auto-fix formatting
```

---

## Known Limitations and Gotchas

1. **Multiple index slots replace single index** — The old `faiss_index/` directory is no longer used. Indexes are stored under `faiss_indexes/{slot}/`. The default slot is `"default"`.

2. **Cross-encoder re-ranking downloads model weights on first use** — Toggling "Cross-encoder re-ranking" in the sidebar downloads ~50 MB of model weights the first time. Subsequent uses are cached for the lifetime of the server process via `@st.cache_resource`.

3. **Semantic chunking makes embedding API calls during processing** — The semantic chunking strategy calls the active embedding provider (OpenAI or Ollama) once per document page during the "Process" step.

4. **Ollama requires a running local server** — Select the Ollama provider only when `ollama serve` is running and the required models have been pulled (`ollama pull <model>`).

5. **Hybrid search falls back to vector-only when rank-bm25 is missing** — `rank-bm25` is in `requirements.txt` by default. If removed, `HybridRetriever` silently falls back to pure FAISS similarity search.

6. **Per-document filter is only shown for multi-file indexes** — The "Filter by document" multiselect only appears when the active index contains more than one file. It resets to empty whenever new documents are processed.

7. **Cost tracker is OpenAI-only** — The tracker widget is hidden for Ollama sessions. It uses `get_openai_callback()` from `langchain_community`; actual billed amounts may differ from estimates.

8. **`text_chunks` and `chunk_metadatas` are stored in session state** — They are needed by `HybridRetriever` at query time. They are populated during the Process step and cleared when a new index is loaded from disk (disk-loaded indexes do not persist chunk lists between server restarts; hybrid search is disabled automatically in that case because `text_chunks` is empty).

9. **History sent to the LLM is truncated, but the full log is kept in session state** — `_truncate_history()` caps the context at `MAX_HISTORY_TURNS` (20) pairs before each chain invocation, preventing token-limit errors.

10. **No test suite for UI functions** — All Streamlit widget code must be manually verified by running `streamlit run app.py`.

11. **`htmlTemplates.py` is now unused** — The file is kept for reference but is no longer imported.

---

## What AI Assistants Should Know

- **The entire application is in `app.py`.** `htmlTemplates.py` is no longer used.
- **Session state is the persistence layer.** No database. Disk state: `faiss_indexes/{slot}/` (FAISS + metadata.json) and `sessions/*.json` (chat history).
- **FAISS indexes are per-slot.** `active_path = slot_path(st.session_state.active_slot)` is computed once at the top of `main()` and reused throughout.
- **Chain is rebuilt per invocation** — `get_conversation_chain()` is called inside `handle_userinput()` so a fresh `StreamHandler` can be injected. No memory object is needed; history is passed explicitly.
- **Chain invocation:** `chain.invoke({"input": user_question, "chat_history": st.session_state.chat_history})`. Response keys: `response["answer"]` (str) and `response["context"]` (list[Document]).
- **History is managed manually** — after each turn, `HumanMessage` and `AIMessage` are appended to `st.session_state.chat_history`.
- **Retriever build order:** base (HybridRetriever or FAISS) → FilteredRetriever (if `doc_filter`) → RerankingRetriever (if `reranker_enabled`).
- **BM25 index is cached** via `@st.cache_resource` in `_build_bm25()`; the cache key is `id(corpus)` (changes when `st.session_state.text_chunks` is replaced after re-processing).
- **Cost tracker dict keys:** `turns`, `prompt_tokens`, `completion_tokens`, `total_cost` (float). Updated in `handle_userinput()` inside a `get_openai_callback()` context only for the OpenAI provider.
- **Run `make lint` before committing** to ensure ruff passes cleanly.
- **Do not pin dependencies to exact versions.** Use `>=x.y.z` style.
- **`save_index_metadata` and `load_index_metadata` require `index_path`** — they no longer use a global constant. Pass the result of `slot_path(slot_name)`.
- **API keys are required.** The app will fail immediately without a valid `OPENAI_API_KEY` set in `.env` or the sidebar.
