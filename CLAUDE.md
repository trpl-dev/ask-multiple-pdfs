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
    ├── test_chunking.py  # 7 tests for get_text_chunks()
    ├── test_pdf.py       # 5 tests for get_pdf_text()
    └── test_metadata.py  # 5 tests for save/load_index_metadata()
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
| Embedding | `get_vectorstore()` | `OpenAIEmbeddings` (default) or `HuggingFaceInstructEmbeddings` |
| Vector store | `get_vectorstore()` | `FAISS.from_texts()` — saved to `faiss_indexes/{slot}/` |
| Retrieval | `get_conversation_chain()` | Similarity or MMR retriever; optionally wrapped by `RerankingRetriever` |
| Conversational chain | `get_conversation_chain()` | LCEL: `create_history_aware_retriever` + `create_retrieval_chain`; chat history passed explicitly |
| Streaming | `StreamHandler` | `BaseCallbackHandler` subclass; writes tokens into `st.empty()` placeholder |
| UI rendering | `handle_userinput()`, `render_chat_history()`, `main()` | Native `st.chat_message()` / `st.chat_input()` |

---

## Key Files

### `app.py` (~620 lines)

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
| `RETRIEVAL_MODES` | `["Similarity", "MMR"]` | FAISS retrieval modes |
| `DEFAULT_TEMPERATURE` | `0.0` | LLM temperature |
| `DEFAULT_RETRIEVAL_K` | `4` | Default number of retrieved chunks |
| `RERANKER_MODEL` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model name |

**Index slot helpers**
- **`slot_path(slot)`** — returns `faiss_indexes/{slot}`
- **`list_index_slots()`** — returns sorted list of existing slot directory names
- **`save_index_metadata(filenames, chunk_count, index_path)`** — writes `metadata.json` into the slot dir
- **`load_index_metadata(index_path)`** — reads `metadata.json`; returns `None` if missing or corrupt

**Session persistence helpers** (serialize/deserialize `BaseMessage` and `Document` objects to/from JSON)
- `_serialize_messages()` / `_deserialize_messages()`
- `_serialize_sources()` / `_deserialize_sources()`
- **`list_sessions()`**, **`save_session()`**, **`load_session()`**, **`delete_session()`**

**`get_api_key()`** — UI input priority over `OPENAI_API_KEY` env var; returns `None` if neither set.

**`StreamHandler(BaseCallbackHandler)`** — appends `▌` cursor on each token; removes it on LLM end. Uses `st.empty()` placeholder passed at construction time.

**`get_pdf_text(pdf_docs)`** → `list[tuple[str, str]]` — returns `(text, filename)` tuples; skips files with no extractable text.

**`get_text_chunks(texts_with_meta, strategy, chunk_size, chunk_overlap, semantic_threshold, api_key)`** → `tuple[list[str], list[dict]]` — supports:
- `CHUNK_STRATEGY_CHAR`: `CharacterTextSplitter(separator="\n", ...)`
- `CHUNK_STRATEGY_SEMANTIC`: `SemanticChunker` from `langchain-experimental` (lazy import)

**`get_vectorstore(text_chunks, metadatas, api_key)`** → `FAISS`

**`load_vectorstore(index_path, api_key)`** → `FAISS | None` — loads from `index_path`; shows `st.warning()` on failure.

**`get_conversation_chain(vectorstore, model, stream_handler, api_key, temperature, retrieval_k, retrieval_mode, system_prompt, reranker_enabled)`** → `Runnable`:
- LCEL pipeline: `create_history_aware_retriever` (question condensing) → `create_stuff_documents_chain` (QA) → `create_retrieval_chain`
- Separate non-streaming `condense_llm` and streaming `answer_llm`
- `ChatPromptTemplate` with `MessagesPlaceholder` for history; optional system prompt prefix
- Retriever wrapping: `RerankingRetriever` when `reranker_enabled=True`, else plain FAISS retriever
- No `memory` parameter — chat history passed explicitly via `chain.invoke({"input": ..., "chat_history": ...})`
- Response keys: `response["answer"]` (str) and `response["context"]` (list[Document])

**`RerankingRetriever(BaseRetriever)`** — Pydantic model wrapping a base retriever. Calls `base_retriever.invoke(query)`, scores pairs with `CrossEncoder`, returns top `k`. Falls back to original order on `ImportError` or any exception.

**`generate_suggested_questions(text_chunks, api_key, model)`** → `list[str]` — LLM call (temperature 0.3) on first 3000 chars of combined chunks; returns up to 5 questions.

**`_render_sources(sources)`** — expander with deduped filename + 250-char preview per source document.

**`format_conversation_as_markdown(chat_history, sources)`** → `str` — Markdown export for download.

**`render_chat_history()`** — replays all `st.session_state.chat_history` turns using `st.chat_message()`; shows sources below each assistant turn.

**`handle_userinput(user_question)`** — guards against missing vectorstore; renders user bubble; streams answer into assistant bubble via fresh `StreamHandler`; appends sources to `st.session_state.sources`.

**`main()`** — Streamlit entry point: `load_dotenv()`, session state init, auto-load of active slot's FAISS index, chat area (history + suggested questions + `st.chat_input()`), full sidebar.

### `htmlTemplates.py`

No longer imported by `app.py` (replaced by native `st.chat_message()` in Step 2). Kept for reference. Contains `css`, `bot_template`, `user_template` with self-contained SVG avatar data URIs.

### `tests/`

| File | Tests | What it covers |
|---|---|---|
| `test_chunking.py` | 7 | `get_text_chunks()`: parallel list lengths, metadata keys, multi-file attribution, chunk size bounds, relative chunk count, empty input, `ValueError` on `chunk_overlap > chunk_size` |
| `test_pdf.py` | 5 | `get_pdf_text()`: single/multiple PDFs, no-text excluded, broken PDF warning, mixed good/broken |
| `test_metadata.py` | 5 | `save_index_metadata()` / `load_index_metadata()`: roundtrip, missing file → None, corrupt JSON → None, overwrite, UTC ISO timestamp |

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
| `chunk_strategy` | `str` | `CHUNK_STRATEGY_CHAR` or `CHUNK_STRATEGY_SEMANTIC` |
| `chunk_size` | `int` | Character splitter: chars per chunk |
| `chunk_overlap` | `int` | Character splitter: overlapping chars |
| `semantic_threshold` | `int` | Semantic splitter: breakpoint percentile (50–99) |
| `suggested_questions` | `list[str]` | LLM-generated one-click questions |
| `active_slot` | `str` | Active FAISS index slot name (default: `"default"`) |

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
| `make test` | Run pytest (17 unit tests) |
| `make docker-build` | Build Docker image |
| `make docker-up` | Build and start via Docker Compose (detached) |
| `make docker-down` | Stop Docker Compose services |

---

## Dependencies

| Package | Version Range | Purpose |
|---|---|---|
| `langchain` | `>=0.3.0` | LCEL chains, prompts, base classes |
| `langchain-openai` | `>=0.2.0` | OpenAI embeddings and chat models |
| `langchain-community` | `>=0.3.0` | FAISS vector store, HuggingFace integrations |
| `langchain-text-splitters` | `>=0.3.0` | `CharacterTextSplitter` |
| `pypdf` | `>=4.0.0` | PDF text extraction |
| `python-dotenv` | `>=1.0.0` | `.env` file loading |
| `streamlit` | `>=1.35.0` | Web UI framework |
| `faiss-cpu` | `>=1.8.0` | In-memory vector similarity search |
| `pytest` | `>=8.0.0` | Unit testing |

**Optional (uncomment in `requirements.txt`)**:
- `sentence-transformers>=2.2.2` — cross-encoder re-ranking + Instructor embeddings
- `InstructorEmbedding>=1.0.1` — Instructor embedding model
- `huggingface-hub>=0.20.0` — HuggingFace LLM backend
- `langchain-experimental>=0.0.60` — `SemanticChunker` for semantic chunking

---

## Code Conventions

- **Flat procedural structure** — No classes except `StreamHandler` and `RerankingRetriever`. All logic in `app.py`.
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

2. **Re-ranking requires opt-in install** — Toggle "Cross-encoder re-ranking" in the sidebar, but first run `pip install sentence-transformers`. First use downloads ~50 MB model weights. Falls back gracefully when not installed.

3. **Semantic chunking requires opt-in install** — Uncomment `langchain-experimental` in `requirements.txt` and run `make install`.

4. **No truncation of conversation history** — The full `chat_history` list is passed to every chain invocation. For very long sessions, token limits may eventually be exceeded.

5. **No test suite for UI functions** — All Streamlit widget code must be manually verified by running `streamlit run app.py`.

6. **`htmlTemplates.py` is now unused** — The file is kept for reference but is no longer imported. The chat UI uses native `st.chat_message()` bubbles.

---

## What AI Assistants Should Know

- **The entire application is in `app.py`.** `htmlTemplates.py` is no longer used.
- **Session state is the persistence layer.** No database. Disk state: `faiss_indexes/{slot}/` (FAISS + metadata.json) and `sessions/*.json` (chat history).
- **FAISS indexes are per-slot.** `active_path = slot_path(st.session_state.active_slot)` is computed once at the top of `main()` and reused throughout.
- **Chain is rebuilt per invocation** — `get_conversation_chain()` is called inside `handle_userinput()` so a fresh `StreamHandler` can be injected. No memory object is needed; history is passed explicitly.
- **Chain invocation:** `chain.invoke({"input": user_question, "chat_history": st.session_state.chat_history})`. Response keys: `response["answer"]` (str) and `response["context"]` (list[Document]).
- **History is managed manually** — after each turn, `HumanMessage` and `AIMessage` are appended to `st.session_state.chat_history`.
- **Run `make lint` before committing** to ensure ruff passes cleanly.
- **Do not pin dependencies to exact versions.** Use `>=x.y.z` style.
- **`save_index_metadata` and `load_index_metadata` require `index_path`** — they no longer use a global constant. Pass the result of `slot_path(slot_name)`.
- **API keys are required.** The app will fail immediately without a valid `OPENAI_API_KEY` set in `.env` or the sidebar.
