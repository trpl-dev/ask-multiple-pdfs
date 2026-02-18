# CLAUDE.md — AI Assistant Reference for ask-multiple-pdfs

## Project Overview

**MultiPDF Chat App** is a single-script Python web application that allows users to upload multiple PDF files and ask natural language questions about their contents via a conversational interface. It implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI, FAISS, and Streamlit.

This is an educational project accompanying a YouTube tutorial. It is not intended for production use, and pull requests are not accepted upstream.

---

## Repository Structure

```
ask-multiple-pdfs/
├── app.py                # Sole application file — all business logic and Streamlit UI
├── htmlTemplates.py      # HTML/CSS string templates for chat bubble rendering
├── requirements.txt      # Python dependencies (unpinned ranges, modern versions)
├── pyproject.toml        # Ruff linter/formatter configuration
├── Makefile              # Convenience targets: run, install, lint, format
├── .env.example          # Template for required environment variables
├── .python-version       # pyenv Python version pin (3.11)
├── .gitignore            # Standard Python gitignore (excludes .env, __pycache__, etc.)
├── readme.md             # User-facing documentation
└── docs/
    └── PDF-LangChain.jpg # Architecture diagram referenced in readme.md
```

There is no package structure, no test suite, and no CI/CD configuration.

---

## Architecture: RAG Pipeline

The application follows a five-step Retrieval-Augmented Generation pipeline, all implemented in `app.py`:

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store → Conversational Chain → Answer
```

| Step | Function | Implementation |
|---|---|---|
| Text extraction | `get_pdf_text()` | `pypdf.PdfReader` iterates pages |
| Chunking | `get_text_chunks()` | `CharacterTextSplitter` (size=1000, overlap=200, separator=`\n`) |
| Embedding | `get_vectorstore()` | `OpenAIEmbeddings` (default) or `HuggingFaceInstructEmbeddings` |
| Vector store | `get_vectorstore()` | `FAISS.from_texts()` — saved to `faiss_index/` on disk; auto-loaded on startup |
| Conversational chain | `get_conversation_chain()` | `ConversationalRetrievalChain` + `ConversationBufferMemory` |
| UI rendering | `handle_userinput()`, `main()` | Streamlit + raw HTML via `unsafe_allow_html=True` |

---

## Key Files

### `app.py` (134 lines)

The entire application. Five functions plus `main()`:

- **`get_pdf_text(pdf_docs)`** — Takes a list of Streamlit file upload objects; returns concatenated text from all PDF pages. Per-file errors are caught and shown as `st.warning()` rather than crashing.
- **`get_text_chunks(text)`** — Splits raw text into 1,000-character overlapping chunks.
- **`get_vectorstore(text_chunks)`** — Creates OpenAI embeddings and builds a FAISS vector store.
- **`load_vectorstore()`** — Loads a previously saved FAISS index from `faiss_index/` on disk. Returns `None` if the path doesn't exist or loading fails.
- **`get_conversation_chain(vectorstore)`** — Builds a `ConversationalRetrievalChain` with `ConversationBufferMemory` (key: `chat_history`).
- **`handle_userinput(user_question)`** — Guards against no processed documents; invokes the chain via `.invoke()`; stores response in `st.session_state.chat_history`; renders alternating user/bot messages. Errors shown via `st.error()`.
- **`main()`** — Streamlit entry point: loads `.env`, configures page, initializes session state, renders sidebar file uploader (PDF-only) and chat input. "Process" button validates that files are selected before processing. Processing errors are caught and shown via `st.error()`.

### `htmlTemplates.py` (61 lines)

Three module-level string variables imported into `app.py`:

- **`css`** — `<style>` block for `.chat-message`, `.avatar`, `.message` CSS classes. Dark-themed (`#2b313e` user, `#475063` bot).
- **`bot_template`** — HTML div with bot avatar (green SVG data URI) and `{{MSG}}` placeholder.
- **`user_template`** — HTML div with user avatar (blue SVG data URI) and `{{MSG}}` placeholder.

Avatars are self-contained base64 SVG data URIs stored in `_BOT_AVATAR` and `_USER_AVATAR` module constants. No external image hosts are used. Templates are f-strings; `{{{{MSG}}}}` in source becomes `{{MSG}}` in the rendered string, which `app.py` replaces via `.replace("{{MSG}}", message.content)`.

---

## Environment Setup

### Prerequisites

- Python 3.11 (pinned via `.python-version` for pyenv)
- An OpenAI API key (required for default embeddings and LLM)
- Optionally: a HuggingFace Hub API token (for open-source model alternatives)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ask-multiple-pdfs

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# or: pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=<your-key>
```

### Environment Variables (`.env`)

```
OPENAI_API_KEY=           # Required for OpenAIEmbeddings and ChatOpenAI
HUGGINGFACEHUB_API_TOKEN= # Required only if using HuggingFace alternatives
```

The `.env` file is loaded via `load_dotenv()` in `main()`. It is excluded from git via `.gitignore`. **Never commit API keys.**

---

## Running the Application

```bash
make run
# or: streamlit run app.py
```

This starts a local Streamlit server (default: http://localhost:8501).

---

## Development Commands (Makefile)

| Command | Description |
|---|---|
| `make run` | Start the Streamlit app |
| `make install` | Install all dependencies from `requirements.txt` |
| `make lint` | Run ruff linter |
| `make format` | Run ruff formatter |

---

## Dependencies

Dependencies use unpinned version ranges to allow `pip` to resolve compatible modern versions.

| Package | Version Range | Purpose |
|---|---|---|
| `langchain` | `>=0.2.0,<0.3.0` | RAG chain, memory, base classes |
| `langchain-openai` | `>=0.1.0` | OpenAI embeddings and chat models |
| `langchain-community` | `>=0.2.0` | FAISS vector store, HuggingFace integrations |
| `langchain-text-splitters` | `>=0.2.0` | `CharacterTextSplitter` |
| `pypdf` | `>=4.0.0` | PDF text extraction (successor to deprecated `PyPDF2`) |
| `python-dotenv` | `>=1.0.0` | `.env` file loading |
| `streamlit` | `>=1.35.0` | Web UI framework |
| `faiss-cpu` | `>=1.8.0` | In-memory vector similarity search |

**Commented-out optional dependencies** (in `requirements.txt` and `app.py`):
- `huggingface-hub>=0.20.0` — HuggingFace Hub access
- `InstructorEmbedding>=1.0.1` — Instructor embedding model
- `sentence-transformers>=2.2.2` — Sentence transformer backbone

### Key API Notes

- Uses **LangChain 0.2.x** module structure: `langchain_openai`, `langchain_community`, `langchain_text_splitters` as separate packages.
- Uses **OpenAI Python SDK 1.x** (via LangChain's wrapper) — the new `client`-based API style.
- Chain invocation uses `.invoke({"question": ...})` not the legacy dict-call style.

---

## Alternative Backends (Commented Out)

Two alternative LLM/embedding backends are available as commented code in `app.py`:

**Embeddings** (`get_vectorstore()`):
```python
# Default (requires OPENAI_API_KEY):
embeddings = OpenAIEmbeddings()

# Alternative (requires sentence-transformers + InstructorEmbedding):
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
```

**LLM** (`get_conversation_chain()`):
```python
# Default (requires OPENAI_API_KEY):
llm = ChatOpenAI()

# Alternative (requires HUGGINGFACEHUB_API_TOKEN + huggingface-hub):
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
```

To switch backends: uncomment the alternative import at the top of `app.py`, uncomment the alternative instantiation inside the function, comment out the default, and ensure the corresponding packages are installed (uncomment them in `requirements.txt` too).

---

## Streamlit Session State

The app stores two keys in `st.session_state`:

| Key | Type | Description |
|---|---|---|
| `conversation` | `ConversationalRetrievalChain` or `None` | The active LangChain chain; `None` until "Process" is clicked successfully |
| `chat_history` | `list[BaseMessage]` or `None` | Full conversation history; alternating HumanMessage/AIMessage objects |

State persists across Streamlit reruns (user interactions) but is reset on page reload or server restart. The FAISS vector store is not persisted to disk.

---

## Code Conventions

- **Flat procedural structure** — No classes, no packages. All logic in `app.py`.
- **Snake_case** — All functions and variables use standard Python snake_case.
- **No type annotations** — No type hints anywhere in the codebase.
- **Error handling** — `try/except` blocks in `get_pdf_text()`, `handle_userinput()`, and the "Process" button handler. Errors surface as `st.error()` / `st.warning()` rather than crashing.
- **Input validation** — "Process" button validates at least one file is uploaded; file uploader restricts to PDF MIME type; empty extraction result is caught with a user-friendly message.
- **No tests** — No test infrastructure exists.
- **Raw HTML injection** — `st.write(..., unsafe_allow_html=True)` is used deliberately for custom chat UI styling.
- **String templating** — `{{MSG}}` replaced via Python `.replace()` — not Jinja2 or any engine.
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

1. **Persisted vector store** — After "Process" is clicked, the FAISS index is saved to `faiss_index/` (gitignored) and auto-loaded on the next server start. Clicking "Clear saved index" wipes it. The index is rebuilt from scratch each time "Process" is clicked.

2. **No chunking of conversation history** — `ConversationBufferMemory` stores the full conversation. For very long sessions, token limits may eventually be exceeded.

3. **No test suite** — All changes must be manually verified by running `streamlit run app.py`.

4. **Single-module design** — All logic is in one file. Any substantial feature addition should consider splitting into focused modules (e.g., `pdf_processing.py`, `vector_store.py`, `chain.py`).

5. **LangChain 0.2 boundary** — The `requirements.txt` pins `langchain<0.3.0`. LangChain 0.3 dropped some legacy APIs (including `ConversationalRetrievalChain` in favor of LCEL). Upgrading past 0.2 requires significant refactoring.

---

## What AI Assistants Should Know

- **The entire application is in `app.py`.** There are no other Python source files to search.
- **Session state is the persistence layer.** There is no database, no file system state (beyond what Streamlit manages), and no external service besides OpenAI (or HuggingFace).
- **`htmlTemplates.py` is UI-only.** Changes there affect only visual appearance, not functionality. Avatar images are self-contained data URIs — do not replace them with external URLs.
- **Use `.invoke()` not dict-call.** LangChain 0.2+ deprecates calling the chain like `chain({"question": ...})`; use `chain.invoke({"question": ...})`.
- **No tests exist.** Any changes must be manually verified by running `streamlit run app.py`.
- **API keys are required.** The app will fail immediately without a valid `OPENAI_API_KEY` set in `.env` (or environment).
- **Run `make lint` before committing** to ensure ruff passes cleanly.
- **Do not pin dependencies to exact versions.** The project moved from pinned to range-based versions intentionally. Use `>=x.y.z` style.
