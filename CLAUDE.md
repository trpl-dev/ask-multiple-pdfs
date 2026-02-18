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
├── requirements.txt      # Pinned Python dependencies
├── .env.example          # Template for required environment variables
├── .python-version       # pyenv Python version pin (3.9)
├── .gitignore            # Standard Python gitignore (excludes .env, __pycache__, etc.)
├── readme.md             # User-facing documentation
└── docs/
    └── PDF-LangChain.jpg # Architecture diagram referenced in readme.md
```

There is no package structure, no test suite, no CI/CD, and no additional modules.

---

## Architecture: RAG Pipeline

The application follows a five-step Retrieval-Augmented Generation pipeline, all implemented in `app.py`:

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store → Conversational Chain → Answer
```

| Step | Function | Implementation |
|---|---|---|
| Text extraction | `get_pdf_text()` | `PyPDF2.PdfReader` iterates pages |
| Chunking | `get_text_chunks()` | `CharacterTextSplitter` (size=1000, overlap=200, separator=`\n`) |
| Embedding | `get_vectorstore()` | `OpenAIEmbeddings` (default) or `HuggingFaceInstructEmbeddings` |
| Vector store | `get_vectorstore()` | `FAISS.from_texts()` — in-memory only, not persisted |
| Conversational chain | `get_conversation_chain()` | `ConversationalRetrievalChain` + `ConversationBufferMemory` |
| UI rendering | `handle_userinput()`, `main()` | Streamlit + raw HTML via `unsafe_allow_html=True` |

---

## Key Files

### `app.py` (105 lines)

The entire application. Five functions plus `main()`:

- **`get_pdf_text(pdf_docs)`** — Takes a list of Streamlit file upload objects; returns concatenated text from all PDF pages.
- **`get_text_chunks(text)`** — Splits raw text into 1,000-character overlapping chunks.
- **`get_vectorstore(text_chunks)`** — Creates OpenAI embeddings and builds a FAISS in-memory vector store.
- **`get_conversation_chain(vectorstore)`** — Builds a `ConversationalRetrievalChain` with `ConversationBufferMemory` (key: `chat_history`).
- **`handle_userinput(user_question)`** — Invokes the chain, stores response in `st.session_state.chat_history`, renders alternating user/bot messages.
- **`main()`** — Streamlit entry point: loads `.env`, configures page, initializes session state, renders sidebar file uploader and chat input.

### `htmlTemplates.py` (44 lines)

Three module-level string variables imported into `app.py`:

- **`css`** — `<style>` block for `.chat-message`, `.avatar`, `.message` CSS classes. Dark-themed (`#2b313e` user, `#475063` bot).
- **`bot_template`** — HTML div with bot avatar (imgbb URL) and `{{MSG}}` placeholder.
- **`user_template`** — HTML div with user avatar (imgbb URL) and `{{MSG}}` placeholder.

Message content is injected via `.replace("{{MSG}}", message.content)` — no templating library.

---

## Environment Setup

### Prerequisites

- Python 3.9 (pinned via `.python-version` for pyenv)
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
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=<your-key>
```

### Environment Variables (`.env`)

```
OPENAI_API_KEY=          # Required for OpenAIEmbeddings and ChatOpenAI
HUGGINGFACEHUB_API_TOKEN= # Required only if using HuggingFace alternatives
```

The `.env` file is loaded via `load_dotenv()` in `main()`. It is excluded from git via `.gitignore`. **Never commit API keys.**

---

## Running the Application

```bash
streamlit run app.py
```

This starts a local Streamlit server (default: http://localhost:8501).

There is no `Makefile`, `docker-compose.yml`, or other runner. The only supported way to run the app is via the `streamlit` CLI.

---

## Dependencies

All dependencies are pinned to specific versions from mid-2023. These are old and may have security advisories or compatibility issues with newer Python tooling.

| Package | Version | Purpose |
|---|---|---|
| `langchain` | 0.0.184 | RAG chain, memory, text splitter, embeddings, LLM wrappers |
| `PyPDF2` | 3.0.1 | PDF text extraction |
| `python-dotenv` | 1.0.0 | `.env` file loading |
| `streamlit` | 1.18.1 | Web UI framework |
| `openai` | 0.27.6 | OpenAI API client (legacy pre-1.0 API style) |
| `faiss-cpu` | 1.7.4 | In-memory vector similarity search |

**Commented-out optional dependencies** (in `requirements.txt` and `app.py`):
- `huggingface-hub==0.14.1` — HuggingFace Hub access
- `InstructorEmbedding==1.0.1` — Instructor embedding model
- `sentence-transformers==2.2.2` — Sentence transformer backbone

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
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
```

To switch backends: uncomment the alternative, comment the default, and ensure the corresponding packages are installed.

---

## Streamlit Session State

The app stores two keys in `st.session_state`:

| Key | Type | Description |
|---|---|---|
| `conversation` | `ConversationalRetrievalChain` or `None` | The active LangChain chain; initialized after "Process" is clicked |
| `chat_history` | `list[BaseMessage]` or `None` | Full conversation history; alternating HumanMessage/AIMessage objects |

State persists across Streamlit reruns (user interactions) but is reset on page reload or server restart. The FAISS vector store is not persisted to disk.

---

## Code Conventions

- **Flat procedural structure** — No classes, no packages. All logic in `app.py`.
- **Snake_case** — All functions and variables use standard Python snake_case.
- **No type annotations** — No type hints anywhere in the codebase.
- **No error handling** — No `try/except` blocks. API failures or malformed PDFs will raise unhandled exceptions and crash the Streamlit app.
- **No tests** — No test infrastructure exists.
- **Raw HTML injection** — `st.write(..., unsafe_allow_html=True)` is used deliberately for custom chat UI styling.
- **String templating** — `{{MSG}}` replaced via Python `.replace()` — not Jinja2 or any engine.

---

## Known Limitations and Gotchas

1. **Old dependency versions** — `langchain==0.0.184` and `openai==0.27.6` use APIs that have since been deprecated. Do not upgrade without significant refactoring (LangChain's public API changed substantially after 0.1.0; OpenAI's Python SDK changed in 1.0.0).

2. **In-memory vector store** — FAISS is built fresh every time "Process" is clicked. Nothing is persisted. Re-uploading PDFs rebuilds from scratch.

3. **No chunking of conversation history** — `ConversationBufferMemory` stores the full conversation. For long sessions, token limits may be exceeded.

4. **No input validation** — No checks on file type, file size, empty uploads, or empty questions.

5. **External avatar images** — Chat bubble avatars are loaded from imgbb URLs. If those URLs become unavailable, avatars will break silently.

6. **Single-module design** — All logic is in one file. Any non-trivial feature addition should consider refactoring into modules.

---

## Development Workflow

There is no formal development workflow defined. For any modifications:

1. Work in a virtual environment with `requirements.txt` installed.
2. Run `streamlit run app.py` to test changes interactively.
3. There are no tests to run and no linter configuration.

If adding tests or linting, recommended tools for this stack:
- `pytest` for unit tests
- `ruff` or `flake8` for linting
- `black` for formatting

---

## What AI Assistants Should Know

- **Do not upgrade dependencies casually.** The pinned versions are intentional for tutorial reproducibility. Upgrading `langchain` or `openai` requires significant code changes due to breaking API changes.
- **The entire application is in `app.py`.** There are no other Python source files to search.
- **Session state is the persistence layer.** There is no database, no file system state (beyond what Streamlit manages), and no external service besides OpenAI (or HuggingFace).
- **`htmlTemplates.py` is UI-only.** Changes there affect only visual appearance, not functionality.
- **No tests exist.** Any changes must be manually verified by running `streamlit run app.py`.
- **API keys are required.** The app will fail immediately without a valid `OPENAI_API_KEY` set in `.env` (or environment).
