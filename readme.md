# MultiPDF Chat App

> Tutorial video: [YouTube](https://youtu.be/dXxQ0LR-3Hg)

## Introduction

The MultiPDF Chat App lets you upload multiple PDF documents and ask natural language questions about their content via a conversational chat interface. It is built on a Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS, and Streamlit.

Two LLM providers are supported:

- **OpenAI** (default) — cloud-based models; requires an API key.
- **Ollama** — locally-running open-source models; no API key needed, requires [Ollama](https://ollama.com) installed on your machine.

## How It Works

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

1. **PDF Loading** — Upload one or more PDFs; text is extracted page by page.
2. **Chunking** — Text is split into overlapping chunks (character-based or semantic).
3. **Embedding** — Each chunk is embedded (OpenAI Embeddings or Ollama Embeddings) and stored in a FAISS vector index.
4. **Question Condensing** — Chat history is used to rewrite follow-up questions as standalone queries, enabling accurate multi-turn conversations.
5. **Retrieval** — The condensed question is matched against the FAISS index; the most relevant chunks are retrieved (Similarity or MMR, optionally re-ranked).
6. **Answer Generation** — Retrieved chunks are passed to the selected LLM (OpenAI or Ollama); the answer streams token-by-token into the chat bubble.

## Features

| Feature | Description |
|---|---|
| Multi-PDF chat | Upload and query multiple PDFs simultaneously |
| Streaming answers | Token-level streaming with a live cursor |
| Source attribution | Expandable source snippets below each answer |
| OpenAI models | Choose between `gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4o` |
| Ollama (local) | Use any locally-running Ollama model (e.g. `llama3.2`, `mistral`) |
| System prompt | Optional instructions prepended to every QA prompt |
| Temperature & k | Tune creativity and number of retrieved chunks |
| Retrieval mode | Similarity or MMR (diversity-aware) |
| Chunking UI | Character splitter (configurable size/overlap) or Semantic splitter |
| Suggested questions | LLM-generated one-click questions after processing |
| Conversation export | Download chat history as Markdown |
| Session persistence | Save, load, and delete named chat sessions (JSON) |
| Multiple index slots | Maintain independent FAISS indexes per project/topic |
| Cross-encoder re-ranking | Opt-in reranking with `sentence-transformers` |
| Docker support | One-command startup with `docker compose up` |

## Installation

### Local

```bash
git clone <repo-url>
cd ask-multiple-pdfs

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt   # or: make install

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=<your-key>  (only needed for OpenAI provider)
```

### Docker

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=<your-key>  (only needed for OpenAI provider)

make docker-up   # or: docker compose up --build -d
```

The app is then available at **http://localhost:8501**.
The FAISS index and saved sessions are persisted in `faiss_indexes/` and `sessions/` via volume mounts.

## Usage

```bash
make run   # or: streamlit run app.py
```

### Using OpenAI

1. Select **OpenAI** in the provider radio at the top of the sidebar (default).
2. Enter your OpenAI API key in the sidebar (or set `OPENAI_API_KEY` in `.env`).
3. Upload one or more PDF files under **Your documents** and click **Process**.
4. Ask questions in the chat input.

### Using Ollama (local)

1. Install [Ollama](https://ollama.com) and start the server: `ollama serve`
2. Pull a chat model and an embedding model:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
3. Uncomment `langchain-ollama>=0.2.0` in `requirements.txt` and run `make install`.
4. Select **Ollama (local)** in the provider radio at the top of the sidebar.
5. Set the base URL (default: `http://localhost:11434`), chat model, and embedding model.
6. Upload PDFs and click **Process**.

> **Note:** The embedding model used when building the index must match the one used when loading it. If you change the embedding model, re-process your documents.

Each answer streams in real time with a collapsible **Sources** expander below.
The active index slot is shown below the page header; switch slots any time from the sidebar.

### Sidebar Options

| Section | Options |
|---|---|
| **Provider** | Switch between OpenAI and Ollama (local); clears the index and history |
| **OpenAI settings** | API key input, model selector (`gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4o`) |
| **Ollama settings** | Base URL, chat model name, embedding model name |
| **LLM & Retrieval** | System prompt, Temperature, Retrieved chunks (k), Retrieval mode, Cross-encoder re-ranking |
| **Sessions** | Save/load/delete named chat sessions (delete requires confirmation) |
| **Index slots** | Create and switch between independent FAISS indexes |
| **Chunking settings** | Character splitter (size, overlap) or Semantic splitter (percentile threshold) |
| **Your documents** | Index status indicator, PDF uploader, Process button, Clear saved index |

## Optional Dependencies

Uncomment the relevant lines in `requirements.txt` and run `make install`:

| Package | Enables |
|---|---|
| `langchain-ollama>=0.2.0` | Ollama local model server support |
| `sentence-transformers>=3.0.0` | Cross-encoder re-ranking + Instructor embeddings |
| `InstructorEmbedding>=1.0.1` | HuggingFace Instructor embeddings |
| `huggingface-hub>=0.20.0` | HuggingFace LLM backend |
| `langchain-experimental>=0.3.0` | Semantic chunking strategy |

## Development

```bash
make lint      # ruff linter
make format    # ruff formatter
make test      # pytest (37 unit tests)
```

## Contributing

This repository is intended for educational purposes and does not accept further contributions. Feel free to fork and adapt it to your own needs.

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
