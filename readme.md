# MultiPDF Chat App

> Tutorial video: [YouTube](https://youtu.be/dXxQ0LR-3Hg)

## Introduction

The MultiPDF Chat App lets you upload multiple PDF documents and ask natural language questions about their content via a conversational chat interface. It is built on a Retrieval-Augmented Generation (RAG) pipeline using LangChain, OpenAI, FAISS, and Streamlit.

## How It Works

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

1. **PDF Loading** — Upload one or more PDFs; text is extracted page by page.
2. **Chunking** — Text is split into overlapping chunks (character-based or semantic).
3. **Embedding** — Each chunk is embedded with OpenAI Embeddings and stored in a FAISS vector index.
4. **Retrieval** — On each question, the most relevant chunks are retrieved (Similarity or MMR).
5. **Answer Generation** — Retrieved chunks are passed to a ChatOpenAI model; the answer streams token-by-token into the chat bubble.

## Features

| Feature | Description |
|---|---|
| Multi-PDF chat | Upload and query multiple PDFs simultaneously |
| Streaming answers | Token-level streaming with a live cursor |
| Source attribution | Expandable source snippets below each answer |
| Model selection | Choose between `gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4o` |
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
# Edit .env and set OPENAI_API_KEY=<your-key>
```

### Docker

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=<your-key>

make docker-up   # or: docker compose up --build -d
```

The app is then available at **http://localhost:8501**.
The FAISS index and saved sessions are persisted in `faiss_indexes/` and `sessions/` via volume mounts.

## Usage

```bash
make run   # or: streamlit run app.py
```

1. Enter your OpenAI API key in the sidebar (or set `OPENAI_API_KEY` in `.env`).
2. Upload one or more PDF files and click **Process**.
3. Watch the progress bar as your documents are extracted, chunked, embedded, and indexed.
4. Use the suggested question buttons or type your own question.
5. Each answer streams in real time with a collapsible **Sources** expander below.

### Sidebar Options

| Section | Options |
|---|---|
| **LLM & Retrieval** | System prompt, Temperature, Retrieved chunks (k), Retrieval mode, Cross-encoder re-ranking |
| **Sessions** | Save/load/delete named chat sessions |
| **Index slots** | Create and switch between independent FAISS indexes |
| **Chunking settings** | Character splitter (size, overlap) or Semantic splitter (percentile threshold) |

## Optional Dependencies

Uncomment the relevant lines in `requirements.txt` and run `make install`:

| Package | Enables |
|---|---|
| `sentence-transformers>=2.2.2` | Cross-encoder re-ranking + Instructor embeddings |
| `InstructorEmbedding>=1.0.1` | HuggingFace Instructor embeddings |
| `huggingface-hub>=0.20.0` | HuggingFace LLM backend |
| `langchain-experimental>=0.0.60` | Semantic chunking strategy |

## Development

```bash
make lint      # ruff linter
make format    # ruff formatter
make test      # pytest (17 unit tests)
```

## Contributing

This repository is intended for educational purposes and does not accept further contributions. Feel free to fork and adapt it to your own needs.

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
