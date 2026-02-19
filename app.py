import json
import os
import shutil
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader

# Uncomment for HuggingFace alternatives:
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings

FAISS_INDEX_PATH = "faiss_index"
FAISS_METADATA_PATH = os.path.join(FAISS_INDEX_PATH, "metadata.json")
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
CHUNK_STRATEGY_CHAR = "Character (fast)"
CHUNK_STRATEGY_SEMANTIC = "Semantic (accurate)"
CHUNK_STRATEGIES = [CHUNK_STRATEGY_CHAR, CHUNK_STRATEGY_SEMANTIC]
RETRIEVAL_MODES = ["Similarity", "MMR"]
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RETRIEVAL_K = 4


def save_index_metadata(filenames, chunk_count):
    """Persist index provenance to metadata.json inside the FAISS directory."""
    meta = {
        "files": filenames,
        "chunks": chunk_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_index_metadata():
    """Return the saved index metadata dict, or None if unavailable."""
    if not os.path.exists(FAISS_METADATA_PATH):
        return None
    try:
        with open(FAISS_METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_api_key():
    """Return the active OpenAI API key.

    Priority: key entered in the sidebar UI > OPENAI_API_KEY env var.
    Returns None if neither is set so callers can show a user-friendly warning.
    """
    ui_key = st.session_state.get("api_key_input", "").strip()
    if ui_key:
        return ui_key
    return os.environ.get("OPENAI_API_KEY") or None


# ---------------------------------------------------------------------------
# Streaming callback — writes LLM tokens into a Streamlit placeholder
# ---------------------------------------------------------------------------


class StreamHandler(BaseCallbackHandler):
    """Streams LLM tokens into a st.empty() placeholder as they arrive."""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        # Show a blinking-cursor effect while streaming
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args, **kwargs):
        # Remove cursor when the LLM finishes
        self.container.markdown(self.text)


# ---------------------------------------------------------------------------
# RAG pipeline helpers
# ---------------------------------------------------------------------------


def get_pdf_text(pdf_docs):
    """Extract text from each PDF; returns list of (text, filename) tuples."""
    results = []
    for pdf in pdf_docs:
        text = ""
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Could not read '{pdf.name}': {e}")
        if text.strip():
            results.append((text, pdf.name))
    return results


def get_text_chunks(
    texts_with_meta,
    strategy=CHUNK_STRATEGY_CHAR,
    chunk_size=1000,
    chunk_overlap=200,
    semantic_threshold=95,
    api_key=None,
):
    """Split each document's text into chunks; returns (chunks, metadatas).

    Args:
        texts_with_meta: list of (text, filename) tuples from get_pdf_text()
        strategy: CHUNK_STRATEGY_CHAR or CHUNK_STRATEGY_SEMANTIC
        chunk_size: Character strategy — target character count per chunk
        chunk_overlap: Character strategy — overlapping characters between chunks
        semantic_threshold: Semantic strategy — percentile threshold (0–100)
            for splitting; lower → more (smaller) chunks
        api_key: OpenAI API key forwarded to SemanticChunker embeddings

    Returns:
        Tuple of (list[str], list[dict]) — the text chunks and their FAISS
        metadata (each dict carries a "source" key with the originating filename).
    """
    if strategy == CHUNK_STRATEGY_SEMANTIC:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
        except ImportError:
            raise ImportError(
                "Semantic chunking requires 'langchain-experimental'. "
                "Uncomment it in requirements.txt and run `make install`."
            )
        kwargs = {"api_key": api_key} if api_key else {}
        splitter = SemanticChunker(
            OpenAIEmbeddings(**kwargs),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=semantic_threshold,
        )
        all_docs = splitter.create_documents(
            [text for text, _ in texts_with_meta],
            metadatas=[{"source": name} for _, name in texts_with_meta],
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
    for text, filename in texts_with_meta:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        all_meta.extend([{"source": filename}] * len(chunks))
    return all_chunks, all_meta


def get_vectorstore(text_chunks, metadatas=None, api_key=None):
    kwargs = {"api_key": api_key} if api_key else {}
    embeddings = OpenAIEmbeddings(**kwargs)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(
        texts=text_chunks, embedding=embeddings, metadatas=metadatas or []
    )
    return vectorstore


def load_vectorstore(api_key=None):
    """Load a previously saved FAISS index from disk. Returns None on failure."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    try:
        kwargs = {"api_key": api_key} if api_key else {}
        embeddings = OpenAIEmbeddings(**kwargs)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Could not load saved index: {e}")
        return None


def get_conversation_chain(
    vectorstore,
    memory,
    model=DEFAULT_MODEL,
    stream_handler=None,
    api_key=None,
    temperature=DEFAULT_TEMPERATURE,
    retrieval_k=DEFAULT_RETRIEVAL_K,
    retrieval_mode="Similarity",
):
    """Build a ConversationalRetrievalChain.

    Uses a non-streaming LLM for question condensation and a streaming-enabled
    LLM (with the given StreamHandler) for the final answer generation so that
    only answer tokens appear in the UI placeholder.
    """
    kwargs = {"api_key": api_key} if api_key else {}
    callbacks = [stream_handler] if stream_handler else []
    answer_llm = ChatOpenAI(
        model=model, temperature=temperature, streaming=True, callbacks=callbacks, **kwargs
    )
    condense_llm = ChatOpenAI(model=model, temperature=0, **kwargs)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

    if retrieval_mode == "MMR":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": retrieval_k, "fetch_k": max(retrieval_k * 3, 20)},
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=answer_llm,
        condense_question_llm=condense_llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return conversation_chain


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_sources(sources):
    """Render source documents inside an expander below a bot message."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        seen = set()
        for doc in sources:
            filename = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:250].replace("\n", " ").strip()
            key = (filename, preview[:50])
            if key in seen:
                continue
            seen.add(key)
            st.markdown(f"**{filename}**  \n{preview}…")


def format_conversation_as_markdown(chat_history, sources):
    """Serialize the conversation to a Markdown string for download."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
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


def render_chat_history():
    """Render all previous turns from session state using st.chat_message()."""
    history = st.session_state.chat_history or []
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
                bot_turn_idx += 1


def handle_userinput(user_question):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process your PDF documents first.")
        return

    # Show the new user message
    with st.chat_message("user"):
        st.write(user_question)

    # Stream the bot response into a placeholder inside the chat bubble
    with st.chat_message("assistant"):
        stream_container = st.empty()
        stream_handler = StreamHandler(stream_container)

        try:
            chain = get_conversation_chain(
                st.session_state.vectorstore,
                st.session_state.memory,
                st.session_state.model,
                stream_handler,
                api_key=get_api_key(),
                temperature=st.session_state.temperature,
                retrieval_k=st.session_state.retrieval_k,
                retrieval_mode=st.session_state.retrieval_mode,
            )
            response = chain.invoke({"question": user_question})
            new_sources = response.get("source_documents", [])
            st.session_state.chat_history = response["chat_history"]
            st.session_state.sources.append(new_sources)
        except Exception as e:
            stream_container.empty()
            st.error(f"Error getting response: {e}")
            return

        # Replace streaming placeholder with final answer
        bot_answer = st.session_state.chat_history[-1].content
        stream_container.markdown(bot_answer)
        _render_sources(new_sources)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    # --- Session state initialisation ---
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sources" not in st.session_state:
        st.session_state.sources = []
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

    # Auto-load a previously saved FAISS index so users don't have to re-upload
    # PDFs after a page reload or server restart.
    if st.session_state.vectorstore is None and os.path.exists(FAISS_INDEX_PATH):
        vs = load_vectorstore(api_key=get_api_key())
        if vs is not None:
            st.session_state.vectorstore = vs
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

    st.header("Chat with multiple PDFs :books:")
    render_chat_history()
    if user_question := st.chat_input("Ask a question about your documents:"):
        handle_userinput(user_question)

    # --- Sidebar ---
    with st.sidebar:
        st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-… (leave blank to use .env)",
            key="api_key_input",
        )
        if not get_api_key():
            st.warning("No API key found. Enter one above or set OPENAI_API_KEY in your .env file.")

        st.divider()
        selected_model = st.selectbox(
            "OpenAI model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.model),
        )
        if selected_model != st.session_state.model:
            st.session_state.model = selected_model
            st.session_state.memory = None
            st.session_state.chat_history = []
            st.session_state.sources = []

        with st.expander("LLM & Retrieval", expanded=False):
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
                st.session_state.chat_history = []
                st.session_state.sources = []
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )
                st.rerun()

        st.subheader("Your documents")
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
                st.caption(
                    ":warning: Semantic chunking calls the OpenAI Embeddings API "
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
            else:
                with st.spinner("Processing"):
                    try:
                        texts_with_meta = get_pdf_text(pdf_docs)
                        if not texts_with_meta:
                            st.error(
                                "No text could be extracted from the uploaded PDFs. "
                                "Ensure the files are not scanned images or password-protected."
                            )
                        else:
                            text_chunks, metadatas = get_text_chunks(
                                texts_with_meta,
                                strategy=st.session_state.chunk_strategy,
                                chunk_size=st.session_state.chunk_size,
                                chunk_overlap=st.session_state.chunk_overlap,
                                semantic_threshold=st.session_state.semantic_threshold,
                                api_key=get_api_key(),
                            )
                            vectorstore = get_vectorstore(text_chunks, metadatas, api_key=get_api_key())
                            vectorstore.save_local(FAISS_INDEX_PATH)
                            save_index_metadata(
                                filenames=[t[1] for t in texts_with_meta],
                                chunk_count=len(text_chunks),
                            )
                            st.session_state.vectorstore = vectorstore
                            st.session_state.memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )
                            st.session_state.chat_history = []
                            st.session_state.sources = []
                            st.success("Documents processed and index saved!")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")

        if os.path.exists(FAISS_INDEX_PATH):
            st.divider()
            meta = load_index_metadata()
            if meta:
                ts = datetime.fromisoformat(meta["timestamp"])
                age_min = int((datetime.now(timezone.utc) - ts).total_seconds() / 60)
                age_str = f"{age_min} min ago" if age_min < 60 else f"{age_min // 60} h ago"
                st.caption(
                    f"**Index loaded** · {meta['chunks']} chunks · {age_str}  \n"
                    + ", ".join(meta["files"])
                )
            if st.button("Clear saved index"):
                shutil.rmtree(FAISS_INDEX_PATH)
                st.session_state.vectorstore = None
                st.session_state.memory = None
                st.session_state.chat_history = []
                st.session_state.sources = []
                st.rerun()


if __name__ == "__main__":
    main()
