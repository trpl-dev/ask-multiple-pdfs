import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, BaseMessage, Document, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader

# Uncomment for HuggingFace alternatives:
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings

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


def slot_path(slot: str) -> str:
    """Return the filesystem path for the given index slot directory."""
    return os.path.join(FAISS_INDEXES_DIR, slot)


def list_index_slots() -> list[str]:
    """Return sorted list of existing index slot names."""
    if not os.path.exists(FAISS_INDEXES_DIR):
        return []
    return sorted(
        d for d in os.listdir(FAISS_INDEXES_DIR) if os.path.isdir(slot_path(d))
    )


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
    except Exception:
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
    os.makedirs(SESSIONS_DIR, exist_ok=True)
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
    path = os.path.join(SESSIONS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return _deserialize_messages(data.get("messages", [])), _deserialize_sources(
            data.get("sources", [])
        )
    except Exception:
        return None, None


def delete_session(name: str) -> None:
    """Delete a saved session file."""
    path = os.path.join(SESSIONS_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)


def get_api_key() -> str | None:
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


def get_pdf_text(pdf_docs: list[Any]) -> list[tuple[str, str]]:
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
    texts_with_meta: list[tuple[str, str]],
    strategy: str = CHUNK_STRATEGY_CHAR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    semantic_threshold: int = 95,
    api_key: str | None = None,
) -> tuple[list[str], list[dict]]:
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


def get_vectorstore(
    text_chunks: list[str], metadatas: list[dict] | None = None, api_key: str | None = None
) -> FAISS:
    kwargs = {"api_key": api_key} if api_key else {}
    embeddings = OpenAIEmbeddings(**kwargs)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(
        texts=text_chunks, embedding=embeddings, metadatas=metadatas or []
    )
    return vectorstore


def load_vectorstore(index_path: str, api_key: str | None = None) -> FAISS | None:
    """Load a previously saved FAISS index from disk. Returns None on failure."""
    if not os.path.exists(index_path):
        return None
    try:
        kwargs = {"api_key": api_key} if api_key else {}
        embeddings = OpenAIEmbeddings(**kwargs)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Could not load saved index: {e}")
        return None


def get_conversation_chain(
    vectorstore: FAISS,
    memory: ConversationBufferMemory,
    model: str = DEFAULT_MODEL,
    stream_handler: StreamHandler | None = None,
    api_key: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    retrieval_mode: str = "Similarity",
    system_prompt: str = "",
    reranker_enabled: bool = False,
) -> ConversationalRetrievalChain:
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

    fetch_k = max(retrieval_k * 4, 20)
    if retrieval_mode == "MMR":
        base_ret = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k if reranker_enabled else retrieval_k, "fetch_k": fetch_k},
        )
    else:
        base_ret = vectorstore.as_retriever(
            search_kwargs={"k": fetch_k if reranker_enabled else retrieval_k}
        )

    retriever = (
        RerankingRetriever(base_retriever=base_ret, top_k=retrieval_k, fetch_k=fetch_k)
        if reranker_enabled
        else base_ret
    )

    prefix = f"{system_prompt.strip()}\n\n" if system_prompt.strip() else ""
    qa_template = (
        f"{prefix}"
        "Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\nHelpful Answer:"
    )
    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=qa_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=answer_llm,
        condense_question_llm=condense_llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return conversation_chain


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking retriever (opt-in; requires sentence-transformers)
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever(BaseRetriever):
    """Wraps any retriever and re-ranks its candidates with a cross-encoder.

    Fetches up to ``fetch_k`` documents from the base retriever, scores them
    with a cross-encoder, and returns the ``top_k`` highest-scoring ones.
    Falls back to the original ordering when sentence-transformers is not
    installed or an error occurs.
    """

    base_retriever: Any
    top_k: int = DEFAULT_RETRIEVAL_K
    fetch_k: int = DEFAULT_RETRIEVAL_K * 4
    model_name: str = RERANKER_MODEL

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.get_relevant_documents(query)
        if len(candidates) <= 1:
            return candidates[: self.top_k]
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            encoder = CrossEncoder(self.model_name)
            scores = encoder.predict([(query, d.page_content) for d in candidates])
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[: self.top_k]]
        except Exception:
            return candidates[: self.top_k]


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------


def generate_suggested_questions(
    text_chunks: list[str], api_key: str | None = None, model: str = DEFAULT_MODEL
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
        kwargs = {"api_key": api_key} if api_key else {}
        llm = ChatOpenAI(model=model, temperature=0.3, **kwargs)
        result = llm.invoke(prompt)
        lines = [ln.strip() for ln in result.content.splitlines() if ln.strip()]
        return lines[:5]
    except Exception:
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
            preview = doc.page_content[:250].replace("\n", " ").strip()
            key = (filename, preview[:50])
            if key in seen:
                continue
            seen.add(key)
            st.markdown(f"**{filename}**  \n{preview}…")


def format_conversation_as_markdown(
    chat_history: list[BaseMessage], sources: list[list[Document]]
) -> str:
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


def render_chat_history() -> None:
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


def handle_userinput(user_question: str) -> None:
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
                system_prompt=st.session_state.system_prompt,
                reranker_enabled=st.session_state.reranker_enabled,
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


def main() -> None:
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
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    if "reranker_enabled" not in st.session_state:
        st.session_state.reranker_enabled = False
    if "active_slot" not in st.session_state:
        st.session_state.active_slot = DEFAULT_SLOT

    # Auto-load the active slot's FAISS index on startup / slot switch.
    active_path = slot_path(st.session_state.active_slot)
    if st.session_state.vectorstore is None and os.path.exists(active_path):
        vs = load_vectorstore(active_path, api_key=get_api_key())
        if vs is not None:
            st.session_state.vectorstore = vs
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

    st.header("Chat with multiple PDFs :books:")
    render_chat_history()

    # Show suggested questions as one-click buttons when the chat is still empty
    if not st.session_state.chat_history and st.session_state.suggested_questions:
        st.caption("Suggested questions — click to ask:")
        cols = st.columns(len(st.session_state.suggested_questions))
        for col, q in zip(cols, st.session_state.suggested_questions):
            if col.button(q, use_container_width=True):
                handle_userinput(q)
                st.rerun()

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
            st.session_state.suggested_questions = []

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
                    "Requires `sentence-transformers` (`pip install sentence-transformers`).  \n"
                    "First use downloads ~50 MB model weights."
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
                st.session_state.suggested_questions = []
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )
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
                    save_session(
                        new_name.strip(),
                        st.session_state.chat_history,
                        st.session_state.sources,
                    )
                    st.success(f"Saved as '{new_name.strip()}'")
                else:
                    st.warning("Nothing to save — start a conversation first.")

            if saved:
                selected_session = st.selectbox("Load session", [""] + saved)
                load_col, del_col2 = st.columns(2)
                if load_col.button("Load", use_container_width=True, disabled=not selected_session):
                    hist, srcs = load_session(selected_session)
                    if hist is not None:
                        st.session_state.chat_history = hist
                        st.session_state.sources = srcs
                        st.session_state.suggested_questions = []
                        st.session_state.memory = ConversationBufferMemory(
                            memory_key="chat_history", return_messages=True
                        )
                        # Replay history into memory so the chain has context
                        for i in range(0, len(hist) - 1, 2):
                            st.session_state.memory.save_context(
                                {"input": hist[i].content},
                                {"output": hist[i + 1].content if i + 1 < len(hist) else ""},
                            )
                        st.rerun()
                    else:
                        st.error("Could not load session.")
                if del_col2.button("Delete", use_container_width=True, disabled=not selected_session):
                    delete_session(selected_session)
                    st.rerun()
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
                st.session_state.memory = None
                st.session_state.chat_history = []
                st.session_state.sources = []
                st.session_state.suggested_questions = []
                st.rerun()

            new_slot_name = st.text_input(
                "New slot name", placeholder="e.g. project-alpha", key="new_slot_input"
            )
            cre_col, del_col = st.columns(2)
            if cre_col.button("Create", use_container_width=True, disabled=not new_slot_name):
                name = new_slot_name.strip()
                if name:
                    os.makedirs(slot_path(name), exist_ok=True)
                    st.session_state.active_slot = name
                    st.session_state.vectorstore = None
                    st.session_state.memory = None
                    st.session_state.chat_history = []
                    st.session_state.sources = []
                    st.session_state.suggested_questions = []
                    st.rerun()
            if del_col.button(
                "Delete slot",
                use_container_width=True,
                disabled=chosen_slot == DEFAULT_SLOT,
                help="Cannot delete the default slot.",
            ):
                shutil.rmtree(slot_path(chosen_slot), ignore_errors=True)
                st.session_state.active_slot = DEFAULT_SLOT
                st.session_state.vectorstore = None
                st.session_state.memory = None
                st.session_state.chat_history = []
                st.session_state.sources = []
                st.session_state.suggested_questions = []
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
                try:
                    with st.status("Processing documents…", expanded=True) as proc_status:
                        # Step 1 — text extraction (per-file)
                        st.write(f"Extracting text from {len(pdf_docs)} PDF(s)…")
                        prog = st.progress(0.0)
                        texts_with_meta = []
                        for idx, pdf in enumerate(pdf_docs):
                            partial = get_pdf_text([pdf])
                            texts_with_meta.extend(partial)
                            prog.progress((idx + 1) / len(pdf_docs) * 0.35)

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
                            )

                            # Step 3 — embedding + FAISS
                            st.write(f"Embedding {len(text_chunks)} chunks and building index…")
                            prog.progress(0.55)
                            vectorstore = get_vectorstore(text_chunks, metadatas, api_key=get_api_key())

                            # Step 4 — persist
                            st.write("Saving index to disk…")
                            prog.progress(0.9)
                            os.makedirs(active_path, exist_ok=True)
                            vectorstore.save_local(active_path)
                            save_index_metadata(
                                filenames=[t[1] for t in texts_with_meta],
                                chunk_count=len(text_chunks),
                                index_path=active_path,
                            )
                            prog.progress(1.0)

                            st.session_state.vectorstore = vectorstore
                            st.session_state.memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )
                            st.session_state.chat_history = []
                            st.session_state.sources = []

                            # Generate suggested questions in the background step
                            st.write("Generating suggested questions…")
                            st.session_state.suggested_questions = generate_suggested_questions(
                                text_chunks, api_key=get_api_key(), model=st.session_state.model
                            )
                            proc_status.update(
                                label=f"Done — {len(text_chunks)} chunks from "
                                f"{len(texts_with_meta)} document(s)",
                                state="complete",
                                expanded=False,
                            )
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

        if os.path.exists(active_path):
            st.divider()
            meta = load_index_metadata(active_path)
            if meta:
                ts = datetime.fromisoformat(meta["timestamp"])
                age_min = int((datetime.now(timezone.utc) - ts).total_seconds() / 60)
                age_str = f"{age_min} min ago" if age_min < 60 else f"{age_min // 60} h ago"
                st.caption(
                    f"**Index loaded** · {meta['chunks']} chunks · {age_str}  \n"
                    + ", ".join(meta["files"])
                )
            if st.button("Clear saved index"):
                shutil.rmtree(active_path)
                st.session_state.vectorstore = None
                st.session_state.memory = None
                st.session_state.chat_history = []
                st.session_state.sources = []
                st.session_state.suggested_questions = []
                st.rerun()


if __name__ == "__main__":
    main()
