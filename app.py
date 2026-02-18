import os
import shutil

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader

from htmlTemplates import bot_template, css, user_template

# Uncomment for HuggingFace alternatives:
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings

FAISS_INDEX_PATH = "faiss_index"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Could not read '{pdf.name}': {e}")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def load_vectorstore():
    """Load a previously saved FAISS index from disk. Returns None on failure."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    try:
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Could not load saved index: {e}")
        return None


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDF documents first.")
        return

    try:
        response = st.session_state.conversation.invoke({"question": user_question})
        st.session_state.chat_history = response["chat_history"]
    except Exception as e:
        st.error(f"Error getting response: {e}")
        return

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Auto-load a previously saved FAISS index so users don't have to re-upload
    # PDFs after a page reload or server restart.
    if st.session_state.conversation is None:
        vectorstore = load_vectorstore()
        if vectorstore is not None:
            st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
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
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():
                            st.error(
                                "No text could be extracted from the uploaded PDFs. "
                                "Ensure the files are not scanned images or password-protected."
                            )
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            vectorstore.save_local(FAISS_INDEX_PATH)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("Documents processed and index saved!")
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")

        if os.path.exists(FAISS_INDEX_PATH):
            st.divider()
            if st.button("Clear saved index"):
                shutil.rmtree(FAISS_INDEX_PATH)
                st.session_state.conversation = None
                st.session_state.chat_history = None
                st.rerun()


if __name__ == "__main__":
    main()
