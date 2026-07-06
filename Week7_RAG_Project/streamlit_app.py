

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="centered")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# API key: read from Streamlit secrets (cloud) or environment variable (local)

def get_api_key():
    if "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    elif os.getenv("GOOGLE_API_KEY"):
        return os.getenv("GOOGLE_API_KEY")
    else:
        return st.text_input("Enter your Google API Key:", type="password")


# Cached resources so we don't reload the embedding model on every interaction
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_and_chunk(uploaded_files):
    docs = []
    for uf in uploaded_files:
        suffix = os.path.splitext(uf.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = uf.name  # use original filename, not temp path
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    embeddings = load_embeddings()
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def build_qa_chain(vectorstore, api_key, k=6):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


# UI
st.title("📄 Document Question Answering (RAG)")
st.caption("Upload a PDF or text file, then ask questions grounded in its content.")

api_key = get_api_key()
if not api_key:
    st.error(
        "No Gemini API key found. Add GOOGLE_API_KEY in Streamlit secrets "
        "(Settings → Secrets) or set it as an environment variable locally."
    )
    st.stop()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if st.button("Build knowledge base", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading files and building the vector database..."):
            chunks = load_and_chunk(uploaded_files)
            st.session_state.vectorstore = build_vectorstore(chunks)
            st.session_state.messages = []
        st.success(f"Ready! Indexed {len(chunks)} chunk(s) from {len(uploaded_files)} file(s).")

    st.divider()
    st.caption("This app uses local, free embeddings (sentence-transformers) "
               "and Google Gemini 2.5 Flash for answer generation.")

if st.session_state.vectorstore is None:
    st.info("👈 Upload one or more documents and click **Build knowledge base** to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your document(s)...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = build_qa_chain(st.session_state.vectorstore, api_key)
                result = qa_chain.invoke({"query": question})
                answer = result["result"]
                st.markdown(answer)

                seen = set()
                sources_md = ""
                for doc in result["source_documents"]:
                    key = doc.page_content[:100]
                    if key in seen:
                        continue
                    seen.add(key)
                    src = doc.metadata.get("source", "unknown")
                    preview = doc.page_content[:150].replace("\n", " ")
                    sources_md += f"- **{src}**: {preview}...\n"

                with st.expander("Sources used"):
                    st.markdown(sources_md)

        st.session_state.messages.append({"role": "assistant", "content": answer})
