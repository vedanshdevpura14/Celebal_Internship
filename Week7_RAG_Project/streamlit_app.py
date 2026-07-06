import os
import streamlit as st
import sys
import tempfile

# ---------------- PAGE CONFIG ----------------
# st.set_page_config MUST be the very first Streamlit command called
st.set_page_config(
    page_title="Document Q&A RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Custom CSS for modern premium design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
}

.main-header {
    background: linear-gradient(90deg, #FF4B4B, #8B5CF6, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.custom-card {
    background-color: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

.source-card {
    background-color: rgba(59, 130, 246, 0.08);
    border-left: 4px solid #3B82F6;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
}

.step-indicator {
    padding: 0.5rem 1rem;
    background-color: rgba(128, 128, 128, 0.05);
    border-radius: 8px;
    border-left: 3px solid #FF4B4B;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR & API KEY RESOLUTION ----------------
st.sidebar.title("🛠️ Configuration")

# API Key lookup sequence
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]

if not api_key:
    st.sidebar.warning("⚠️ Google API Key not found in environment or secrets.")
    api_key = st.sidebar.text_input(
        "Enter Google API Key:",
        type="password",
        help="Get your key at: https://aistudio.google.com/apikey"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.sidebar.info("Please input API Key to unlock Document Q&A features.")
else:
    st.sidebar.success("🔑 API Key configured.")

# Footnote in sidebar
st.sidebar.markdown("---")
st.sidebar.caption(f"Python: {sys.version.split()[0]} | Streamlit: {st.__version__}")

# ---------------- MAIN UI ----------------
st.markdown("<h1 class='main-header'>📄 Document Question Answering (RAG)</h1>", unsafe_allow_html=True)
st.write("Upload a PDF document, and ask natural language questions. The system retrieves the relevant sections from your file to generate accurate, grounded answers.")

# ---------------- PDF LOADER ----------------
def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return docs

# ---------------- TEXT SPLITTER ----------------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)

# ---------------- VECTOR STORE ----------------
def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

# ---------------- MAIN APP LOGIC ----------------
if not api_key:
    st.info("👈 Please enter a Google API Key in the sidebar to get started.")
else:
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type="pdf",
    )

    if uploaded_file:
        # Detect if a different file is uploaded to clear previous vectorstore/responses
        file_changed = False
        if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            file_changed = True
            st.session_state.uploaded_file_name = uploaded_file.name
            
        if "vectorstore" not in st.session_state or file_changed:
            with st.status("Processing PDF...", expanded=True) as status:
                st.markdown("<div class='step-indicator'>1️⃣ Loading PDF pages...</div>", unsafe_allow_html=True)
                docs = load_pdf(uploaded_file)
                
                st.markdown("<div class='step-indicator'>2️⃣ Splitting document into text chunks...</div>", unsafe_allow_html=True)
                chunks = split_docs(docs)
                
                st.markdown("<div class='step-indicator'>3️⃣ Creating embeddings & building vector database...</div>", unsafe_allow_html=True)
                st.session_state.vectorstore = build_vectorstore(chunks)
                
                status.update(label="PDF processed successfully!", state="complete", expanded=False)
            
            # Clear previous results if file changed
            if file_changed:
                if "question" in st.session_state:
                    del st.session_state.question

        st.success(f"Loaded: **{uploaded_file.name}**")

        question = st.text_input("Ask a question about the document:")

        if question:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
            )

            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            # RAG Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """You are an expert Q&A assistant. Answer the user's question based strictly on the retrieved context below.
If the context does not contain enough information to answer the question, say "I cannot find the answer in the uploaded document." 
Do not make up facts or use external knowledge.

Context:
{context}

Question: {input}

Answer:"""
            )

            document_chain = create_stuff_documents_chain(
                llm,
                prompt,
            )

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain,
            )

            with st.spinner("Generating grounded answer..."):
                response = retrieval_chain.invoke(
                    {"input": question}
                )

            st.markdown("### 💬 Answer")
            st.markdown(f"<div class='custom-card'>{response['answer']}</div>", unsafe_allow_html=True)

            # Display citations/source passages
            st.markdown("### 📚 Source Citations")
            for i, doc in enumerate(response["context"]):
                source_name = doc.metadata.get("source", "Uploaded PDF")
                page = doc.metadata.get("page", 0)
                page_num = page + 1 if isinstance(page, int) else page
                snippet = doc.page_content.strip()
                
                st.markdown(f"""
                <div class='source-card'>
                    <strong>Passage {i+1}</strong> (Page {page_num})<br/>
                    <div style='margin-top: 0.4rem; color: #4B5563; font-style: italic; line-height: 1.4;'>
                        "{snippet}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
