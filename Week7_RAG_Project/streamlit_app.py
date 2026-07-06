import os
import streamlit as st

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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Document Q&A RAG", page_icon="📄")
st.title("📄 Document Question Answering (RAG)")
st.write("Upload a PDF and ask questions about it.")

# ---------------- API KEY ----------------
api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

# ---------------- PDF LOADER ----------------
def load_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    os.remove("temp.pdf")
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
        model="models/embedding-001"
    )

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
)

if uploaded_file:

    if "vectorstore" not in st.session_state:

        with st.spinner("Processing PDF..."):

            docs = load_pdf(uploaded_file)
            chunks = split_docs(docs)
            st.session_state.vectorstore = build_vectorstore(chunks)

        st.success("PDF processed successfully!")

    question = st.text_input("Ask a question")

    if question:

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
        )

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        prompt = ChatPromptTemplate.from_template(
           
        )

        document_chain = create_stuff_documents_chain(
            llm,
            prompt,
        )

        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain,
        )

        with st.spinner("Generating answer..."):
            response = retrieval_chain.invoke(
                {"input": question}
            )

        st.subheader("Answer")
        st.write(response["answer"])
