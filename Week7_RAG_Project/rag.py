
import os
import sys
import glob

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

DATA_DIR = "data"
DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



# Stage 1 + 2: Document Ingestion + Text Chunking
def load_documents():
    """Load every PDF/TXT file inside the data/ folder."""
    docs = []
    files = glob.glob(os.path.join(DATA_DIR, "*.pdf")) + glob.glob(os.path.join(DATA_DIR, "*.txt"))

    if not files:
        print(f"No PDF or TXT files found in '{DATA_DIR}/'. Add some files and try again.")
        sys.exit(1)

    for path in files:
        print(f"Loading: {path}")
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    return docs


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# Stage 3 + 4: Embedding Creation + Vector Database
def build_vectorstore(chunks):
    """Embed chunks and persist them in a local Chroma vector database."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore():
    """Load an already-built vector database from disk."""
    if not os.path.exists(DB_DIR):
        print("No vector database found. Run 'python rag.py ingest' first.")
        sys.exit(1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


# Stage 5 + 6 + 7: Query Processing, Context Retrieval, Answer Generation

def build_qa_chain(vectorstore, k=6):
    """Wire retriever + LLM together into a question-answering chain."""
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Get a free key at https://aistudio.google.com/apikey")
        print("Then set it with: set GOOGLE_API_KEY=AI....   (Windows)")
        sys.exit(1)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def ask_question(question):
    vectorstore = load_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    result = qa_chain.invoke({"query": question})

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    print("ANSWER:\n", result["result"])
    print("\n--- Sources used ---")
    seen = set()
    i = 0
    for doc in result["source_documents"]:
        key = doc.page_content[:100]
        if key in seen:
            continue
        seen.add(key)
        i += 1
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"[{i}] {source} (page {page}): {preview}...")
    print()


# CLI entry point

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "ingest":
        print("Step 1: Loading documents...")
        docs = load_documents()
        print(f"Loaded {len(docs)} document page(s).")

        print("Step 2: Splitting into chunks...")
        chunks = chunk_documents(docs)
        print(f"Created {len(chunks)} chunk(s).")

        print("Step 3+4: Creating embeddings and building vector database...")
        build_vectorstore(chunks)
        print(f"Done! Vector database saved to '{DB_DIR}/'.")

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python rag.py ask \"your question here\"")
            sys.exit(1)
        question = sys.argv[2]
        ask_question(question)

    elif command == "chat":
        vectorstore = load_vectorstore()
        qa_chain = build_qa_chain(vectorstore)
        print("Chat mode. Type 'exit' to quit.\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            result = qa_chain.invoke({"query": question})
            print("Bot:", result["result"], "\n")

    elif command == "debug":
        print("Checking what text was actually extracted from each file...\n")
        docs = load_documents()
        by_source = {}
        for d in docs:
            src = d.metadata.get("source", "unknown")
            by_source.setdefault(src, []).append(d)
        for src, pages in by_source.items():
            total_chars = sum(len(p.page_content) for p in pages)
            print(f"File: {src}")
            print(f"  Pages loaded: {len(pages)}")
            print(f"  Total characters extracted: {total_chars}")
            preview = pages[0].page_content[:200].replace("\n", " ") if pages else ""
            print(f"  Preview: {preview!r}")
            if total_chars < 50:
                print("  WARNING: almost no text extracted — this PDF is likely a scanned")
                print("  image rather than real text, so it can't be searched as-is.")
            print()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
