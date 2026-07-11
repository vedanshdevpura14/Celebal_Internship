import os
import re
import json
import requests
from bs4 import BeautifulSoup
import chromadb

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "chatbot_knowledge"

URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
]

# Ensure directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR]:
    os.makedirs(folder, exist_ok=True)

#Scraper
def clean_filename(url: str) -> str:
    name = re.sub(r"https?://", "", url)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_") + ".txt"

def scrape_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (educational project bot)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()

def run_scraper():
    print("--- STEP 1: Scraping ---")
    for url in URLS:
        print(f"Scraping: {url}")
        try:
            text = scrape_page(url)
        except Exception as e:
            print(f"  Failed: {e}")
            continue
        filepath = os.path.join(RAW_DATA_DIR, clean_filename(url))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved {len(text)} characters -> {filepath}")

# Chunker
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words): break
        start = end - overlap
    return chunks

def run_chunker():
    print("\n--- STEP 2: Chunking ---")
    all_chunks = []
    chunk_id = 0
    for filename in os.listdir(RAW_DATA_DIR):
        if not filename.endswith(".txt"): continue
        with open(os.path.join(RAW_DATA_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        for chunk in chunks:
            all_chunks.append({"id": f"chunk_{chunk_id}", "source": filename, "text": chunk})
            chunk_id += 1
        print(f"{filename}: {len(chunks)} chunks created")
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"Total chunks saved: {len(all_chunks)}")

#Embed Store
def run_embed_store():
    print("\n--- STEP 3: Embedding and Storing ---")
    chunks_path = os.path.join(PROCESSED_DATA_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        print("No chunks found. Did you run the chunker?")
        return
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY environment variable is missing!")
        print("Please set it in your terminal or .env file before running this.")
        return
        
    print("Connecting to Google Gemini API for embeddings...")
    import chromadb.utils.embedding_functions as embedding_functions
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/gemini-embedding-2"
    )
    
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]
    
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except:
        pass
        
    collection = client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=google_ef)
    
    print("Generating embeddings via API and storing in Chroma...")
    import time
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        print(f"  Processing batch {i} to {end}...")
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )
        time.sleep(5)
        
    print(f"Stored {len(chunks)} chunks in ChromaDB.")

def main():
    run_scraper()
    run_chunker()
    run_embed_store()
    print("\nSetup complete! You can now run main.py")

if __name__ == "__main__":
    main()
