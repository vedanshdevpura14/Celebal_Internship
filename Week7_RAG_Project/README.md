# Document Question Answering System (RAG)

A Retrieval-Augmented Generation pipeline that answers questions grounded in your own documents (PDFs or text files), instead of relying only on an LLM's internal knowledge.

## How it works (pipeline stages)

1. **Document Ingestion** — load PDFs/TXT files from `data/`
2. **Text Chunking** — split documents into overlapping ~500-character chunks
3. **Embedding Creation** — convert each chunk into a vector (local, free — uses `sentence-transformers`)
4. **Vector Database** — store embeddings in Chroma for similarity search
5. **Query Processing** — convert the user's question into an embedding
6. **Context Retrieval** — fetch the top-k most relevant chunks
7. **Answer Generation** — an LLM (Gemini 1.5 Flash via Google, free tier) generates the final answer using the retrieved context

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a free Google Gemini API key (needed only for the answer-generation step; embeddings are local/free) at **https://aistudio.google.com/apikey** — no credit card required. Then set it as an environment variable:
   ```bash
   export GOOGLE_API_KEY="AI...."
   ```
   On Windows (Command Prompt): `set GOOGLE_API_KEY=AI....`
   On Windows (PowerShell): `$env:GOOGLE_API_KEY="AI...."`

   Want to use OpenAI or a local model instead? See "Using a different LLM" below.

3. Get your documents into `data/`. You have two options:

   **Option A — your own files:** drop PDFs or `.txt` files into `data/` directly (resume, notes, a paper you know well, etc.)

   **Option B — the Open RAG Benchmark dataset** ([vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench)):
   this is a research benchmark of 1000 arXiv papers with 3045 pre-written
   questions + ground-truth answers, built for evaluating RAG systems. It's
   not a flat CSV — it's a BEIR-style structure (`corpus/`, `queries.json`,
   `qrels.json`, `answers.json`). Use the included downloader:
   ```bash
   python download_ragbench.py --num-papers 20
   ```
   This pulls 20 random papers (out of 1000 — the full set is 700MB+), converts
   each into a plain-text file in `data/`, and saves the subset of matching
   questions + ground-truth answers into `eval/` so you can grade your system
   automatically (see "Evaluating against ground truth" below).

## Usage

**Step 1 — Build the vector database** (run once, or again after adding new files):
```bash
python rag.py ingest
```

**Step 2 — Ask a question:**
```bash
python rag.py ask "What is the main idea of the document?"
```

**Or chat interactively:**
```bash
python rag.py chat
```

Each answer also prints which document chunks were used as sources — this is what makes the answer "grounded" rather than hallucinated.

## Evaluating against ground truth (only if using the ragbench dataset)

After running `download_ragbench.py` and `rag.py ingest`, score your pipeline
against the dataset's real questions and reference answers:
```bash
python evaluate.py --limit 10
```
This reports, per question:
- Whether the correct source paper was retrieved (retrieval accuracy)
- The generated answer vs. the reference answer
- A simple word-overlap score as a rough automated proxy

Drop `--limit` to run the full sampled set. This is exactly the kind of
metric you can report in your "Improvements & Experiments" write-up section.

## Using a different LLM

This project uses Gemini by default (free, no billing needed). If you'd rather use something else, swap the LLM in `rag.py`:

- **OpenAI:**
  ```bash
  pip install langchain-openai
  ```
  Replace the `ChatGoogleGenerativeAI` import/line with:
  ```python
  from langchain_openai import ChatOpenAI
  llm = ChatOpenAI(model="gpt-4o-mini")
  ```
  (Requires `OPENAI_API_KEY` and billing credit on your OpenAI account.)

- **Local model via Ollama (fully free, runs on your machine):**
  ```bash
  pip install langchain-ollama
  ```
  ```python
  from langchain_ollama import ChatOllama
  llm = ChatOllama(model="llama3")
  ```

## Suggested experiments (for extra credit / deeper learning)

- Change `chunk_size` / `chunk_overlap` in `rag.py` and see how answer quality changes
- Swap `EMBED_MODEL` for a different embedding model and compare retrieval quality
- Add hybrid search (keyword + vector) using `BM25Retriever` combined with the vector retriever
- Add a re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to reorder retrieved chunks before generation
- Try feeding in multiple documents at once and ask cross-document questions

## Project structure

```
rag_project/
├── rag.py                  # main pipeline (ingestion + retrieval + generation)
├── download_ragbench.py    # downloads & converts a subset of vectara/open_ragbench
├── evaluate.py              # scores your pipeline against ragbench ground truth
├── requirements.txt         # dependencies
├── data/                     # put your PDFs/TXT files here (or auto-filled by download_ragbench.py)
├── eval/                     # queries.json / qrels.json / answers.json (from download_ragbench.py)
└── chroma_db/                # auto-created vector database (after running "ingest")
```

## Key concepts (for your write-up)

- **Retrieval**: finds the most relevant text chunks using embeddings + vector similarity search
- **Augmentation**: the retrieved chunks are inserted into the LLM's prompt as context
- **Generation**: the LLM produces a final answer grounded in that context, reducing hallucination
