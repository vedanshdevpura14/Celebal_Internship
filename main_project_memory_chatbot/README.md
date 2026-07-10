# Memory-Augmented Chatbot — Day 1 Starter (RAG foundation)

This is the starting skeleton for your project. It builds the **static
knowledge layer (RAG)** first, since Knowledge Graph, Memory, and LangGraph
routing all sit on top of a working RAG pipeline.

Everything here runs **locally and for free** — no API keys needed.

## What's in this folder

```
memory_chatbot/
├── requirements.txt        # all Python packages you need
├── data/
│   ├── raw/                # scraped web pages land here (.txt)
│   ├── processed/          # chunked text lands here (chunks.json)
│   └── chroma_db/          # your local vector database (auto-created)
└── src/
    ├── config.py           # all settings in one place
    ├── scraper.py          # STEP 1: scrape web pages
    ├── chunker.py          # STEP 2: split text into chunks
    ├── embed_store.py      # STEP 3: embed chunks + store in vector DB
    ├── retrieve.py         # STEP 4: test searching your knowledge base
    └── llm_answer.py       # STEP 5: full RAG loop with a local LLM (Ollama)
```

## One-time setup

1. Make sure you have **Python 3.10+** installed.
2. Open a terminal in this `memory_chatbot/` folder.
3. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # on Mac/Linux
   venv\Scripts\activate         # on Windows
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. (Optional, for Step 5 only) Install [Ollama](https://ollama.com), then run:
   ```
   ollama pull llama3
   ```
   This downloads a free local LLM (~4-5GB) that runs entirely on your machine.

## How to run it (in order)

```bash
cd src

# Step 1: Scrape a few web pages (edit the URLS list in scraper.py first)
python scraper.py

# Step 2: Split scraped text into chunks
python chunker.py

# Step 3: Embed chunks and store them in the local vector database
python embed_store.py

# Step 4: Test that retrieval works (no LLM needed yet)
python retrieve.py "What is machine learning?"

# Step 5 (optional, needs Ollama installed): full RAG answer
python llm_answer.py "What is machine learning?"
```

If Step 4 returns relevant chunks about machine learning, **your RAG
foundation works.** That's a real, working win for Day 1.

## What each step teaches you

| Step | Concept | File |
|---|---|---|
| 1 | Web scraping & cleaning HTML | `scraper.py` |
| 2 | Chunking (why & how to split text) | `chunker.py` |
| 3 | Embeddings + vector databases | `embed_store.py` |
| 4 | Semantic similarity search | `retrieve.py` |
| 5 | Prompting an LLM with retrieved context (RAG) | `llm_answer.py` |

## Roadmap: where this fits in your full project

This starter covers **Section 3.1 (Static Knowledge Layer / RAG)** of your
problem statement. Here's the suggested order for the rest:

1. ✅ **RAG pipeline** (this starter) — scrape → chunk → embed → retrieve → generate
2. **Knowledge Graph layer** — extract entities/relationships from your chunks
   (can use an LLM prompt to extract triples like `(Entity A, relation, Entity B)`),
   store them in Neo4j (free local install or free cloud tier)
3. **Long-term memory** — a simple table (SQLite/MongoDB) keyed by `user_id`,
   storing past questions/preferences; loaded at the start of each conversation
4. **LangGraph orchestration** — wrap RAG, Knowledge Graph, Memory, and Tools
   as nodes in a graph, with a router node deciding which path to take per query
5. **Dynamic tools** — add a node that calls a live API (e.g. weather, search)
   when the question needs real-time info
6. **Evaluation** — once everything works, measure context relevance,
   faithfulness, and answer correctness (can use `ragas` library, or write
   simple LLM-as-judge prompts)
7. **FastAPI wrapper** — expose the whole system as a web API so it can be
   used from a frontend or Postman

## Troubleshooting tips

- If `sentence-transformers` model download fails: check your internet
  connection — it only needs internet the *first* time, then it's cached.
- If Chroma throws version errors: make sure your `pip install -r requirements.txt`
  completed without errors.
- If `llm_answer.py` can't connect: make sure Ollama is actually running
  (check with `ollama list` in a terminal).
