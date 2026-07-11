# Memory-Augmented Chatbot 

**Live Demo:** [https://memory-chatbot-bf47.onrender.com](https://memory-chatbot-bf47.onrender.com)
*(Note: As this is hosted on a free tier, it may take ~50 seconds to wake up if it has been inactive for 15 minutes.)*

This project demonstrates a full end-to-end intelligent chatbot system that integrates **Retrieval-Augmented Generation (RAG)**, a **Knowledge Graph**, **Long-Term Memory**, and **Dynamic Web Search Tools**—all orchestrated cleanly using **LangGraph**.

While it meets advanced architectural requirements, the code was deliberately structured in a beginner-friendly way, keeping all the logic contained in just a few files rather than a massive enterprise framework.

## Project Architecture

```
memory_chatbot/
├── requirements.txt        # All Python packages needed
├── data/
│   ├── memory.db           # SQLite database for User Memory & Knowledge Graph
│   └── chroma_db/          # Local vector database for RAG context
└── src/
    ├── rag_setup.py        # Script used to scrape, chunk, and embed static data
    ├── main.py             # The core backend, LangGraph, and FastAPI UI
    └── eval.py             # Automated LLM-as-a-judge evaluation
```

## Features Implemented

1. **RAG Pipeline:** We scraped, chunked, and embedded static web knowledge into a local ChromaDB vector database using the Google Gemini API.
2. **Knowledge Graph:** The bot dynamically extracts `(Entity A, relation, Entity B)` triples from conversations and stores them in a local SQLite Triplestore.
3. **Long-Term Memory:** User preferences and chat history are extracted and saved persistently to a local SQLite database.
4. **LangGraph Orchestration:** We defined a `StateGraph` that intelligently routes queries to either the `RAG/Graph Node`, the `Web Search Node` (DuckDuckGo), or the `Direct LLM Node`.
5. **Context-Aware Responses:** The AI merges user memory, chat history, and retrieved context to answer accurately.
6. **Evaluation Framework:** We built an automated script (`eval.py`) that uses an LLM-as-a-judge to score answers out of 10 for Correctness and Context Relevance.
7. **FastAPI & UI:** We created a beautiful, responsive glassmorphism web interface powered directly by the FastAPI backend, which is currently deployed live on Render.

## How to run locally (For testing)

1. Make sure you have **Python 3.10+** installed.
2. Open a terminal in this project folder.
3. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   # On Mac/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set your Google Gemini API key:
   - Mac/Linux: `export GEMINI_API_KEY="your_api_key"`
   - Windows: `$env:GEMINI_API_KEY="your_api_key"`
6. Start the FastAPI server:
   ```bash
   python src/main.py
   ```
7. Open your web browser and go to `http://localhost:8000`.

*(Note: You do not need to run `rag_setup.py` as the `data/` folder containing the pre-built vector database and memory triplestore is already included in this repository).*
