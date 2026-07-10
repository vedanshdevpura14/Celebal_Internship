# Memory-Augmented Chatbot (Advanced Beginner Project)

This project demonstrates a full end-to-end intelligent chatbot system that integrates **Retrieval-Augmented Generation (RAG)**, a **Knowledge Graph**, **Long-Term Memory**, and **Dynamic Web Search Tools**—all orchestrated cleanly using **LangGraph**.

While it meets advanced architectural requirements, the code is structured in a beginner-friendly way, keeping all the logic contained in just a few files rather than a massive enterprise framework.

## What's in this folder

```
memory_chatbot/
├── requirements.txt        # All Python packages needed
├── data/
│   ├── memory.db           # SQLite database for User Memory & Knowledge Graph
│   └── chroma_db/          # Local vector database for RAG context
└── src/
    ├── rag_setup.py        # STEP 1: Scrape, chunk, and embed static data
    ├── main.py             # STEP 2: The core backend, LangGraph, and UI
    └── eval.py             # STEP 3: Automated LLM-as-a-judge evaluation
```

## Features Complete

1. **RAG Pipeline:** Static web knowledge is scraped, chunked, and embedded into a local ChromaDB vector database.
2. **Knowledge Graph:** Extracts `(Entity A, relation, Entity B)` triples dynamically and stores them in a local SQLite Triplestore.
3. **Long-Term Memory:** Extracts user preferences and chat history and saves them to a local SQLite database.
4. **LangGraph Orchestration:** A defined `StateGraph` routes queries to either the `RAG/Graph Node`, the `Web Search Node` (DuckDuckGo), or the `Direct LLM Node`.
5. **Context-Aware Responses:** Merges user memory, chat history, and retrieved context to answer accurately.
6. **Evaluation Framework:** An automated script (`eval.py`) that uses LLM-as-a-judge to score answers out of 10 for Correctness and Context Relevance.
7. **FastAPI & UI:** A beautiful, responsive glassmorphism interface powered directly by the FastAPI backend.

## One-time Setup

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

## How to run the project

### 1. Build the Knowledge Base
You only need to run this once to scrape the Wikipedia data and build the Vector Database:
```bash
python src/rag_setup.py
```

### 2. Start the Chatbot
This launches the FastAPI server and the LangGraph workflow:
```bash
python src/main.py
```
After running this, open your web browser and go to `http://localhost:8000` to interact with the beautiful UI!

### 3. Run the Evaluator
To test the accuracy of the bot, open a separate terminal and run the evaluation script:
```bash
python src/eval.py
```

## Deployment
This project is ready to be deployed on platforms like **Render**, **Railway**, or **Heroku**. 
- Simply upload the files to GitHub.
- Connect your GitHub repo to your hosting platform.
- Use the Build Command: `pip install -r requirements.txt && python src/rag_setup.py`
- Use the Start Command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
