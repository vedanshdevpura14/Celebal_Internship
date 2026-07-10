import os
import json
import sqlite3
import requests
import urllib.parse
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DB_PATH = os.path.join(BASE_DIR, "data", "memory.db")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
CHROMA_COLLECTION_NAME = "chatbot_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure data directory exists
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

app = FastAPI(title="Beginner AI Chatbot")

#  Memory System (SQLite)
def get_db_connection():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id TEXT, memory_key TEXT, memory_value TEXT,
                PRIMARY KEY (user_id, memory_key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, user_id TEXT, role TEXT, content TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                entity_a TEXT, relation TEXT, entity_b TEXT
            )
        """)
    return conn

def get_memory_string(user_id: str) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT memory_key, memory_value FROM user_memories WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    if not rows: return "No long-term memories stored yet."
    return "\n".join([f"- {row['memory_key']}: {row['memory_value']}" for row in rows])

def add_memory(user_id: str, key: str, value: str):
    conn = get_db_connection()
    with conn:
        conn.execute("""
            INSERT INTO user_memories (user_id, memory_key, memory_value)
            VALUES (?, ?, ?) ON CONFLICT(user_id, memory_key) DO UPDATE SET memory_value = excluded.memory_value
        """, (user_id, key, value))
    conn.close()

def save_chat_message(session_id: str, user_id: str, role: str, content: str):
    conn = get_db_connection()
    with conn:
        conn.execute("INSERT INTO chat_history (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                     (session_id, user_id, role, content))
    conn.close()

def get_chat_history(session_id: str, limit: int = 5) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = cursor.fetchall()
    conn.close()
    history = [f"{row['role'].capitalize()}: {row['content']}" for row in reversed(rows)]
    return "\n".join(history)

#  LLM Integration
def call_llm(prompt: str, system_instruction: str = "") -> str:
    if os.environ.get("GEMINI_API_KEY"):
        from google import genai
        client = genai.Client()
        config = {"system_instruction": system_instruction} if system_instruction else None
        return client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=config).text
    elif os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        messages = [{"role": "system", "content": system_instruction}] if system_instruction else []
        messages.append({"role": "user", "content": prompt})
        return client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2).choices[0].message.content
    else:
        # Fallback to local Ollama
        full_prompt = f"System: {system_instruction}\nUser: {prompt}" if system_instruction else prompt
        try:
            res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": full_prompt, "stream": False}, timeout=120)
            res.raise_for_status()
            return res.json()["response"]
        except Exception as e:
            return f"Error calling Ollama: {e}"

#  Tools
def web_search(query: str) -> str:
    print("Performing web search...")
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.post(url, headers=headers, data={"q": query}, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for element in soup.find_all("div", class_="result")[:3]:
            title = element.find("a", class_="result__url")
            snippet = element.find("a", class_="result__snippet")
            if title and snippet:
                results.append(f"Source: {title.get_text(strip=True)}\nSnippet: {snippet.get_text(strip=True)}")
        return "\n\n".join(results) if results else "No web results found."
    except Exception as e:
        return f"Web search failed: {e}"

def retrieve_rag(query: str) -> str:
    print("Performing RAG retrieval...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        return "\n\n".join(results["documents"][0]) if results and "documents" in results else "No knowledge base documents found."
    except Exception as e:
        return f"RAG retrieval failed: {e}"

#  Core Workflow (Router)
def update_memory_from_text(user_id: str, text: str):
    prompt = f"Extract facts about the user from this text as JSON (e.g. {{\"name\": \"Alice\", \"hobby\": \"coding\"}}). Text: {text}"
    response = call_llm(prompt, "Return pure JSON only.")
    try:
        # Simple JSON extraction
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            data = json.loads(response[start:end])
            for k, v in data.items():
                if v: add_memory(user_id, k, str(v))
    except:
        pass

# --- Knowledge Graph System ---
def extract_and_store_triples(text: str):
    prompt = f"Extract knowledge graph triples from this text as JSON (format: [{{\"a\": \"Entity1\", \"r\": \"relation\", \"b\": \"Entity2\"}}]). Text: {text}"
    response = call_llm(prompt, "Return pure JSON list only.")
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end != -1:
            data = json.loads(response[start:end])
            conn = get_db_connection()
            with conn:
                for triple in data:
                    if "a" in triple and "r" in triple and "b" in triple:
                        conn.execute("INSERT INTO knowledge_graph (entity_a, relation, entity_b) VALUES (?, ?, ?)",
                                     (str(triple["a"]), str(triple["r"]), str(triple["b"])))
            conn.close()
    except Exception as e:
        pass

def get_graph_context(query: str) -> str:
    # A super simple search: just look for words from the query in the graph
    words = [w for w in query.lower().split() if len(w) > 3]
    if not words: return ""
    
    conn = get_db_connection()
    cursor = conn.cursor()
    results = []
    for word in words:
        cursor.execute("SELECT * FROM knowledge_graph WHERE LOWER(entity_a) LIKE ? OR LOWER(entity_b) LIKE ? LIMIT 3", (f"%{word}%", f"%{word}%"))
        results.extend(cursor.fetchall())
    conn.close()
    
    if not results: return "No structured graph data found."
    return "\n".join([f"- {r['entity_a']} {r['relation']} {r['entity_b']}" for r in results[:5]])

# --- Core Workflow (LangGraph) ---
from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query: str
    user_id: str
    session_id: str
    route: str
    context: str
    source_used: str
    response: str

def router_node(state: AgentState):
    route_prompt = f"Decide the best source to answer the query: '{state['query']}'. Output exactly one word: 'RAG', 'WEB', or 'DIRECT'."
    route = call_llm(route_prompt).strip().upper()
    return {"route": route}

def rag_node(state: AgentState):
    vec_context = retrieve_rag(state['query'])
    kg_context = get_graph_context(state['query'])
    combined = f"Vector RAG:\n{vec_context}\n\nKnowledge Graph:\n{kg_context}"
    return {"context": combined, "source_used": "Knowledge Base (RAG & Graph)"}

def web_node(state: AgentState):
    return {"context": web_search(state['query']), "source_used": "Web Search"}

def generate_node(state: AgentState):
    memory = get_memory_string(state['user_id'])
    history = get_chat_history(state['session_id'])
    
    prompt = f"User Memory:\n{memory}\n\nRecent History:\n{history}\n\nContext:\n{state.get('context', 'None')}\n\nUser Query: {state['query']}"
    response = call_llm(prompt, "You are a helpful assistant. Use the context and memory to answer the question.")
    
    # Save to history & update memory and graph
    save_chat_message(state['session_id'], state['user_id'], "assistant", response)
    update_memory_from_text(state['user_id'], f"User: {state['query']}\\nAssistant: {response}")
    extract_and_store_triples(f"User: {state['query']}\\nAssistant: {response}")
    
    return {"response": response}

def decide_next_node(state: AgentState):
    if "WEB" in state["route"]: return "web_node"
    if "DIRECT" in state["route"]: return "generate_node"
    return "rag_node"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("router_node", router_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("web_node", web_node)
workflow.add_node("generate_node", generate_node)

workflow.set_entry_point("router_node")
workflow.add_conditional_edges("router_node", decide_next_node, {
    "rag_node": "rag_node",
    "web_node": "web_node",
    "generate_node": "generate_node"
})
workflow.add_edge("rag_node", "generate_node")
workflow.add_edge("web_node", "generate_node")
workflow.add_edge("generate_node", END)

app_workflow = workflow.compile()

def process_chat(query: str, user_id: str, session_id: str) -> dict:
    save_chat_message(session_id, user_id, "user", query)
    initial_state = {"query": query, "user_id": user_id, "session_id": session_id, "context": "", "source_used": "Direct"}
    result = app_workflow.invoke(initial_state)
    
    diagnostics = f"LangGraph Route: {result.get('route', 'Direct')} Node | "
    diagnostics += f"Source: {result.get('source_used', 'None')} | "
    diagnostics += "Memory & Knowledge Graph Updated"
    
    return {"response": result["response"], "source_used": result["source_used"], "diagnostics": diagnostics}

# API Endpoints
class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: str = "default_session"

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return process_chat(req.query, req.user_id, req.session_id)

@app.get("/api/memory")
async def get_user_memory(user_id: str = "default_user"):
    return {"memory": get_memory_string(user_id)}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Memory Chatbot</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * { box-sizing: border-box; font-family: 'Outfit', sans-serif; }
        body {
            margin: 0; padding: 0;
            background: linear-gradient(135deg, #0f172a, #1e293b, #334155);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: #fff;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .container {
            width: 100%;
            max-width: 800px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            display: flex;
            flex-direction: column;
            height: 90vh;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        .container:hover { transform: scale(1.005); }
        .header {
            padding: 25px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            font-weight: 600;
            font-size: 24px;
            letter-spacing: 1.5px;
            background: rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .status-dot {
            width: 10px; height: 10px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.7); } 70% { box-shadow: 0 0 0 10px rgba(16,185,129,0); } 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); } }
        #chat {
            flex: 1;
            padding: 25px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
            scroll-behavior: smooth;
        }
        #chat::-webkit-scrollbar { width: 8px; }
        #chat::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
        #chat::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 10px; }
        
        .msg {
            max-width: 80%; padding: 15px 22px; border-radius: 20px; 
            line-height: 1.6; font-size: 16px; 
            animation: slideUp 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
            opacity: 0;
            transform: translateY(20px);
        }
        .user { 
            background: linear-gradient(135deg, #6366f1, #8b5cf6); 
            align-self: flex-end; border-bottom-right-radius: 4px; 
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        .bot { 
            background: rgba(255,255,255,0.1); 
            border: 1px solid rgba(255,255,255,0.05);
            align-self: flex-start; border-bottom-left-radius: 4px; 
        }
        .meta { 
            font-size: 0.75em; 
            margin-top: 10px; 
            padding: 8px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            border-left: 3px solid #10b981;
            color: #94a3b8;
            font-family: monospace;
        }
        
        .input-area {
            display: flex; padding: 20px; border-top: 1px solid rgba(255,255,255,0.1);
            background: rgba(0,0,0,0.2);
            gap: 15px;
        }
        input {
            flex: 1; padding: 16px 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); outline: none;
            background: rgba(0,0,0,0.4); color: white; font-size: 16px; transition: all 0.3s;
        }
        input:focus { 
            background: rgba(0,0,0,0.6); border-color: #8b5cf6;
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.3); 
        }
        button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white;
            border: none; border-radius: 12px; padding: 0 30px; cursor: pointer;
            font-weight: 600; font-size: 16px; transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5); }
        button:active { transform: translateY(0); }
        
        .typing { display: none; align-self: flex-start; padding: 15px 22px; background: rgba(255,255,255,0.05); border-radius: 20px; border-bottom-left-radius: 4px; }
        .typing-dots span { display: inline-block; width: 6px; height: 6px; background: #fff; border-radius: 50%; margin: 0 2px; animation: bounce 1.4s infinite ease-in-out both; }
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes slideUp { to { opacity: 1; transform: translateY(0); } }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="status-dot"></div>
            Memory Chatbot
        </div>
        <div id="chat">
            <div class="msg bot">Hello! I'm your memory-augmented AI. Ask me anything!</div>
            <div class="typing" id="typing-indicator">
                <div class="typing-dots"><span></span><span></span><span></span></div>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="query" placeholder="Ask me a question..." onkeypress="if(event.key==='Enter') send()" autocomplete="off">
            <button onclick="send()">Send</button>
        </div>
    </div>

    <script>
        async function send() {
            const input = document.getElementById('query');
            const chat = document.getElementById('chat');
            const typing = document.getElementById('typing-indicator');
            const q = input.value.trim();
            if(!q) return;
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'msg user';
            userMsg.textContent = q;
            chat.insertBefore(userMsg, typing);
            input.value = '';
            
            // Show typing indicator and scroll to bottom
            typing.style.display = 'block';
            chat.scrollTop = chat.scrollHeight;
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: q})
                });
                const data = await res.json();
                
                // Hide typing indicator
                typing.style.display = 'none';
                
                // Add bot message
                const botMsg = document.createElement('div');
                botMsg.className = 'msg bot';
                botMsg.innerHTML = marked.parse(data.response) + `<div class="meta">🧠 Diagnostics: ${data.diagnostics}</div>`;
                chat.insertBefore(botMsg, typing);
            } catch (e) {
                typing.style.display = 'none';
                const errorMsg = document.createElement('div');
                errorMsg.className = 'msg bot';
                errorMsg.textContent = "Oops! Something went wrong. Make sure the server is running.";
                chat.insertBefore(errorMsg, typing);
            }
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

if __name__ == "__main__":
    print("Starting Beginner Chatbot on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
