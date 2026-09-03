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
from dotenv import load_dotenv

load_dotenv() # .env file se saari environment variables (jaise API keys) load karta hai

# --- Config ---
# Project ka root folder ka path nikal rahe hain takki paths absolute rahein
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DB_PATH = os.path.join(BASE_DIR, "data", "memory.db") # SQLite database kahan save hoga
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db") # Vector DB kahan save hoga
CHROMA_COLLECTION_NAME = "chatbot_knowledge" # Collection ka naam

# Ensure data directory exists
# Agar data folder nahi hai toh bana do (exist_ok=True error nahi deta agar folder pehle se hai)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# FastAPI ka app instance bana rahe hain, yahi humara server hai
app = FastAPI(title="Beginner AI Chatbot")

#  Memory System (SQLite)
def get_db_connection():
    # SQLite database se connection establish kar rahe hain
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row # Rows ko dictionary jaise access karne ke liye
    with conn: # 'with' use karne se autocommit ho jata hai (transaction safe)
        # user_memories table banayenge agar nahi hai (user ki details store karne ke liye)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id TEXT, memory_key TEXT, memory_value TEXT,
                PRIMARY KEY (user_id, memory_key)
            )
        """)
        # chat_history table banayenge pura conversation save karne ke liye
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, user_id TEXT, role TEXT, content TEXT
            )
        """)
        # knowledge_graph table banayenge structured facts save karne ke liye (entity relation model)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                entity_a TEXT, relation TEXT, entity_b TEXT
            )
        """)
    return conn

def get_memory_string(user_id: str) -> str:
    # Kisi user ki saari purani memories (facts) nikalne ke liye
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT memory_key, memory_value FROM user_memories WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    # Agar memory nahi mili toh default message
    if not rows: return "No long-term memories stored yet."
    
    # Har row ko "- key: value" format mein list comprehension se jod rahe hain
    return "\n".join([f"- {row['memory_key']}: {row['memory_value']}" for row in rows])

def add_memory(user_id: str, key: str, value: str):
    # Nayi memory database mein add karne ya update karne ke liye (upsert)
    conn = get_db_connection()
    with conn:
        conn.execute("""
            INSERT INTO user_memories (user_id, memory_key, memory_value)
            VALUES (?, ?, ?) ON CONFLICT(user_id, memory_key) DO UPDATE SET memory_value = excluded.memory_value
        """, (user_id, key, value)) # ON CONFLICT update kar dega agar key pehle se exist karti hai
    conn.close()

def save_chat_message(session_id: str, user_id: str, role: str, content: str):
    # Ek message (user ya assistant ka) chat history mein save karte hain
    conn = get_db_connection()
    with conn:
        conn.execute("INSERT INTO chat_history (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                     (session_id, user_id, role, content))
    conn.close()

def get_chat_history(session_id: str, limit: int = 5) -> str:
    # Pichle kuch messages (limit = 5) nikalne ke liye context maintain karne ko
    conn = get_db_connection()
    cursor = conn.cursor()
    # DESC order mein latein hain takki latest pehle aaye, aur limit apply karte hain
    cursor.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit))
    rows = cursor.fetchall()
    conn.close()
    
    # History ko wapas straight order mein list mein format karte hain (reversed use karke)
    history = [f"{row['role'].capitalize()}: {row['content']}" for row in reversed(rows)]
    return "\n".join(history)

#  LLM Integration
def call_llm(prompt: str, system_instruction: str = "") -> str:
    # Cascading fallback: Groq -> Gemini -> OpenAI -> Ollama
    api_errors = []
    
    # 1. Groq
    if os.environ.get("GROQ_API_KEY"):
        try:
            from groq import Groq
            client = Groq()
            messages = [{"role": "system", "content": system_instruction}] if system_instruction else []
            messages.append({"role": "user", "content": prompt})
            
            # Try multiple Groq models since they frequently update/decommission them
            groq_models = [
                "qwen/qwen3.8-27b", 
                "qwen/qwen3.6-27b",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "llama-3.3-70b-versatile", 
                "llama-3.1-8b-instant", 
                "mixtral-8x7b-32768"
            ]
            for model_name in groq_models:
                try:
                    return client.chat.completions.create(model=model_name, messages=messages, temperature=0.2).choices[0].message.content
                except Exception as e:
                    if "404" in str(e) or "400" in str(e) or "decommissioned" in str(e).lower():
                        continue # Model not found, try the next one in the list
                    raise e # For other errors (like rate limits), fallback to Gemini
            api_errors.append("Groq: All attempted models returned 404/400.")
        except Exception as e:
            api_errors.append(f"Groq: {e}")
            
    # 2. Gemini
    if os.environ.get("GEMINI_API_KEY"):
        try:
            from google import genai
            client = genai.Client()
            config = {"system_instruction": system_instruction} if system_instruction else None
            return client.models.generate_content(model='gemini-1.5-flash', contents=prompt, config=config).text
        except Exception as e:
            api_errors.append(f"Gemini: {e}")
            
    # 3. OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            messages = [{"role": "system", "content": system_instruction}] if system_instruction else []
            messages.append({"role": "user", "content": prompt})
            return client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2).choices[0].message.content
        except Exception as e:
            api_errors.append(f"OpenAI: {e}")
            
    # 4. Ollama (Local Fallback)
    try:
        full_prompt = f"System: {system_instruction}\nUser: {prompt}" if system_instruction else prompt
        res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": full_prompt, "stream": False}, timeout=120)
        res.raise_for_status()
        return res.json()["response"]
    except Exception as e:
        api_errors.append(f"Ollama: {e}")
        
    error_summary = " | ".join(api_errors)
    return f"LLM API Error: All APIs failed. Details: {error_summary}"

#  Tools
def web_search(query: str) -> str:
    # Internet se real-time information nikalne ke liye DuckDuckGo ka scraper
    print("Performing web search...")
    url = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": "Mozilla/5.0"} # Fake user agent takki block na ho
    try:
        response = requests.post(url, headers=headers, data={"q": query}, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        
        # Pehle 3 search results extract kar rahe hain
        for element in soup.find_all("div", class_="result")[:3]:
            title = element.find("a", class_="result__url")
            snippet = element.find("a", class_="result__snippet")
            if title and snippet:
                results.append(f"Source: {title.get_text(strip=True)}\nSnippet: {snippet.get_text(strip=True)}")
        
        # Saare results ko jod kar string bana denge
        return "\n\n".join(results) if results else "No web results found."
    except Exception as e:
        return f"Web search failed: {e}"

def retrieve_rag(query: str) -> str:
    # Vector Database (Chroma) se relevant knowledge base nikalne ke liye RAG (Retrieval-Augmented Generation) function
    print("Performing RAG retrieval...")
    try:
        import chromadb.utils.embedding_functions as embedding_functions
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "RAG retrieval failed: GEMINI_API_KEY is not set."
            
        # Gemini API ka use kar rahe hain text ko numbers (embeddings) mein convert karne ke liye
        google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=api_key,
            model_name="models/text-embedding-004"
        )
        
        # Chroma DB connect karte hain aur query run karte hain (n_results=3 yani top 3 match)
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=google_ef)
        results = collection.query(query_texts=[query], n_results=3)
        
        # Results ko string mein combine kar rahe hain
        return "\n\n".join(results["documents"][0]) if results and "documents" in results and results["documents"][0] else "No knowledge base documents found."
    except Exception as e:
        return f"RAG retrieval failed: {e}"

#  Core Workflow (Router)
def update_memory_from_text(user_id: str, text: str):
    # LLM ka use karke text se naye facts nikalte hain JSON format mein (e.g. name, hobby)
    prompt = f"Extract facts about the user from this text as JSON (e.g. {{\"name\": \"Alice\", \"hobby\": \"coding\"}}). Text: {text}"
    response = call_llm(prompt, "Return pure JSON only.")
    try:
        # Simple JSON extraction (curly braces extract karte hain)
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            data = json.loads(response[start:end])
            # Har nayi memory ko database mein add karte hain
            for k, v in data.items():
                if v: add_memory(user_id, k, str(v))
    except:
        pass

# --- Knowledge Graph System ---
def extract_and_store_triples(text: str):
    # LLM se Knowledge Graph (Entity-Relation-Entity) triples nikalte hain (list format mein)
    prompt = f"Extract knowledge graph triples from this text as JSON (format: [{{\"a\": \"Entity1\", \"r\": \"relation\", \"b\": \"Entity2\"}}]). Text: {text}"
    response = call_llm(prompt, "Return pure JSON list only.")
    try:
        # List [ ] nikalne ka logic
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end != -1:
            data = json.loads(response[start:end])
            conn = get_db_connection()
            with conn:
                for triple in data:
                    # Agar a, r, b teenon hain toh SQL table mein insert kar do
                    if "a" in triple and "r" in triple and "b" in triple:
                        conn.execute("INSERT INTO knowledge_graph (entity_a, relation, entity_b) VALUES (?, ?, ?)",
                                     (str(triple["a"]), str(triple["r"]), str(triple["b"])))
            conn.close()
    except Exception as e:
        pass

def get_graph_context(query: str) -> str:
    # Knowledge Graph table mein query ke words dhoondhne ke liye simple search
    
    # Query ko words mein tod lete hain (length > 3 wale sirf)
    words = [w for w in query.lower().split() if len(w) > 3]
    if not words: return ""
    
    conn = get_db_connection()
    cursor = conn.cursor()
    results = []
    
    # Har word ke liye knowledge graph me entity A ya B ko match karwate hain
    for word in words:
        cursor.execute("SELECT * FROM knowledge_graph WHERE LOWER(entity_a) LIKE ? OR LOWER(entity_b) LIKE ? LIMIT 3", (f"%{word}%", f"%{word}%"))
        results.extend(cursor.fetchall())
    conn.close()
    
    if not results: return "No structured graph data found."
    
    # Pehle 5 results ko "- A relation B" format me return kar denge
    return "\n".join([f"- {r['entity_a']} {r['relation']} {r['entity_b']}" for r in results[:5]])

# --- Core Workflow (LangGraph) ---
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Agent ki state define kar rahe hain ki current query, context, response kya hai
class AgentState(TypedDict):
    query: str
    user_id: str
    session_id: str
    route: str
    context: str
    source_used: str
    response: str

def router_node(state: AgentState):
    # LLM se puchte hain ki answer kahan se lana hai: VectorDB(RAG), Internet(WEB), ya direct LLM knowledge(DIRECT)
    route_prompt = f"""Decide the best source to answer the user query.
Rules:
- If the query asks for real-time information, current events, news, or current office holders, output 'WEB'.
- If the query asks about the user's past conversations or personal details, output 'RAG'.
- For general knowledge or simple conversation, output 'DIRECT'.

Query: '{state['query']}'
Output exactly one word: 'RAG', 'WEB', or 'DIRECT'."""
    route = call_llm(route_prompt).strip().upper()
    
    # Graceful error handling: If Ollama returns an error string, default to a safe route
    if "ERROR" in route or route not in ["RAG", "WEB", "DIRECT"]:
        print(f"Routing failed, defaulting to DIRECT. Reason: {route}")
        route = "DIRECT"
        
    return {"route": route}

def rag_node(state: AgentState):
    # Agar RAG decide hua toh ye function run hoga. Ye Vector DB aur Graph DB dono se context layega
    vec_context = retrieve_rag(state['query'])
    kg_context = get_graph_context(state['query'])
    combined = f"Vector RAG:\n{vec_context}\n\nKnowledge Graph:\n{kg_context}"
    return {"context": combined, "source_used": "Knowledge Base (RAG & Graph)"}

def web_node(state: AgentState):
    # Agar WEB decide hua toh internet search chalega
    return {"context": web_search(state['query']), "source_used": "Web Search"}

def generate_node(state: AgentState):
    # Final step: Purani memory, chat history aur retrieved context (rag ya web se aya hua) combine karke final answer generate karenge
    memory = get_memory_string(state['user_id'])
    history = get_chat_history(state['session_id'])
    
    # Final prompt ban raha hai LLM ke liye
    prompt = f"User Memory:\n{memory}\n\nRecent History:\n{history}\n\nContext:\n{state.get('context', 'None')}\n\nUser Query: {state['query']}"
    response = call_llm(prompt, "You are a helpful assistant. Answer the user's question directly using the provided Context and User Memory. Do not mention your internal knowledge cutoff date.")
    
    # Final response ko database mein save karte hain aur Memory/Graph mein update trigger karte hain
    save_chat_message(state['session_id'], state['user_id'], "assistant", response)
    update_memory_from_text(state['user_id'], f"User: {state['query']}\\nAssistant: {response}")
    extract_and_store_triples(f"User: {state['query']}\\nAssistant: {response}")
    
    return {"response": response}

def decide_next_node(state: AgentState):
    # LangGraph ko batata hai ki route ke basis pe agla node kaunsa hoga
    if "WEB" in state["route"]: return "web_node"
    if "DIRECT" in state["route"]: return "generate_node" # Seedha answer do
    return "rag_node"

# Build Graph
# Ek naya state graph banate hain aur usme apne saare functions (nodes) jodte hain
workflow = StateGraph(AgentState)
workflow.add_node("router_node", router_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("web_node", web_node)
workflow.add_node("generate_node", generate_node)

# Entry point: graph humesha router se start hoga
workflow.set_entry_point("router_node")

# Conditional edges banate hain: router ke baad route ke hisaab se kaha jana hai
workflow.add_conditional_edges("router_node", decide_next_node, {
    "rag_node": "rag_node",
    "web_node": "web_node",
    "generate_node": "generate_node"
})

# RAG ya Web ke baad finally generate_node (answer banane) par jana hai
workflow.add_edge("rag_node", "generate_node")
workflow.add_edge("web_node", "generate_node")
# generate_node ke baad kaam khatam (END)
workflow.add_edge("generate_node", END)

# Graph ko compile karke executable banate hain
app_workflow = workflow.compile()

def process_chat(query: str, user_id: str, session_id: str) -> dict:
    # User ke query ko accept karta hai, save karta hai aur workflow start karta hai
    save_chat_message(session_id, user_id, "user", query)
    
    # Initial state (input data) graph ko de rahe hain
    initial_state = {"query": query, "user_id": user_id, "session_id": session_id, "context": "", "source_used": "Direct"}
    
    # Graph ko invoke (start) kar rahe hain
    result = app_workflow.invoke(initial_state)
    
    # Diagnostics UI pe dikhane ke liye log string banate hain
    diagnostics = f"LangGraph Route: {result.get('route', 'Direct')} Node | "
    diagnostics += f"Source: {result.get('source_used', 'None')} | "
    diagnostics += "Memory & Knowledge Graph Updated"
    
    return {"response": result["response"], "source_used": result["source_used"], "diagnostics": diagnostics}

# API Endpoints
# Pydantic model request payload ka format define karne ke liye (FastAPI requirement)
class ChatRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    session_id: str = "default_session"

# /api/chat endpoint banaya jo process_chat function ko call karega POST request aane pe
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return process_chat(req.query, req.user_id, req.session_id)

# /api/memory endpoint user ki long-term memory dekhne ke liye
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
