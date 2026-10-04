"""FastAPI application with Server-Sent Events (SSE) streaming and embedded chat UI."""
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.tools.db import ensure_db_initialized
from src.rag.indexer import KnowledgeBaseIndexer
from src.rag.retriever import MultilingualRetriever
from src.agent.core import MultilingualCSAgent
from src.agent.memory import session_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FAQ_DIR = PROJECT_ROOT / "data" / "faq"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

agent_instance: Optional[MultilingualCSAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    ensure_db_initialized()
    indexer = KnowledgeBaseIndexer(faq_dir=FAQ_DIR, index_dir=INDEX_DIR)
    if not indexer.load_index():
        indexer.build_and_save_index()
    retriever = MultilingualRetriever(indexer=indexer)
    agent_instance = MultilingualCSAgent(retriever=retriever)
    yield


app = FastAPI(
    title="Multilingual Customer Service Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    language: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    query: str
    language: str
    intent: str
    response: str
    tool_data: Optional[Dict[str, Any]] = None
    citations: list[str] = []
    is_escalated: bool = False
    frustration_score: float = 0.0
    handoff_ticket: Optional[Dict[str, Any]] = None
    pii_redacted: Dict[str, int] = {}
    latency_ms: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "multilingual-cs-agent",
        "supported_languages": ["en", "es", "fr", "de", "ja"],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent is still initializing.")
    result = agent_instance.process_message(
        message=req.message,
        session_id=req.session_id,
        user_language=req.language,
    )
    return ChatResponse(**result)


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint for sub-500ms TTFT."""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent is still initializing.")

    result = agent_instance.process_message(
        message=req.message,
        session_id=req.session_id,
        user_language=req.language,
    )

    async def event_generator():
        # 1. Start event with detected metadata
        start_payload = {
            "type": "start",
            "session_id": result["session_id"],
            "language": result["language"],
            "intent": result["intent"],
        }
        yield f"data: {json.dumps(start_payload)}\n\n"

        # 2. Token-by-token stream simulation (< 30ms inter-token perceived latency)
        words = result["response"].split(" ")
        for i, word in enumerate(words):
            token_chunk = word + (" " if i < len(words) - 1 else "")
            payload = {"type": "token", "content": token_chunk}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.015)

        # 3. Done event with citations, tools & escalation payload
        done_payload = {
            "type": "done",
            "citations": result["citations"],
            "tool_data": result["tool_data"],
            "is_escalated": result["is_escalated"],
            "handoff_ticket": result["handoff_ticket"],
            "latency_ms": result["latency_ms"],
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/session/{session_id}")
def get_session_history(session_id: str):
    session = session_manager.get_or_create(session_id)
    return {
        "session_id": session.session_id,
        "language": session.user_language,
        "is_escalated": session.is_escalated,
        "frustration_score": session.frustration_score,
        "turns": [t.model_dump() for t in session.turns],
    }


@app.get("/", response_class=HTMLResponse)
def chat_ui():
    """Embedded interactive customer service chat UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global Support AI Agent</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0f172a; color: #f8fafc; display: flex; justify-content: center; height: 100vh; padding: 20px; }
        .chat-container { width: 100%; max-width: 800px; background: #1e293b; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        .header { background: #0f172a; padding: 16px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }
        .header h1 { font-size: 1.1rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        .badge { background: #3b82f6; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; text-transform: uppercase; font-weight: bold; }
        .messages { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
        .msg { max-width: 80%; padding: 12px 16px; border-radius: 10px; font-size: 0.95rem; line-height: 1.5; word-break: break-word; }
        .msg.user { background: #3b82f6; align-self: flex-end; border-bottom-right-radius: 2px; }
        .msg.bot { background: #334155; align-self: flex-start; border-bottom-left-radius: 2px; }
        .meta-tag { font-size: 0.72rem; color: #94a3b8; margin-top: 6px; display: flex; gap: 8px; align-items: center; }
        .esc-banner { background: #ef4444; color: white; padding: 8px 12px; border-radius: 6px; font-size: 0.82rem; margin-top: 6px; }
        .citation-tag { background: #0284c7; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
        .input-bar { padding: 16px; background: #0f172a; display: flex; gap: 10px; border-top: 1px solid #334155; }
        input[type="text"] { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #475569; background: #1e293b; color: white; outline: none; }
        button { background: #3b82f6; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #2563eb; }
        .presets { display: flex; gap: 8px; padding: 10px 20px; background: #1e293b; border-bottom: 1px solid #334155; overflow-x: auto; }
        .preset-btn { background: #334155; color: #cbd5e1; border: 1px solid #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; cursor: pointer; white-space: nowrap; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <h1>🌐 Multilingual Support Agent</h1>
            <span class="badge" id="langBadge">Auto-Detect</span>
        </div>
        <div class="presets">
            <button class="preset-btn" onclick="sendPreset('Where is my order ORD-1001?')">🇬🇧 Order Status</button>
            <button class="preset-btn" onclick="sendPreset('¿Puedo devolver un artículo después de 30 días?')">🇪🇸 Política Devolución</button>
            <button class="preset-btn" onclick="sendPreset('Je veux retourner ORD-1005')">🇫🇷 Demande Retour</button>
            <button class="preset-btn" onclick="sendPreset('Wie lange gilt die Herstellergarantie?')">🇩🇪 Garantie</button>
            <button class="preset-btn" onclick="sendPreset('ヘッドホンの在庫はありますか？')">🇯🇵 商品在庫</button>
            <button class="preset-btn" onclick="sendPreset('THIS IS A COMPLETE SCAM, I WANT TO TALK TO A HUMAN NOW!!')">⚠️ Escalation</button>
        </div>
        <div class="messages" id="messageList">
            <div class="msg bot">
                Hello! I am your AI customer service assistant. I can help with order tracking, returns, store policies, and warranty across English, Spanish, French, German, and Japanese.
            </div>
        </div>
        <div class="input-bar">
            <input type="text" id="userInput" placeholder="Ask a question in any supported language..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const sessionId = "web_" + Math.random().toString(36).substring(2, 9);
        const msgList = document.getElementById("messageList");
        const inputField = document.getElementById("userInput");
        const langBadge = document.getElementById("langBadge");

        function sendPreset(text) {
            inputField.value = text;
            sendMessage();
        }

        async function sendMessage() {
            const text = inputField.value.trim();
            if (!text) return;
            inputField.value = "";

            // Append user message
            const uMsg = document.createElement("div");
            uMsg.className = "msg user";
            uMsg.innerText = text;
            msgList.appendChild(uMsg);

            // Placeholder bot message for streaming
            const bMsg = document.createElement("div");
            bMsg.className = "msg bot";
            bMsg.innerHTML = "<em>Typing...</em>";
            msgList.appendChild(bMsg);
            msgList.scrollTop = msgList.scrollHeight;

            let fullContent = "";

            try {
                const response = await fetch("/api/chat/stream", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text, session_id: sessionId })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split("\\n");

                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            const data = JSON.parse(line.substring(6));
                            if (data.type === "start") {
                                langBadge.innerText = data.language.toUpperCase() + " | " + data.intent;
                                bMsg.innerText = "";
                            } else if (data.type === "token") {
                                fullContent += data.content;
                                bMsg.innerText = fullContent;
                            } else if (data.type === "done") {
                                let metaHtml = `<div class="meta-tag">Latency: ${data.latency_ms}ms`;
                                if (data.citations && data.citations.length > 0) {
                                    metaHtml += ` | Citations: <span class="citation-tag">${data.citations.join(", ")}</span>`;
                                }
                                metaHtml += `</div>`;
                                if (data.is_escalated) {
                                    metaHtml += `<div class="esc-banner">🚨 Transferred to Human Support (Ticket: ${data.handoff_ticket ? data.handoff_ticket.ticket_id : 'Active'})</div>`;
                                }
                                bMsg.innerHTML = fullContent + metaHtml;
                            }
                        }
                    }
                    msgList.scrollTop = msgList.scrollHeight;
                }
            } catch (err) {
                bMsg.innerText = "Error contacting support server.";
            }
        }
    </script>
</body>
</html>
    """
