"""
FastAPI server for the Weather Advisory Bot.

Provides:
- POST /chat  — Main chat endpoint (takes user message, returns advisory response).
- GET  /       — Serves the frontend HTML.
- GET  /health — Health check.

Session management:
- Each browser session gets a unique session_id (generated client-side).
- The server maintains a dict of session_id → message_history.
- Message history is passed to the graph on each invocation so follow-up
  questions have conversation context.
- Sessions are in-memory only — they reset on server restart, as specified.
"""

import os
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

load_dotenv()

from app.graph import advisory_graph  # noqa: E402 (after load_dotenv)
from app.sop_loader import load_sops  # noqa: E402


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Weather Advisory Bot",
    description="A weather-aware safety advisory chatbot backed by LangGraph and SOPs.",
    version="1.0.0",
)

# Allow CORS for frontend (same-origin or local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → list of message dicts
sessions: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    matched_sops: list[dict] = []
    weather_summary: Optional[str] = None
    resolved_location: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the chat frontend."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html"
    )
    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    sops = load_sops()
    return {
        "status": "healthy",
        "sops_loaded": len(sops),
        "sop_ids": [s["id"] for s in sops],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a user message through the advisory graph.

    Flow:
    1. Retrieve or create session message history.
    2. Append user message to history.
    3. Invoke the LangGraph advisory pipeline with full history.
    4. Append bot response to history.
    5. Return response with metadata (matched SOPs, location, etc.).
    """
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = []

    message_history = sessions[session_id]

    # Add user message to history
    message_history.append({"role": "user", "content": user_message})

    try:
        # Invoke the LangGraph pipeline
        result = advisory_graph.invoke({
            "messages": list(message_history),  # Pass a copy
            "current_query": user_message,
            "location_name": None,
            "activities": [],
            "time_context": None,
            "is_weather_query": True,
            "latitude": None,
            "longitude": None,
            "resolved_location": None,
            "weather_data": None,
            "matched_sops": [],
            "response": "",
            "error": None,
        })

        bot_response = result.get("response", "I'm sorry, something went wrong.")

        # Add bot response to session history
        message_history.append({"role": "assistant", "content": bot_response})

        # Build metadata for the response
        matched_sops = result.get("matched_sops", [])
        # Clean matched_sops for JSON serialization (remove condition details to keep response lean)
        clean_sops = [
            {
                "id": s["id"],
                "name": s["name"],
                "severity": s["severity"],
                "category": s["category"],
            }
            for s in matched_sops
        ]

        return ChatResponse(
            response=bot_response,
            session_id=session_id,
            matched_sops=clean_sops,
            resolved_location=result.get("resolved_location"),
        )

    except Exception as e:
        # Don't leave a dangling user message without a response
        error_response = (
            "I encountered an unexpected error processing your request. "
            "Please try again. If the issue persists, the weather service "
            "may be temporarily unavailable."
        )
        message_history.append({"role": "assistant", "content": error_response})

        return ChatResponse(
            response=error_response,
            session_id=session_id,
            matched_sops=[],
        )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session's history."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"Starting Weather Advisory Bot on {host}:{port}")
    print(f"Loaded {len(load_sops())} SOPs")
    print(f"Frontend: http://localhost:{port}/")

    uvicorn.run(app, host=host, port=port)
