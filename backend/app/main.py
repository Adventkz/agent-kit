"""FastAPI entry point: chat over SSE, document upload, health."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm, tools
from .config import settings
from .rag import store

settings.ensure_dirs()

app = FastAPI(title="agent-kit", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.model,
        "tools": sorted(tools.REGISTRY),
        "documents": store.sources(),
        "api_key_configured": bool(settings.openai_api_key),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    history = [m.model_dump() for m in request.messages]

    def generate():
        try:
            for event in llm.run_agent(history):
                yield llm.sse(event)
        except Exception as exc:  # never leave the UI hanging on a dead stream
            yield llm.sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/docs")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    name = file.filename or "upload.txt"

    if name.lower().endswith(".pdf"):
        try:
            import io

            from pypdf import PdfReader

            text = "\n\n".join(
                page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages
            )
        except ImportError as exc:
            raise HTTPException(400, "pypdf is not installed — pip install pypdf") from exc
    else:
        text = raw.decode("utf-8", errors="replace")

    added = store.add(name, text)
    if added == 0:
        raise HTTPException(400, "no readable text found in the file")
    return {"source": name, "chunks": added, "documents": store.sources()}


@app.get("/api/docs")
def documents() -> dict[str, Any]:
    return {"documents": store.sources(), "chunks": len(store.chunks)}


@app.delete("/api/docs")
def clear_documents() -> dict[str, Any]:
    store.clear()
    return {"documents": [], "chunks": 0}


# Single-container deployments (see Dockerfile.prod) copy the built frontend here.
# Mounted last so it never shadows the /api/* routes above.
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
