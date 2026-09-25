"""FastAPI HTTP boundary for the WhatsApp RAG pipeline with per-chat isolation."""

import json
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from pipeline.ingest import ingest_whatsapp_export
from pipeline.query_pipeline import WhatsAppRAGPipeline

CHATS_REGISTRY_PATH = BASE_DIR / "chats_registry.json"

app = FastAPI(title="WhatsRAG API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-chat pipeline cache: { chat_id: WhatsAppRAGPipeline }
_pipelines: dict[str, WhatsAppRAGPipeline] = {}
_pipeline_lock = Lock()
_ingest_lock = Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    chat_id: str = Field(min_length=1)


# ── Registry helpers ─────────────────────────────────────────────────

def _load_registry() -> dict:
    if not CHATS_REGISTRY_PATH.exists():
        return {}
    with CHATS_REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    with CHATS_REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ── Pipeline factory ─────────────────────────────────────────────────

def _get_pipeline(chat_id: str) -> WhatsAppRAGPipeline:
    if chat_id not in _pipelines:
        with _pipeline_lock:
            if chat_id not in _pipelines:
                registry = _load_registry()
                entry = registry.get(chat_id)
                if not entry:
                    raise HTTPException(
                        status_code=409,
                        detail=f"No ingested chat found for chat_id='{chat_id}'. Ingest first.",
                    )
                _pipelines[chat_id] = WhatsAppRAGPipeline(
                    known_contacts=entry.get("senders", []),
                    chat_id=chat_id,
                )
    return _pipelines[chat_id]


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    registry = _load_registry()
    return {"status": "ok", "chat_count": len(registry)}


@app.get("/api/chats")
def get_chats() -> dict:
    """Return all registered chats for the frontend card list."""
    return {"chats": _load_registry()}


# Keep backwards compat — frontend may still call /api/contacts
@app.get("/api/contacts")
def contacts() -> dict:
    registry = _load_registry()
    all_contacts = set()
    for entry in registry.values():
        all_contacts.update(entry.get("senders", []))
    return {"contacts": sorted(all_contacts)}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    global _pipelines

    filename = file.filename or ""
    if not filename.lower().endswith(".txt"):
        raise HTTPException(status_code=415, detail="Only WhatsApp .txt exports are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded export is empty.")

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temporary_file:
            temporary_file.write(contents)
            temporary_path = temporary_file.name

        with _ingest_lock:
            result = await run_in_threadpool(ingest_whatsapp_export, temporary_path)

        chat_id = result["chat_id"]
        registry = _load_registry()
        registry[chat_id] = {
            "chat_id": chat_id,
            "display_name": result["display_name"],
            "senders": result["senders"],
            "chunk_count": result["chunk_count"],
            "message_count": result["message_count"],
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "source_filename": filename,
        }
        _save_registry(registry)

        # Invalidate cached pipeline for this chat_id so next query rebuilds it
        with _pipeline_lock:
            _pipelines.pop(chat_id, None)

        return {
            "message": "Export ingested and embeddings stored in Qdrant.",
            "filename": filename,
            "chat": registry[chat_id],
            "chats": registry,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {error}") from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    try:
        rag_pipeline = await run_in_threadpool(_get_pipeline, request.chat_id)
        answer = await run_in_threadpool(rag_pipeline.answer, request.message)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Chat failed: {error}") from error
