"""
Ingestion pipeline:

    WhatsApp metadata-aware loader
    -> delete_by_chat_id (Delete existing points for this chat_id before re-ingest)
    -> group_into_sessions (Pass 1: hard time-gap session grouping)
    -> split_sessions_semantically (Pass 2: message-turn semantic sub-chunking with hard token cap & chat metadata)
     -> QdrantVectorStore.from_documents (LangChain vector store)
"""

from typing import Dict, List, Optional
import re

from loaders.whatsapp_loader import load_whatsapp_messages
from splitters.semantic_splitter import group_into_sessions, split_sessions_semantically
from vectorstore.qdrant_store import delete_by_chat_id, get_embeddings, build_vectorstore_from_documents


def _slugify(text: str) -> str:
    """Convert contact name or filename into a clean chat_id slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_") or "chat"


def ingest_whatsapp_export(
    file_path: str,
    chat_id: Optional[str] = None,
    display_name: Optional[str] = None,
) -> Dict[str, object]:
    """
    Ingest a WhatsApp .txt export into Qdrant with explicit chat_id metadata isolation.
    If chat_id is provided, existing points with that chat_id are deleted first.
    Returns metadata dict for registry storage.
    """
    print(f"[1/5] Loading WhatsApp messages: {file_path}")
    message_docs = load_whatsapp_messages(file_path)
    print(f"      -> {len(message_docs)} messages")

    known_senders = sorted({d.metadata["sender"] for d in message_docs})
    other_senders = [s for s in known_senders if s.lower() != "you"]
    primary_sender = other_senders[0] if other_senders else (known_senders[0] if known_senders else "Unknown")

    resolved_display_name = display_name or primary_sender
    resolved_chat_id = chat_id or _slugify(resolved_display_name)

    print(f"[2/5] Cleaning existing vector data for chat_id='{resolved_chat_id}'")
    delete_by_chat_id(resolved_chat_id)

    print("[3/5] Grouping into time-based sessions (Pass 1)")
    sessions = group_into_sessions(message_docs)
    print(f"      -> {len(sessions)} sessions")

    print("[4/5] Message-turn semantic sub-chunking with token cap and chat metadata (Pass 2)")
    embeddings = get_embeddings()
    chunks = split_sessions_semantically(
        sessions,
        embeddings,
        chat_id=resolved_chat_id,
        chat_display_name=resolved_display_name,
    )
    print(f"      -> {len(chunks)} final chunks")

    print("[5/5] Embedding + storing chunks in Qdrant via QdrantVectorStore")
    build_vectorstore_from_documents(chunks, embeddings)
    print("Ingestion complete.")

    return {
        "chat_id": resolved_chat_id,
        "display_name": resolved_display_name,
        "senders": known_senders,
        "chunk_count": len(chunks),
        "message_count": len(message_docs),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.ingest <path_to_whatsapp_export.txt>")
        sys.exit(1)

    res = ingest_whatsapp_export(sys.argv[1])
    print(f"Ingested result: {res}")
                                                                                            