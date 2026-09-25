"""
Vector store setup using langchain_qdrant directly - no custom class.

QdrantVectorStore.from_documents(...) handles collection creation, embedding,
and upsert in one call. get_vectorstore(...) reconnects to an existing
collection for query time. delete_by_chat_id(...) removes all points
tagged with a given chat_id before re-ingestion.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector, Distance, VectorParams

import socket
from threading import Lock

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    LOCAL_QDRANT_PATH,
)


_client_lock = Lock()
_client: QdrantClient | None = None


def _get_qdrant_kwargs() -> dict:
    """Return connection kwargs: uses remote/localhost URL if accessible, otherwise falls back to local embedded path."""
    if QDRANT_URL and not QDRANT_URL.startswith("http://localhost") and not QDRANT_URL.startswith("http://127.0.0.1"):
        return {"url": QDRANT_URL, "api_key": QDRANT_API_KEY}
    if QDRANT_URL:
        try:
            sock = socket.create_connection(("127.0.0.1", 6333), timeout=0.5)
            sock.close()
            return {"url": QDRANT_URL, "api_key": QDRANT_API_KEY}
        except OSError:
            pass
    return {"path": LOCAL_QDRANT_PATH}


def _get_qdrant_client() -> QdrantClient:
    """Return the process-wide client so local Qdrant storage is opened once."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                kwargs = _get_qdrant_kwargs()
                _client = QdrantClient(**kwargs)
    return _client


def _get_vectorstore(embeddings: Embeddings) -> QdrantVectorStore:
    """Build a vector store around the shared client without reopening storage."""
    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )


class E5Embeddings(Embeddings):
    """LangChain adapter enforcing the asymmetric E5 retrieval convention."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(f"query: {text}")


def get_embeddings() -> Embeddings:
    return E5Embeddings()


def delete_by_chat_id(chat_id: str) -> int:
    """
    Delete all Qdrant points whose metadata.chat_id matches the given chat_id.
    Returns the number of points deleted (approximate — Qdrant doesn't always
    surface exact counts on delete).
    """
    client = _get_qdrant_client()

    # Check collection exists before trying to delete
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in existing:
        return 0

    delete_filter = Filter(
        must=[FieldCondition(key="metadata.chat_id", match=MatchValue(value=chat_id))]
    )
    client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=FilterSelector(filter=delete_filter),
    )
    # result is an UpdateResult; we don't have an exact count, return 0 as sentinel
    return 0


def build_vectorstore_from_documents(
    documents: List[Document],
    embeddings: Embeddings,
) -> QdrantVectorStore:
    """Embed and store documents/chunks into a fresh (or existing) Qdrant collection."""
    client = _get_qdrant_client()
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        vector_size = len(embeddings.embed_documents(["dummy_text"])[0])
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    vectorstore = _get_vectorstore(embeddings)
    vectorstore.add_documents(documents)
    return vectorstore


def get_existing_vectorstore(embeddings: Embeddings) -> QdrantVectorStore:
    """Reconnect to an already-populated Qdrant collection (used at query time)."""
    return _get_vectorstore(embeddings)
