"""
Chunking pipeline (LangChain-native version):

1. Pass 1 (hard time-gap): group_into_sessions merges consecutive per-message
   Documents into conversation sessions based on a time threshold
   (HARD_TIME_GAP_MINUTES).
2. Pass 2 (semantic sub-chunking): split_sessions_semantically splits each
   session at the message-turn level using:
   - Per-message embeddings (via a LangChain Embeddings implementation)
   - Cosine similarity check against the running average vector of the last
     SEMANTIC_WINDOW_SIZE messages, computed with LangChain's built-in
     cosine_similarity util (langchain_community.utils.math)
   - A hard token ceiling (MAX_CHUNK_TOKENS, 400-500 tokens) enforced
     regardless of topic continuity, counted via LangChain's built-in
     tiktoken length function (langchain_text_splitters)
   - Preserves message turn integrity (no mid-message sentence splitting)
   - Accurate slice metadata (senders, date, start_time, end_time, token_count)
   - Optional chat_id / chat_display_name forwarded into every chunk's metadata
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, List, Optional, Union

import numpy as np
import tiktoken
from langchain_community.utils.math import cosine_similarity
from langchain_core.documents import Document

from config import (
    HARD_TIME_GAP_MINUTES,
    MAX_CHUNK_TOKENS,
    MIN_CHUNK_MESSAGES,
    MIN_CHUNK_TOKENS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    SEMANTIC_WINDOW_SIZE,
    WHATSAPP_TIMEZONE,
)

_TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens with the public tiktoken encoder used by LangChain."""
    return len(_TOKEN_ENCODER.encode(text))


def _to_datetime(doc: Document) -> datetime:
    return datetime.strptime(
        f"{doc.metadata['date']} {doc.metadata['time']}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=ZoneInfo(WHATSAPP_TIMEZONE))


def _create_chunk_document(
    messages: List[Document],
    chat_id: Optional[str] = None,
    chat_display_name: Optional[str] = None,
    session_id: Optional[str] = None,
    chunk_index: Optional[int] = None,
) -> Document:
    """Format message turns into one chunk with accurate slice metadata."""
    # Preserve the timestamp on every turn, not merely on the parent chunk.
    # This is essential for exact requests such as "list messages from 4–5pm":
    # a retrieval chunk can overlap a time window while containing turns both
    # inside and outside it.
    text = "\n".join(
        f"[{d.metadata['date']} {d.metadata['time']}] {d.metadata['sender']}: {d.page_content}"
        for d in messages
    )
    senders = sorted({d.metadata["sender"] for d in messages})
    start_dt = _to_datetime(messages[0])
    end_dt = _to_datetime(messages[-1])
    metadata: dict = {
        "senders": senders,
        # Retained for backward-compatible display and exact-date filtering.
        "date": messages[0].metadata["date"],
        "start_time": messages[0].metadata["time"],
        "end_time": messages[-1].metadata["time"],
        "start_datetime": start_dt.isoformat(timespec="minutes"),
        "end_datetime": end_dt.isoformat(timespec="minutes"),
        # Numeric timestamps support correct interval-overlap filtering in Qdrant.
        "start_ts": int(start_dt.timestamp()),
        "end_ts": int(end_dt.timestamp()),
        "source_message_start": messages[0].metadata.get("message_index"),
        "source_message_end": messages[-1].metadata.get("message_index"),
        "message_count": len(messages),
        "token_count": count_tokens(text),
    }
    if chat_id:
        metadata["chat_id"] = chat_id
    if chat_display_name:
        metadata["chat_display_name"] = chat_display_name
    if session_id is not None:
        metadata["session_id"] = session_id
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    return Document(page_content=text, metadata=metadata)


def group_into_sessions(message_docs: List[Document]) -> List[List[Document]]:
    """Group consecutive messages into sessions by the hard time-gap limit."""
    if not message_docs:
        return []

    sessions: List[List[Document]] = [[message_docs[0]]]
    for prev, curr in zip(message_docs, message_docs[1:]):
        gap_minutes = (_to_datetime(curr) - _to_datetime(prev)).total_seconds() / 60.0
        if gap_minutes > HARD_TIME_GAP_MINUTES:
            sessions.append([curr])
        else:
            sessions[-1].append(curr)

    return sessions


def split_sessions_semantically(
    sessions: Union[List[List[Document]], List[Document]],
    embeddings: Any,
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    window_size: int = SEMANTIC_WINDOW_SIZE,
    max_tokens: int = MAX_CHUNK_TOKENS,
    min_chunk_messages: int = MIN_CHUNK_MESSAGES,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    chat_id: Optional[str] = None,
    chat_display_name: Optional[str] = None,
) -> List[Document]:
    """Split sessions at message-turn boundaries using LangChain utilities."""
    if not sessions:
        return []

    if isinstance(sessions[0], Document):
        session_list: List[List[Document]] = group_into_sessions(sessions)  # type: ignore
    else:
        session_list = sessions  # type: ignore

    final_chunks: List[Document] = []

    for session_number, session in enumerate(session_list):
        if not session:
            continue

        session_id = (
            f"{chat_id or 'chat'}:{session_number}:"
            f"{session[0].metadata['date']}T{session[0].metadata['time']}"
        )
        chunk_index = 0

        def emit(messages: List[Document]) -> None:
            nonlocal chunk_index
            final_chunks.append(
                _create_chunk_document(
                    messages, chat_id, chat_display_name, session_id, chunk_index
                )
            )
            chunk_index += 1

        if len(session) == 1:
            emit(session)
            continue

        message_texts = [f"{d.metadata['sender']}: {d.page_content}" for d in session]
        message_tokens = [count_tokens(t) for t in message_texts]
        raw_embeddings = embeddings.embed_documents(message_texts)
        vectors = np.array(raw_embeddings, dtype=np.float32)

        current_chunk_msgs: List[Document] = []
        current_chunk_embs: List[np.ndarray] = []
        current_chunk_tokens = 0

        for doc, text, tokens, vec in zip(session, message_texts, message_tokens, vectors):
            if not current_chunk_msgs:
                current_chunk_msgs.append(doc)
                current_chunk_embs.append(vec)
                current_chunk_tokens = tokens
                continue

            prospective_tokens = current_chunk_tokens + tokens + 1
            if prospective_tokens > max_tokens:
                emit(current_chunk_msgs)
                current_chunk_msgs = [doc]
                current_chunk_embs = [vec]
                current_chunk_tokens = tokens
                continue

            window = np.array(current_chunk_embs[-window_size:])
            running_avg = window.mean(axis=0, keepdims=True)
            similarity = cosine_similarity(vec.reshape(1, -1), running_avg)[0][0]

            # Never turn acknowledgements and question/answer pairs into tiny,
            # independently retrievable documents. Topic shifts only count once
            # there is enough preceding conversational context.
            may_split_semantically = (
                len(current_chunk_msgs) >= min_chunk_messages
                or current_chunk_tokens >= min_chunk_tokens
            )
            if similarity < similarity_threshold and may_split_semantically:
                emit(current_chunk_msgs)
                current_chunk_msgs = [doc]
                current_chunk_embs = [vec]
                current_chunk_tokens = tokens
            else:
                current_chunk_msgs.append(doc)
                current_chunk_embs.append(vec)
                current_chunk_tokens = prospective_tokens

        if current_chunk_msgs:
            emit(current_chunk_msgs)

    return final_chunks
