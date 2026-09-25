"""
Retriever routing with strict chat_id metadata boundary enforcement.

All queries (recency scroll, structured filter, hybrid similarity, semantic similarity)
enforce a mandatory FieldCondition on metadata.chat_id.
"""

from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from config import (
    ADJACENT_CHUNKS, LEXICAL_BACKFILL_K, MAX_CONTEXT_CHUNKS, TOP_K_MEANING, TOP_K_SEMANTIC, TOP_K_HYBRID,
    GROQ_API_KEY, GROQ_MODEL_NAME, WHATSAPP_TIMEZONE,
)
from retrievers.query_classifier import classify_query, ParsedQuery


def _build_filter(parsed: ParsedQuery, chat_id: str) -> Filter:
    """Build Qdrant filter with mandatory chat_id boundary."""
    conditions = [
        FieldCondition(key="metadata.chat_id", match=MatchValue(value=chat_id))
    ]
    # WhatsApp call notifications are system turns from sender "-", even
    # though the neighboring human turn identifies who initiated the call.
    is_call_query = bool(re.search(r"\b(call|called|phone|ring)\b", parsed.raw_query, re.IGNORECASE))
    if parsed.sender and not is_call_query:
        conditions.append(
            FieldCondition(key="metadata.senders", match=MatchValue(value=parsed.sender))
        )
    if parsed.date:
        timezone = ZoneInfo(WHATSAPP_TIMEZONE)
        day_start = datetime.strptime(parsed.date, "%Y-%m-%d").replace(tzinfo=timezone)
        start = day_start
        end = day_start + timedelta(days=1) - timedelta(seconds=1)
        if parsed.start_time:
            start = datetime.strptime(
                f"{parsed.date} {parsed.start_time}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone)
        if parsed.end_time:
            end = datetime.strptime(
                f"{parsed.date} {parsed.end_time}", "%Y-%m-%d %H:%M"
            ).replace(tzinfo=timezone)
            if parsed.start_time and end <= start:
                # Handle midnight wrap-around (e.g. 11 pm to 12 am / 00:00)
                end += timedelta(days=1)
        # Interval overlap, not containment: retain chunks that began before
        # the requested time but contain messages inside it (and vice versa).
        conditions.extend([
            FieldCondition(key="metadata.start_ts", range=Range(lte=int(end.timestamp()))),
            FieldCondition(key="metadata.end_ts", range=Range(gte=int(start.timestamp()))),
        ])
    return Filter(must=conditions)


def _payload_document(record) -> Document:
    payload = record.payload or {}
    return Document(
        page_content=payload.get("page_content", ""),
        metadata=payload.get("metadata", {}),
    )


_LEXICAL_STOP_WORDS = {
    "about", "ahmad", "ahmed", "asked", "did", "for", "from", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "the", "to", "was", "what", "where",
    "who", "why", "you", "your",
}

def _expand_query_terms(query: str) -> set[str]:
    return set(re.findall(r"[\w']+", query.lower())) - _LEXICAL_STOP_WORDS


def _is_meaning_query(query: str) -> bool:
    return bool(re.search(r"\b(like|similar|related|about|type of|meaning|se related|jaisa|jaisi|wala|wali)\b", query, re.IGNORECASE))


def _semantic_query_variants(query: str) -> List[str]:
    """Create short cross-language search variants without inventing evidence."""
    if not GROQ_API_KEY:
        return [query]
    try:
        expander = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=0,
            max_tokens=160,
            reasoning_effort="low",
        )
        response = expander.invoke(
            "Rewrite this chat-history search request into at most three short "
            "search phrases. Preserve its meaning, names, dates, and intent. "
            "Include an English phrasing and, when relevant, a Roman Urdu "
            "equivalent phrasing using Latin letters only, never Urdu script, and "
            "the likely vocabulary from the original chat. Return only one phrase per line, with "
            "no numbering or explanation. Do not answer the request.\n\n"
            f"Request: {query}"
        )
        variants = [
            line.strip(" -*\t\"")
            for line in response.content.splitlines()
            if line.strip()
            and not line.lower().startswith((
                "i’m sorry", "i'm sorry", "sorry", "i cannot", "i can't",
                "i can’t", "i am unable",
            ))
        ]
        return list(dict.fromkeys([query, *variants[:3]]))
    except Exception:
        return [query]


def _semantic_documents(
    vectorstore: QdrantVectorStore,
    query: str,
    qdrant_filter: Filter,
    limit: int,
) -> List[Document]:
    """Search original and cross-language paraphrases, keeping best hits first."""
    ranked: Dict[Tuple[str, int], Tuple[float, Document]] = {}
    for variant in _semantic_query_variants(query):
        for document, score in vectorstore.similarity_search_with_score(
            variant,
            k=limit,
            filter=qdrant_filter,
        ):
            key = (
                document.metadata.get("session_id", ""),
                document.metadata.get("chunk_index", id(document)),
            )
            current = ranked.get(key)
            if current is None or score > current[0]:
                ranked[key] = (score, document)
    return [
        document
        for _, document in sorted(ranked.values(), key=lambda item: item[0], reverse=True)[:limit]
    ]


def _lexical_backfill(
    vectorstore: QdrantVectorStore, query: str, qdrant_filter: Filter, existing: List[Document]
) -> List[Document]:
    """Add exact-term matches to dense retrieval for names, amounts, and Roman Urdu terms.

    This is deliberately scoped by the same Qdrant metadata filter as dense
    search.  It is a dependable small-chat fallback; at larger scale replace
    it with Qdrant sparse-vector/BM25 retrieval and reciprocal-rank fusion.
    """
    query_terms = _expand_query_terms(query)
    if not query_terms:
        return existing
    records = []
    offset = None
    while True:
        page, offset = vectorstore.client.scroll(
            collection_name=vectorstore.collection_name,
            scroll_filter=qdrant_filter,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            break
    existing_ids = {
        (doc.metadata.get("session_id"), doc.metadata.get("chunk_index")) for doc in existing
    }
    scored = []
    for record in records:
        doc = _payload_document(record)
        doc_id = (doc.metadata.get("session_id"), doc.metadata.get("chunk_index"))
        if doc_id in existing_ids:
            continue
        terms = set(re.findall(r"[\w']+", doc.page_content.lower()))
        score = len(query_terms & terms)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: (item[0], item[1].metadata.get("token_count", 0)), reverse=True)
    return existing + [doc for _, doc in scored[:MAX_CONTEXT_CHUNKS]]


def _exact_date_documents(
    vectorstore: QdrantVectorStore, query: str, qdrant_filter: Filter
) -> List[Document]:
    """Retrieve date-filtered messages by terms, including short URL-only turns."""
    query_terms = _expand_query_terms(query)
    records = []
    offset = None
    while True:
        page, offset = vectorstore.client.scroll(
            collection_name=vectorstore.collection_name,
            scroll_filter=qdrant_filter,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            break

    candidates = []
    for record in records:
        doc = _payload_document(record)
        content_terms = set(re.findall(r"[\w']+", doc.page_content.lower()))
        score = len(query_terms & content_terms)
        if score:
            candidates.append((score, doc))
    candidates.sort(
        key=lambda item: (item[0], item[1].metadata.get("start_ts", 0)),
        reverse=True,
    )
    return [doc for _, doc in candidates[:MAX_CONTEXT_CHUNKS]]


def _call_event_documents(
    vectorstore: QdrantVectorStore, qdrant_filter: Filter
) -> List[Document]:
    """Retrieve WhatsApp system call notifications from a date-filtered scope."""
    records = []
    offset = None
    while True:
        page, offset = vectorstore.client.scroll(
            collection_name=vectorstore.collection_name,
            scroll_filter=qdrant_filter,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            break

    call_docs = [
        _payload_document(record)
        for record in records
        if re.search(r"\[call\]|\bcall\b", record.payload.get("page_content", ""), re.IGNORECASE)
    ]
    call_docs.sort(key=lambda doc: doc.metadata.get("start_ts", 0))
    return call_docs[:MAX_CONTEXT_CHUNKS]


_TURN_TIMESTAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})\] ")


def _keep_only_requested_turns(docs: List[Document], parsed: ParsedQuery) -> List[Document]:
    """For an exact window, send the LLM only individual matching turns.

    Vector filtering operates on chunk intervals, which is intentionally broad
    enough to preserve overlap. This second pass makes list-style answers
    exact after retrieval, now that each stored turn carries a timestamp.
    """
    if not parsed.date or not (parsed.start_time or parsed.end_time):
        return docs
    start = parsed.start_time or "00:00"
    end = parsed.end_time or "23:59"
    if end == "00:00":
        end = "23:59"
    result = []
    for doc in docs:
        lines = []
        for line in doc.page_content.splitlines():
            match = _TURN_TIMESTAMP.match(line)
            if match and match.group(1) == parsed.date and start <= match.group(2) <= end:
                lines.append(line)
        if lines:
            result.append(Document(page_content="\n".join(lines), metadata=doc.metadata))
    return result


def _expand_with_neighbors(
    vectorstore: QdrantVectorStore, docs: List[Document], chat_id: str
) -> List[Document]:
    """Add immediate chunks from the same session so a hit keeps its dialogue context."""
    wanted: Dict[str, set[int]] = {}
    for doc in docs:
        meta = doc.metadata
        session_id, chunk_index = meta.get("session_id"), meta.get("chunk_index")
        if session_id is not None and isinstance(chunk_index, int):
            wanted.setdefault(session_id, set()).update(
                range(max(0, chunk_index - ADJACENT_CHUNKS), chunk_index + ADJACENT_CHUNKS + 1)
            )
    if not wanted:
        return docs  # Compatibility with vectors created before this schema.

    expanded: Dict[Tuple[str, int], Document] = {}
    seed_keys = []
    for doc in docs:
        key = (doc.metadata.get("session_id"), doc.metadata.get("chunk_index"))
        if key not in expanded:
            expanded[key] = doc
            seed_keys.append(key)
    for session_id, indices in wanted.items():
        session_filter = Filter(must=[
            FieldCondition(key="metadata.chat_id", match=MatchValue(value=chat_id)),
            FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id)),
        ])
        records, _ = vectorstore.client.scroll(
            collection_name=vectorstore.collection_name,
            scroll_filter=session_filter,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            doc = _payload_document(record)
            index = doc.metadata.get("chunk_index")
            if index in indices:
                expanded[(session_id, index)] = doc

    seed_docs = [expanded[key] for key in seed_keys if key in expanded]
    neighbor_docs = [doc for key, doc in expanded.items() if key not in seed_keys]
    neighbor_docs.sort(key=lambda d: (d.metadata.get("start_ts", 0), d.metadata.get("chunk_index", 0)))
    return (seed_docs + neighbor_docs)[:MAX_CONTEXT_CHUNKS] or docs


def _get_recent_documents(
    vectorstore: QdrantVectorStore,
    qdrant_filter: Filter,
    limit: int = 4,
) -> List[Document]:
    """
    Fetch the chronologically most recent chunks matching the filter (with chat_id boundary).
    Used for recency-oriented queries ('last message', 'latest', etc.).
    """
    records = []
    offset = None
    try:
        while True:
            page, offset = vectorstore.client.scroll(
                collection_name=vectorstore.collection_name,
                scroll_filter=qdrant_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                break
    except Exception:
        return []

    if not records:
        return []

    def _sort_key(rec):
        meta = rec.payload.get("metadata", {})
        # Prefer numeric UNIX timestamp for strict chronological sorting
        ts = meta.get("end_ts")
        if ts is not None:
            return ts
        return (meta.get("date", ""), meta.get("end_time", ""))

    sorted_records = sorted(records, key=_sort_key, reverse=True)
    recent_docs = [
        Document(
            page_content=r.payload.get("page_content", ""),
            metadata=r.payload.get("metadata", {}),
        )
        for r in sorted_records[:limit]
    ]
    # Return in chronological order (earliest to latest in the recent window)
    return recent_docs[::-1]


def make_router(
    vectorstore: QdrantVectorStore,
    known_contacts: List[str],
    chat_id: str,
) -> RunnableLambda:
    """
    Returns a Runnable that routes queries to the vectorstore while strictly
    scoping all retrievals to the given chat_id.
    """

    def _route(query: str) -> List[Document]:
        parsed = classify_query(query, known_contacts)
        qdrant_filter = _build_filter(parsed, chat_id=chat_id)

        if parsed.is_recency:
            recent_docs = _get_recent_documents(vectorstore, qdrant_filter, limit=4)
            if recent_docs:
                return recent_docs

        if parsed.date and re.search(r"\b(call|called|phone|ring)\b", query, re.IGNORECASE):
            call_docs = _call_event_documents(vectorstore, qdrant_filter)
            if call_docs:
                return call_docs

        # Exact date questions must not depend on dense similarity: URL-only
        # WhatsApp turns are short and routinely rank below conversational text.
        if parsed.date and not parsed.start_time and not parsed.end_time:
            exact_docs = _exact_date_documents(vectorstore, query, qdrant_filter)
            if exact_docs:
                return exact_docs

        if parsed.query_type == "structured":
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 50, "filter": qdrant_filter}
            )
            dense_docs = retriever.invoke(query)
        else:
            dense_docs = _semantic_documents(
                vectorstore,
                query,
                qdrant_filter,
                TOP_K_MEANING if _is_meaning_query(query) else (
                    TOP_K_HYBRID if parsed.query_type == "hybrid" else TOP_K_SEMANTIC
                ),
            )
        merged_docs = _lexical_backfill(vectorstore, query, qdrant_filter, dense_docs)
        if _is_meaning_query(query):
            return merged_docs[:TOP_K_MEANING]
        expanded_docs = _expand_with_neighbors(vectorstore, merged_docs, chat_id)
        return _keep_only_requested_turns(expanded_docs, parsed)

    return RunnableLambda(_route)
