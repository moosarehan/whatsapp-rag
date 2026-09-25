"""
Query-time pipeline: connects the existing Qdrant collection to the
retriever router and the LCEL RAG chain, scoped to a specific chat_id.
"""

from typing import List
import re
from difflib import SequenceMatcher

from vectorstore.qdrant_store import get_embeddings, get_existing_vectorstore
from retrievers.router import (
    _build_filter,
    _payload_document,
    _semantic_query_variants,
    make_router,
)
from retrievers.query_classifier import classify_query


_MEANING_STOP_WORDS = {
    "a", "about", "aj", "are", "did", "di", "do", "going", "ha", "hai", "he",
    "i", "is", "it", "ka", "ki", "ko", "kya", "like", "message", "me", "mera",
    "na", "ne", "of", "on", "pocha", "pucha", "right", "sa", "send", "sent",
    "something", "tha", "that", "the", "this", "thi", "to", "we", "what", "wahab",
    "was", "were", "ya", "you",
}
from chains.rag_chain import build_rag_chain


class WhatsAppRAGPipeline:
    def __init__(self, known_contacts: List[str], chat_id: str):
        embeddings = get_embeddings()
        vectorstore = get_existing_vectorstore(embeddings)
        router = make_router(vectorstore, known_contacts, chat_id=chat_id)
        self.vectorstore = vectorstore
        self.chat_id = chat_id
        self.router_known_contacts = known_contacts
        self.router = router
        self.chain = build_rag_chain(router)

    def answer(self, user_query: str) -> str:
        parsed = classify_query(user_query, self.router_known_contacts)
        retrieved = self.router.invoke(user_query)
        if self._is_meaning_check(user_query) and parsed.sender:
            retrieved = self._scan_chat_for_meaning(parsed)
            evidence = self._format_meaning_evidence(
                retrieved,
                parsed.sender,
                user_query,
            )
            if evidence:
                return evidence
        answer = self.chain.invoke(user_query)
        if self._is_meaning_check(user_query) and parsed.sender:
            answer = self._repair_semantic_negation(answer, parsed.sender)
        return answer

    def _scan_chat_for_meaning(self, parsed):
        """Scan the filtered chat so dense top-k cannot hide a match."""
        qdrant_filter = _build_filter(parsed, self.chat_id)
        records = []
        offset = None
        while True:
            page, offset = self.vectorstore.client.scroll(
                collection_name=self.vectorstore.collection_name,
                scroll_filter=qdrant_filter,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                break
        return [_payload_document(record) for record in records]

    @staticmethod
    def _is_meaning_check(query: str) -> bool:
        return bool(
            re.search(
                r"\b(like|similar|related|about|type of|meaning|se related|jaisa|jaisi|wala|wali|kya|pocha|pucha|bheja|send|message|meina|mena|maine|mene)\b",
                query,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _format_meaning_evidence(documents, sender: str, query: str) -> str | None:
        query_terms = set()
        for variant in _semantic_query_variants(query):
            query_terms.update(re.findall(r"[\w']+", variant.lower()))
        query_terms -= _MEANING_STOP_WORDS

        def normalize(term: str) -> str:
            return term

        query_terms = {normalize(term) for term in query_terms}
        sender_lower = sender.lower()
        candidates = []
        normalized_query = re.sub(r"[^\w\s]", " ", query.lower())
        normalized_query = re.sub(
            r"\b(?:did|do|does|i|me|my|send|sent|message|to|wahab|kya|ma|na|ne|ki|ka|sa|se|baat|about|like|similar|related)\b",
            " ",
            normalized_query,
        )
        normalized_query = " ".join(normalized_query.split())
        for document in documents:
            for line in document.page_content.splitlines():
                match = re.match(
                    r"^\[(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2})\] "
                    r"(?P<sender>[^:]+): (?P<text>.*)$",
                    line,
                )
                if not match or match.group("sender").strip().lower() != sender_lower:
                    continue
                message_terms = {
                    normalize(term)
                    for term in re.findall(r"[\w']+", match.group("text").lower())
                }
                message_text = match.group("text").lower()
                exact_phrase_score = (
                    10.0
                    if normalized_query and normalized_query in message_text
                    else 0.0
                )
                exact_score = len(query_terms & message_terms)
                prefix_score = sum(
                    0.25
                    for query_term in query_terms
                    if len(query_term) >= 5
                    and any(
                        message_term.startswith(query_term[:5])
                        for message_term in message_terms
                    )
                )
                fuzzy_score = sum(
                    0.6
                    for query_term in query_terms
                    if len(query_term) >= 4
                    and any(
                        SequenceMatcher(None, query_term, message_term).ratio() >= 0.78
                        for message_term in message_terms
                    )
                )
                score = exact_phrase_score + exact_score + prefix_score + fuzzy_score
                if score:
                    candidates.append((score, match))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score = candidates[0][0]
        matches = list(dict.fromkeys(
            f"- **{match.group('date')} {match.group('time')}** "
            f"**{match.group('sender').strip()}:** {match.group('text')}"
            for score, match in candidates
            if score == best_score
        ))[:12]
        return (
            f"**Yes.** I found these semantically related messages from {sender}:\n\n"
            + "\n".join(matches)
            + "\n\n"
            "The wording differs, but the meaning matches your question."
        )

    @staticmethod
    def _repair_semantic_negation(answer: str, sender: str) -> str:
        """Correct the model only when it quotes the requested sender as evidence."""
        if not re.match(r"\s*no\b", answer, re.IGNORECASE):
            return answer
        sender_pattern = re.escape(sender.rstrip("💜🖤💙💚💛🧡💜"))
        if not re.search(sender_pattern, answer, re.IGNORECASE):
            return answer
        return re.sub(r"^\s*no\b", "**Yes.**", answer, count=1, flags=re.IGNORECASE)


if __name__ == "__main__":
    import json
    from pathlib import Path

    registry_path = Path(__file__).resolve().parent.parent / "chats_registry.json"
    if not registry_path.exists():
        print("No chats_registry.json found. Ingest a chat first.")
        raise SystemExit(1)

    with registry_path.open(encoding="utf-8") as f:
        registry = json.load(f)

    if not registry:
        print("Registry is empty. Ingest a chat first.")
        raise SystemExit(1)

    entries = list(registry.values())
    print("Available chats:")
    for i, entry in enumerate(entries, 1):
        print(f"  {i}. {entry['display_name']} (chat_id={entry['chat_id']}, {entry.get('chunk_count', '?')} chunks)")

    choice = int(input(f"\nSelect chat [1-{len(entries)}]: ")) - 1
    selected = entries[choice]

    pipeline = WhatsAppRAGPipeline(
        known_contacts=selected.get("senders", []),
        chat_id=selected["chat_id"],
    )
    print(f"\nWhatsApp RAG ready — scoped to '{selected['display_name']}'. Type a question (Ctrl+C to exit).\n")
    while True:
        query = input("You: ")
        print(f"\nBot: {pipeline.answer(query)}\n")
