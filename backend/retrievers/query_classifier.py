"""
Lightweight rule-based query classifier.

LangChain doesn't ship a "detect date/time/sender constraints in a sentence"
component, so this stays custom - but it's a small pure function, not a
class, and its only job is to produce a Qdrant filter dict that gets fed
into LangChain's retriever `search_kwargs`.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from dateutil import parser as dateutil_parser

TIME_RANGE_PATTERN = re.compile(
    r"(?:(?:between|from)\s+(?P<start1>\d{1,2}(?::\d{2})?\s*[APap]?[Mm]?)\s+(?:and|to|till|until|-)\s+(?P<end1>\d{1,2}(?::\d{2})?\s*[APap]?[Mm]?))|"
    r"(?:(?P<start2>\d{1,2}(?::\d{2})?\s*[APap][Mm])\s*(?:to|till|until|-)\s*(?P<end2>\d{1,2}(?::\d{2})?\s*[APap][Mm]))",
    re.IGNORECASE,
)

MONTH_PATTERN = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
NUMERIC_DATE_PATTERN = re.compile(r"\b\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}\b")
RELATIVE_DATE_PATTERN = re.compile(r"\b(today|yesterday)\b", re.IGNORECASE)

RECENCY_PATTERNS = re.compile(
    r"\b(last|latest|recent|most recent|final|end|ended)\b",
    re.IGNORECASE,
)


@dataclass
class ParsedQuery:
    query_type: str  # "structured" | "semantic" | "hybrid"
    sender: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    is_recency: bool = False
    raw_query: str = ""


def _extract_time_range(text: str):
    match = TIME_RANGE_PATTERN.search(text)
    if not match:
        return None, None
    s_raw = match.group("start1") or match.group("start2")
    e_raw = match.group("end1") or match.group("end2")
    if not s_raw or not e_raw:
        return None, None
    try:
        start = dateutil_parser.parse(s_raw.strip()).strftime("%H:%M")
        end = dateutil_parser.parse(e_raw.strip()).strftime("%H:%M")
        return start, end
    except (ValueError, OverflowError):
        return None, None


def _extract_date(text: str) -> Optional[str]:
    now = datetime.now()
    rel_match = RELATIVE_DATE_PATTERN.search(text)
    if rel_match:
        rel = rel_match.group(1).lower()
        if rel == "today":
            return now.strftime("%Y-%m-%d")
        elif rel == "yesterday":
            return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    # Only attempt date parsing if there is an explicit date indicator (month name or numeric date pattern)
    if not (MONTH_PATTERN.search(text) or NUMERIC_DATE_PATTERN.search(text)):
        return None

    def1 = datetime(1900, 1, 1)
    def2 = datetime(1901, 2, 2)
    try:
        dt1 = dateutil_parser.parse(text, fuzzy=True, default=def1)
        dt2 = dateutil_parser.parse(text, fuzzy=True, default=def2)
        has_month = dt1.month == dt2.month
        has_day = dt1.day == dt2.day
        has_year = dt1.year == dt2.year

        if not (has_month and has_day):
            return None

        year = dt1.year if has_year else now.year
        return f"{year:04d}-{dt1.month:02d}-{dt1.day:02d}"
    except (ValueError, OverflowError):
        return None


def _extract_sender(text: str, known_contacts: List[str]) -> Optional[str]:
    text_lower = text.lower()
    for contact in known_contacts:
        if contact.lower() == "you":
            # Only match "You" if the user explicitly refers to their own messages,
            # NOT when conversational "you" is addressed to the AI assistant
            if re.search(
                r"\b(my\s+messages?|sent\s+by\s+me|from\s+me|did\s+i\s+send|what\s+did\s+i\s+say|i\s+sent|i\s+send|what\s+i\s+wrote|what\s+i\s+said|i\s+messaged|me\s+and|main|maine|mene|meina|mena|mera\s+message|ma\s+(?:ne|na))\b",
                text_lower,
            ):
                return contact
            continue

        names = [contact.lower()]
        parts = contact.split()
        clean_contact = re.sub(r"[^\w\s]", "", contact).strip()
        names = [contact.lower(), clean_contact.lower()]
        parts = clean_contact.split()
        if len(parts) > 1 and len(parts[0]) > 2:
            names.append(parts[0].lower())
        names = list(dict.fromkeys(n for n in names if n))

        for name in names:
            escaped_name = re.escape(name)
            # A mere name mention is not an authorship constraint. For example,
            # "what did I ask Ahmad?" must retrieve the user's messages, while
            # "what did Ahmad say?" may safely filter to Ahmad's messages.
            authored_patterns = [
                rf"\b(?:from|by)\s+{escaped_name}\b",
                rf"\b{escaped_name}\s+(?:said|wrote|sent|replied|forwarded|forward)\b",
                rf"\b{escaped_name}\s+(?:ne|na)\b",
                rf"\bwhat\s+did\s+{escaped_name}\s+say\b",
                # A named subject is an authorship constraint; a named object
                # ("I asked Ahmad") is deliberately not.
                rf"\b{escaped_name}\s+(?:was|is|did|went|going|call|called|asked|sent)\b",
                rf"\b{escaped_name}(?:'s|s)?\s+messages?\b",
                rf"\bwhere\s+{escaped_name}\s+(?:said|sent|wrote)\b",
                rf"\bdid\s+{escaped_name}\s+(?:say|send|write|tell|forward)\b",
            ]
            if any(re.search(pattern, text_lower) for pattern in authored_patterns):
                return contact

    return None


def classify_query(query: str, known_contacts: List[str]) -> ParsedQuery:
    sender = _extract_sender(query, known_contacts)
    date = _extract_date(query)
    start_time, end_time = _extract_time_range(query)
    is_recency = bool(RECENCY_PATTERNS.search(query))

    has_structure = any([sender, date, start_time, end_time])
    # Routing must not depend on a hardcoded vocabulary. Every unconstrained
    # query is semantic; sender/date/time fields only add metadata filters.
    query_type = "hybrid" if has_structure else "semantic"

    return ParsedQuery(
        query_type=query_type,
        sender=sender,
        date=date,
        start_time=start_time,
        end_time=end_time,
        is_recency=is_recency,
        raw_query=query,
    )
