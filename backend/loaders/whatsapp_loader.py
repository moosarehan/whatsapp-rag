"""Load WhatsApp exports into metadata-rich LangChain Documents.

The stock LangChain WhatsApp loader returns one cleaned Document and drops
the per-message date, time, and sender fields. The export format therefore
requires this small format-specific parser before the data can enter the
LangChain pipeline.
"""

import re
from datetime import datetime
from typing import List

from langchain_core.documents import Document
from config import WHATSAPP_DAY_FIRST

# WhatsApp's cleaned message lines look like: "Umar Afzal: Are you up?"
# The raw export is parsed directly so date/time metadata is preserved.
MESSAGE_PATTERNS = [
    # iOS format: [9/7/26, 8:50:15 PM] Ahmad Abdullah: Ni ki
    re.compile(
        r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*"
        r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s*"
        r"(?P<sender>[^:]+):\s*(?P<text>.*)$"
    ),
    # Android format: 9/7/26, 8:50 PM - Ahmad Abdullah: Ni ki
    re.compile(
        r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*"
        r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s*-\s*"
        r"(?P<sender>[^:]+):\s*(?P<text>.*)$"
    ),
]


def _normalize_date(raw_date: str) -> str:
    day_first = ("%d/%m/%y", "%d/%m/%Y")
    month_first = ("%m/%d/%y", "%m/%d/%Y")
    formats = (*day_first, *month_first) if WHATSAPP_DAY_FIRST else (*month_first, *day_first)
    for fmt in (*formats, "%Y-%m-%d"):
        try:
            return datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw_date


def _normalize_time(raw_time: str) -> str:
    raw_time = raw_time.strip()
    for fmt in ("%I:%M:%S %p", "%I:%M:%S%p", "%H:%M:%S", "%I:%M %p", "%I:%M%p", "%H:%M"):
        try:
            return datetime.strptime(raw_time, fmt).strftime("%H:%M")
        except ValueError:
            continue
    return raw_time


def load_whatsapp_messages(file_path: str) -> List[Document]:
    """
    Load a WhatsApp export into one LangChain Document per message with
    clean {sender, date, time} metadata.

    Args:
        file_path: path to the raw WhatsApp .txt export

    Returns:
        List[Document], one per message, page_content = message text,
        metadata = {"sender": ..., "date": "YYYY-MM-DD", "time": "HH:MM"}
    """
    # Supports both iOS and Android WhatsApp export formats. This parser is
    # intentionally format-specific because the stock loader drops metadata.
    messages: List[Document] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip("\ufeff\n\r")
            match = None
            for pattern in MESSAGE_PATTERNS:
                m = pattern.match(line)
                if m:
                    match = m
                    break

            if match:
                date = _normalize_date(match.group("date"))
                time = _normalize_time(match.group("time"))
                sender = match.group("sender").strip()
                text = match.group("text").strip()
                messages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "sender": sender,
                            "date": date,
                            "time": time,
                            # Stable source position lets chunks be expanded back
                            # to the exact neighboring conversation turns.
                            "message_index": len(messages),
                        },
                    )
                )
            elif messages:
                # Continuation line of a multi-line message (if not a system/notification line)
                if not (line.startswith("[") or (len(line) > 8 and line[0].isdigit() and "/" in line[:5])):
                    messages[-1].page_content += "\n" + line.strip()

    return messages
