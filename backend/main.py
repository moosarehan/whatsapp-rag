"""
Usage:
    python main.py ingest <path_to_export.txt>
    python main.py chat
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.ingest import ingest_whatsapp_export
from pipeline.query_pipeline import WhatsAppRAGPipeline

CHATS_REGISTRY_PATH = BASE_DIR / "chats_registry.json"


def _load_registry() -> dict:
    if not CHATS_REGISTRY_PATH.exists():
        return {}
    with CHATS_REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    with CHATS_REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def run_ingest(file_path: str):
    from datetime import datetime, timezone

    result = ingest_whatsapp_export(file_path)
    chat_id = result["chat_id"]

    registry = _load_registry()
    registry[chat_id] = {
        "chat_id": chat_id,
        "display_name": result["display_name"],
        "senders": result["senders"],
        "chunk_count": result["chunk_count"],
        "message_count": result["message_count"],
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source_filename": Path(file_path).name,
    }
    _save_registry(registry)
    print(f"Saved chat '{result['display_name']}' (chat_id={chat_id}) to {CHATS_REGISTRY_PATH}")


def run_chat():
    registry = _load_registry()
    if not registry:
        print("No chats found. Run `python main.py ingest <file>` first.")
        sys.exit(1)

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
        try:
            query = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break
        print(f"\nBot: {pipeline.answer(query)}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <path_to_export.txt>")
            sys.exit(1)
        run_ingest(sys.argv[2])
    elif command == "chat":
        run_chat()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
