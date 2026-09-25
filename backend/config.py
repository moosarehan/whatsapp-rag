"""Central configuration for the LangChain-based WhatsApp RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

# Load .env from root or backend
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# Version the collection when changing embedding spaces.  Never mix vectors
# from different models in one collection.
QDRANT_COLLECTION_NAME = "whatsapp_chunks_e5"
LOCAL_QDRANT_PATH = str(BACKEND_DIR / "local_qdrant")

# E5 is multilingual and substantially more reliable for this application's
# English + Roman Urdu WhatsApp exports.  It requires query/passage prefixes;
# vectorstore.qdrant_store applies them through E5Embeddings.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3800"))

# Semantic sub-chunking parameters (Pass 2):
# - Running average window of the last 3-5 messages
# - Cosine similarity threshold for topic boundary detection
# - Hard token ceiling enforced regardless of topic continuity
SEMANTIC_WINDOW_SIZE = 4
# Short chat turns have noisy embeddings.  Only use semantic boundaries when
# the current chunk is already substantial; see the two MIN_* settings below.
SEMANTIC_SIMILARITY_THRESHOLD = 0.30
MIN_CHUNK_MESSAGES = 3
MIN_CHUNK_TOKENS = 80
MAX_CHUNK_TOKENS = 500

# Hard time-gap session boundary (minutes) applied BEFORE semantic splitting,
# same reasoning as before: a multi-hour gap is a new conversation regardless
# of embedding similarity.
HARD_TIME_GAP_MINUTES = 90

TOP_K_SEMANTIC = 12
TOP_K_HYBRID = 16
TOP_K_MEANING = 100
ADJACENT_CHUNKS = 1
MAX_CONTEXT_CHUNKS = 12
LEXICAL_BACKFILL_K = 2

# WhatsApp does not carry a locale in its text export. The historic behavior
# was month/day, so retain that safe compatibility default. Set this to true
# explicitly for day/month exports; avoid relying on a country-level guess.
WHATSAPP_DAY_FIRST = os.getenv("WHATSAPP_DAY_FIRST", "false").lower() in {"1", "true", "yes"}
WHATSAPP_TIMEZONE = os.getenv("WHATSAPP_TIMEZONE", "Asia/Karachi")
