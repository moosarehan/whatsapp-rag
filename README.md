# WhatsApp RAG System — LangChain Edition

Same simple (non-agentic, non-CRAG) RAG pipeline as before, rebuilt entirely
on LangChain primitives.

## LangChain components used

| Piece | LangChain component |
|---|---|
| Document model | `langchain_core.documents.Document` with WhatsApp metadata |
| Text splitter | Conversation-aware chunker: inactivity sessions + conservative message-turn semantic sub-chunks + adjacent-chunk retrieval expansion |
| Embedding model | `langchain_huggingface.HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) |
| Vector store | `langchain_qdrant.QdrantVectorStore` (no custom class — used directly) |
| Retriever | `vectorstore.as_retriever(search_kwargs={...})` — a standard `VectorStoreRetriever`, with a Qdrant `Filter` passed in for structured/hybrid queries |
| Prompt | `langchain_core.prompts.ChatPromptTemplate` |
| Model | `langchain_groq.ChatGroq` |
| Chain | LCEL — `RunnableParallel`, `RunnableLambda`, `RunnablePassthrough`, piped into the prompt, model, and `StrOutputParser` |

## What's custom, and why

1. **WhatsApp metadata loading** (`loaders/whatsapp_loader.py`) — the export
   format requires a small regex parser to preserve per-message date/time/
   sender fields in one LangChain Document per message.
2. **Conversation-aware message-turn chunking** (`splitters/semantic_splitter.py`) —
   - **Pass 1 (inactivity sessioning)**: A new session begins only after a >90 minute gap *between consecutive messages*. An active conversation may last for hours without an arbitrary time-based cut.
   - **Pass 2 (conservative semantic sub-chunking)**: Splits at message-turn boundaries using a running embedding average and a token target. A topic shift cannot create a tiny chunk: the preceding chunk must contain at least three messages or 80 tokens.
   - Each chunk receives `session_id`, `chunk_index`, source-message range, and absolute start/end timestamps. Retrieval expands a hit to its adjacent chunks in the same session.
3. **Query classification** (`retrievers/query_classifier.py`) — rule-based
   detection of date/time/sender constraints in the question, used to decide
   which retriever filter to build. This feeds directly into a LangChain
   `VectorStoreRetriever`'s `search_kwargs["filter"]`.

## Architecture

```
raw .txt export
   -> WhatsApp metadata-aware Documents              (loaders/whatsapp_loader.py)
  -> group_into_sessions (Pass 1: inactivity gap) (splitters/semantic_splitter.py)
  -> split_sessions_semantically (Pass 2: turn-level similarity + token cap) (splitters/semantic_splitter.py)
  -> QdrantVectorStore.from_documents            (vectorstore/qdrant_store.py)

query time:
  question
  -> classify_query (structured/semantic/hybrid) (retrievers/query_classifier.py)
  -> dense E5 retrieval + scoped lexical backfill + adjacent-session context expansion (retrievers/router.py)
  -> LCEL chain: {context, question} | prompt | ChatGroq | StrOutputParser
                                                   (chains/rag_chain.py)
```

## Setup

```bash
# 1. Install backend dependencies (from root)
pip install -r requirements.txt

# 2. Configure environment (in root)
cp .env.example .env   # fill in GROQ_API_KEY

# 3. Install frontend dependencies
cd frontend
npm install
cd ..
```

## Running the Application

### Option A: Web Interface (Full Stack)
```bash
# Terminal 1: Backend API (port 8000)
cd backend
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend UI (port 5173)
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser.

### Option B: Terminal CLI
```bash
# Ingest export
python backend/main.py ingest /path/to/WhatsApp_Chat_Export.txt

# Interactive terminal chat
python backend/main.py chat
```

Ask things like:
- `Explain this message: [paste text]`
- `Send me the list of messages between 1am and 2am with Umar Afzal on 11 July`
- `Summarize what Umar and I discussed about money last month`

## Note on retrieval semantics

After changing the chunk schema, re-ingest every chat once. Older vectors do
not have the session identifiers or absolute timestamps needed for context
expansion and interval-overlap filters.

For unambiguous chronology, set `WHATSAPP_DAY_FIRST=true` for day/month
exports (month/day is the compatibility default), and set
`WHATSAPP_TIMEZONE` to the export's IANA time zone when it is not
`Asia/Karachi`.

LangChain's `VectorStoreRetriever` always performs a vector similarity call
under the hood — there's no built-in "pure metadata scroll" retriever for
Qdrant in LangChain. For structured queries, this pipeline still passes the
query through similarity search, but combined with an exact metadata filter
and a large `k` (50) — Qdrant applies the filter first and ranks matches by
similarity, so results are still exact-match, just similarity-ordered rather
than returned in raw scroll order. If you need guaranteed-exhaustive
structured results independent of similarity ranking, you'd drop to the
Qdrant client's native `.scroll()` method directly instead of going through
a LangChain retriever for that branch.
