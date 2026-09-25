"""
The actual RAG chain, built with LangChain Expression Language (LCEL).

Components used:
- ChatPromptTemplate  : the prompt template
- ChatGroq             : the model
- RunnableLambda/RunnablePassthrough/RunnableParallel : the runnables gluing
  retrieval -> formatting -> prompt -> model -> output parser into one chain
- StrOutputParser      : final output parsing

Flow:
  question (str)
    -> RunnableParallel({"context": router | format_docs, "question": passthrough})
    -> prompt
    -> llm
    -> StrOutputParser
    -> answer (str)
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

import tiktoken

from config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MAX_CONTEXT_TOKENS,
)

_TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a user's WhatsApp chat "
    "history. You will be given retrieved chat excerpts as context. "
    "Answer ONLY using the information in the provided context. "
    "If the context does not contain enough information to answer, say so "
    "clearly instead of guessing. When listing messages for a requested date "
    "or time window, include ONLY turns whose individual bracketed timestamp "
    "falls inside that window. Format each as sender, time, and text."
    "You are an expert AI assistant that answers questions about a user's WhatsApp chat history.\n"
    "You will be given retrieved chat excerpts as context.\n\n"
    "Guidelines:\n"
    "1. Strict Grounding: Answer ONLY using information in the provided context. If the context does not contain enough information, state so clearly.\n"
    "2. Language & Roman Urdu: Understand questions in English, Urdu, and Roman Urdu (e.g. 'jldi aja' / 'jaldi aja' = come quickly, 'mail py bhej dya' = sent on email, 'meet pa' = on Google Meet, 'aya hya' = coming).\n"
    "3. Verification Questions: When asked if someone said something (e.g. 'Was there a message where X said Y?'), start with a direct 'Yes' or 'No', quote the exact message, provide the timestamp and sender, and describe relevant surrounding context (such as links or replies). If there are multiple matching or related messages, list all of them.\n"
    "4. Recency & Last Messages: When asked for the 'last message', 'latest message', or 'last message sent by [sender]', identify the chronologically latest turn in the provided context, stating the sender, timestamp, and exact text.\n"
    "5. Listing Messages: When asked to list messages for a requested date or time window, comprehensively list each individual turn falling within that window (Sender, Time, and Text) without truncating or stopping prematurely.\n"
    "6. Summarization: When asked to summarize, provide a clear, chronological breakdown of key discussion points, shared links, and decisions."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Retrieved chat context:\n\n{context}\n\nUser question: {question}\n\n"
            "Answer the question grounded strictly in the context above.",
        ),
    ]
)


def format_docs(docs: List[Document]) -> str:
    """Turn retrieved Documents into the block of text the prompt expects, capped by token budget."""
    if not docs:
        return "No relevant messages found."
    blocks = []
    total_tokens = 0
    for doc in docs:
        meta = doc.metadata
        senders = ", ".join(meta.get("senders", [meta.get("sender", "")]))
        date = meta.get("date", "")
        start = meta.get("start_time", meta.get("time", ""))
        end = meta.get("end_time", meta.get("time", ""))
        block = f"[{date} {start}-{end} | {senders}]\n{doc.page_content}"
        block_tokens = len(_TOKEN_ENCODER.encode(block))
        if blocks and (total_tokens + block_tokens > MAX_CONTEXT_TOKENS):
            # Respect hard context token ceiling to stay comfortably within Groq TPM limits
            break
        blocks.append(block)
        total_tokens += block_tokens
    return "\n\n---\n\n".join(blocks)


def build_rag_chain(router: RunnableLambda):
    """
    Compose the full LCEL RAG chain.

    Args:
        router: a Runnable that takes a query string and returns List[Document]
                (see retrievers/router.py::make_router)

    Returns:
        A Runnable chain: str (question) -> str (answer)
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        reasoning_effort="low",
    )

    retrieval_and_format = router | RunnableLambda(format_docs)

    rag_chain = (
        RunnableParallel(
            {
                "context": retrieval_and_format,
                "question": RunnablePassthrough(),
            }
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain
