"""LLM setup and the RAG answer chain.

Supports Anthropic (Claude), OpenAI, or Groq as the generation backend,
selected via LLM_PROVIDER in .env.
"""

from functools import lru_cache
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.retrieval import RetrievedChunk, format_context, retrieve

NO_CONTEXT_NOTE = "(No relevant documentation was found for this specific query.)"

SYSTEM_PROMPT = """You are a friendly, helpful customer support assistant. \
You have access to context retrieved from the company's support \
documentation, shown below. It may be empty or marked as not found if \
nothing relevant matched the user's message.

How to respond, depending on what the user said:

1. Greetings, thanks, small talk, or a vague/general request for help \
(e.g. "hi", "I need help", "can you help me?", "what can you do?") — \
Respond warmly and naturally, like a real support agent would. Briefly \
mention the kinds of things you can help with based on the context topics \
available (if any context is given), and invite them to ask a specific \
question. Never respond to these with "I don't have information about that."

2. A specific factual question:
   - If the context below answers it, answer using ONLY that context — do \
not add outside knowledge. Cite the source name when relevant.
   - If the context below does NOT answer it (or says no documentation was \
found), say so plainly and suggest the user rephrase or contact support. Do \
not guess, and do not fabricate an answer.

Be concise and direct — 2-4 sentences unless the question genuinely needs \
more. Use a friendly, professional support tone.

Context:
{context}"""

# Kept modest on purpose: shorter completions finish faster and support
# answers rarely need more than this to be complete.
MAX_OUTPUT_TOKENS = 600


@lru_cache(maxsize=1)
def get_llm():
    """Build (and cache) the chat model client for the configured provider.

    Cached because constructing the client re-does connection/auth setup on
    every call; the underlying model choice doesn't change during a process's
    lifetime, so building it once is safe and meaningfully faster per query.
    """
    provider = settings.llm_provider

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    if provider == "groq":
        # Groq exposes an OpenAI-compatible API, so the OpenAI chat client
        # works unmodified by pointing it at Groq's base URL. Groq's inference
        # is already very fast (LPU-backed) — this is the lowest-latency
        # option of the three.
        from langchain_openai import ChatOpenAI

        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and add it to your .env file."
            )
        return ChatOpenAI(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            temperature=0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', or 'groq'."
    )


def _history_to_messages(history: list[dict]):
    messages = []
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    return messages


def _build_chain(history: list[dict]):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            *[(m.type, m.content) for m in _history_to_messages(history)],
            ("human", "{question}"),
        ]
    )
    return prompt | get_llm()


def answer_question(query: str, history: list[dict] | None = None) -> dict:
    """Run the full RAG pipeline for one user turn (non-streaming).

    The LLM is always called — retrieval only decides what context it sees.
    A greeting or a vague "I need help" with zero matching chunks still gets
    a real, helpful reply; a specific factual question with zero matching
    chunks gets an honest "I don't have that" instead of a hallucination
    (enforced via the system prompt, not by skipping the call).

    Returns a dict with:
      - "answer": str
      - "sources": list[str] (unique source filenames used)
      - "grounded": bool (True when at least one relevant chunk was found
        and passed to the model as context)
    """
    history = history or []
    chunks = retrieve(query)
    context = format_context(chunks) if chunks else NO_CONTEXT_NOTE

    chain = _build_chain(history)
    response = chain.invoke({"context": context, "question": query})

    sources = sorted({chunk.source for chunk in chunks})
    return {"answer": response.content, "sources": sources, "grounded": bool(chunks)}


def prepare_answer_stream(
    query: str, history: list[dict] | None = None
) -> tuple[list[str], bool, Iterator[str]]:
    """Retrieval-then-stream split for UIs that want to render tokens live.

    Retrieval happens eagerly (it's fast and cached), so the caller gets
    `sources` and `grounded` immediately. The LLM is always streamed — see
    `answer_question` for why the retrieval-gate approach was replaced.

    Returns (sources, grounded, token_generator).
    """
    history = history or []
    chunks: list[RetrievedChunk] = retrieve(query)
    context = format_context(chunks) if chunks else NO_CONTEXT_NOTE
    sources = sorted({chunk.source for chunk in chunks})
    chain = _build_chain(history)

    def _stream() -> Iterator[str]:
        for chunk in chain.stream({"context": context, "question": query}):
            if chunk.content:
                yield chunk.content

    return sources, bool(chunks), _stream()
