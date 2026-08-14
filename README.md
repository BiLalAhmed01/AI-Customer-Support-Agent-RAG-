# AI Customer Support Agent (RAG)

A retrieval-augmented generation (RAG) chatbot that answers customer support
questions grounded in your own documentation (PDF/text/Markdown). Built with
LangChain, ChromaDB for vector storage, and Claude (Anthropic) or OpenAI for
generation. If the knowledge base doesn't contain relevant information, the
bot says so instead of making something up.

## Architecture

```
                        ┌─────────────────────┐
                        │   data/docs/*.pdf    │
                        │   data/docs/*.txt    │
                        │   data/docs/*.md     │
                        └──────────┬───────────┘
                                   │ python -m src.ingest
                                   ▼
                    ┌──────────────────────────────┐
                    │ 1. Load documents             │
                    │ 2. Split into ~1000-char      │
                    │    chunks (150 char overlap)  │
                    │ 3. Embed chunks (local         │
                    │    sentence-transformers)      │
                    │ 4. Store in ChromaDB            │
                    │    (persisted to ./chroma_db)  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Chroma vector      │
                        │   store (on disk)    │
                        └──────────┬───────────┘
                                   │
   User question ──► Streamlit ──►│ similarity_search_with_score(query, k=4)
   (app.py)          chat UI      │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Retrieval filter:              │
                    │ keep chunks with distance       │
                    │ <= RELEVANCE_SCORE_THRESHOLD    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │ No relevant chunks?           │
                    │  → return canned "I don't      │
                    │    know" message, skip LLM     │
                    └──────────────┬───────────────┘
                                   │ relevant chunks found
                                   ▼
                    ┌──────────────────────────────┐
                    │ Build prompt:                  │
                    │  system + retrieved context     │
                    │  + chat history + question      │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ LLM (Claude or GPT via         │
                    │ LangChain chat model)           │
                    │ "answer only from context"      │
                    └──────────────┬───────────────┘
                                   ▼
                          Answer + source list
                          rendered in chat UI
```

**Key design decision:** when retrieval returns zero chunks above the
relevance threshold, the app returns a fixed fallback message **without
calling the LLM at all**. This is a hard guarantee against hallucination on
out-of-scope questions, in addition to the "answer only from context" system
prompt instruction that guards against weak/partial context.

## Project structure

```
.
├── app.py                  # Streamlit chat interface
├── requirements.txt
├── .env.example             # copy to .env and fill in your keys
├── data/docs/               # put your PDF/TXT/MD source documents here
├── chroma_db/                # persisted vector store (created on ingest)
└── src/
    ├── config.py             # settings loaded from .env
    ├── embeddings.py          # shared embedding function
    ├── ingest.py               # CLI script: load, chunk, embed, store
    ├── retrieval.py             # retrieval + relevance filtering
    └── llm.py                    # LLM setup + RAG answer chain
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS/Linux
```

Edit `.env`:

- Set `LLM_PROVIDER` to `anthropic`, `openai`, or `groq`.
- Fill in `ANTHROPIC_API_KEY` (https://console.anthropic.com), `OPENAI_API_KEY`
  (https://platform.openai.com), or `GROQ_API_KEY`, matching your chosen provider.
  Groq is the only one with a genuinely free tier — no card required, get a key
  at https://console.groq.com/keys.
- Defaults are otherwise sensible for local development. Embeddings run
  locally via `sentence-transformers` — no embedding API key needed.

### 3. Add your documents

Drop PDF, `.txt`, or `.md` files into `data/docs/`. A sample FAQ is already
included there so you can try the app immediately.

### 4. Ingest documents into the vector store

```bash
python -m src.ingest
```

Re-run with `--reset` to wipe and rebuild the collection from scratch, e.g.
after editing source documents:

```bash
python -m src.ingest --reset
```

### 5. Run the chat app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Example queries

Using the included sample FAQ (`data/docs/sample_faq.md`):

- "How long does standard shipping take?"
- "Can I return a sale item?"
- "What payment methods do you accept?"
- "Is my product covered if I drop it in water?"
- "What are your support hours?"

Try an out-of-scope question to see the no-hallucination fallback in action:

- "What's your policy on cryptocurrency payments?"
- "Can you help me plan a vacation?"

## Configuration reference (.env)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `groq` |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Any current Claude model ID |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any OpenAI chat model |
| `GROQ_API_KEY` | — | Required if using Groq (free, no billing) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq-hosted model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where the vector store is saved |
| `COLLECTION_NAME` | `support_docs` | Chroma collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `RELEVANCE_SCORE_THRESHOLD` | `0.35` | Max distance for a chunk to be considered relevant (lower = stricter) |
| `CHUNK_SIZE` | `1000` | Characters per chunk during ingestion |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |

### Tuning the relevance threshold

`RELEVANCE_SCORE_THRESHOLD` controls how aggressively the app rejects
irrelevant matches. Chroma returns a distance score (lower = more similar).
If the bot is answering questions it shouldn't (hallucinating from weak
matches), lower the threshold. If it's saying "I don't know" too often on
questions it should be able to answer, raise it slightly and re-test.

## Notes

- Vector embeddings run locally (no API cost, no data sent to a third party
  for embedding) — only the final generation step calls the LLM API.
- Switching `EMBEDDING_MODEL` after documents are already ingested requires
  re-running `python -m src.ingest --reset`, since old vectors are not
  compatible with a different embedding model.
- This app keeps chat history in Streamlit session state only (not
  persisted); refreshing the page starts a new conversation.
