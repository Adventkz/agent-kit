# agent-kit

A small, honest starting point for building an LLM agent app: streaming chat,
tool calling, and retrieval over documents you upload — in about 700 lines of
readable code, with no framework to learn.

It exists because most "AI app" templates are either a single `curl` example or
a 40-file framework. This is the middle: enough structure that adding a tool or
a data source is a one-file change, little enough that you can read all of it
in ten minutes.

```
┌──────────┐   SSE    ┌───────────────┐   tool calls   ┌───────────┐
│ React UI │ ───────► │ FastAPI agent │ ─────────────► │ tools.py  │
│          │ ◄─────── │  loop (llm.py)│ ◄───────────── │ rag.py    │
└──────────┘  tokens  └───────────────┘    results     └───────────┘
```

## What you get

- **Streaming answers** over Server-Sent Events, tokens rendered as they arrive.
- **Tool calling** with a real loop: the model calls a tool, sees the result,
  and keeps going until it has an answer (bounded by `MAX_TOOL_ITERATIONS`).
- **A visible tool trace** in the UI — every call and its result, collapsible.
  Debugging an agent blind is the main way hackathon demos die.
- **Retrieval** over uploaded `.txt` / `.md` / `.csv` / `.json` / `.pdf`:
  chunking, embeddings, cosine search, persisted to a JSON file.
- **Schema-from-signature**: write a Python function, decorate it, and the JSON
  schema sent to the API is derived from its type hints.
- **Multilingual by default** — the system prompt answers in the language it is
  asked in (Kazakh, Russian, English).

## Quick start

```bash
cp .env.example .env                     # put your OPENAI_API_KEY in it

cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --env-file ../.env   # http://localhost:8000

cd ../frontend
npm install
npm run dev                              # http://localhost:5173
```

`--env-file` is required: unlike the Docker setup below, nothing else loads `.env`
into the process environment for a native run.

With Docker instead:

```bash
docker compose up --build   # http://localhost:5173
```

## Adding a tool

That is the whole workflow:

```python
# backend/app/tools.py

@tool("Look up a company by its BIN in the state registry.")
def lookup_company(bin: str) -> str:
    return httpx.get(f"https://example.gov/api/company/{bin}").text
```

No registration list to update, no schema to hand-write. The parameter types
become the JSON schema; parameters without a default become `required`.

## Adding a data source

`rag.py` exposes `store.add(source, text)`. Anything you can turn into a string
— a scraped page, a database dump, a transcript — can be indexed with one call,
and `search_docs` will find it.

## Layout

```
backend/
  app/config.py   settings from the environment
  app/llm.py      OpenAI client, streaming, the tool-calling loop
  app/tools.py    tool registry + built-in tools
  app/rag.py      chunking, embeddings, cosine search, persistence
  app/main.py     FastAPI routes (chat, docs, health)
  tests/          runs without an API key or network
frontend/
  src/api.ts      fetch + SSE parsing (EventSource can't POST)
  src/App.tsx     chat UI with the tool trace
```

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | required |
| `OPENAI_BASE_URL` | OpenAI | set it when you are given a gateway URL |
| `MODEL` | `gpt-4.1-mini` | any chat-completions model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | |
| `MAX_TOOL_ITERATIONS` | `6` | guard against tool loops |
| `SYSTEM_PROMPT` | see `config.py` | |

The Chat Completions API is used deliberately: the same code runs against any
OpenAI-compatible endpoint, which matters when you are handed a proxy URL
instead of `api.openai.com`.

## Tests

```bash
cd backend && python -m pytest -q
```

They cover the tool registry (schema generation, error handling, no code
execution in `calculator`), chunking and cosine similarity — everything that
does not need the network.

## Кратко по-русски

Стартовый каркас для приложения с LLM-агентом: стриминг ответов, вызов
инструментов, поиск по загруженным документам. `docker compose up --build`,
открыть `http://localhost:5173`. Новый инструмент — это одна функция с
декоратором `@tool` в `backend/app/tools.py`, схема для модели генерируется из
аннотаций типов.

## License

MIT — see [LICENSE](LICENSE).
