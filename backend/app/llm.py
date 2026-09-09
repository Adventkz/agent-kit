"""OpenAI access: streaming chat with tool-calling, plus embeddings.

Uses the Chat Completions API because it is the most portable surface —
the same code runs against any OpenAI-compatible gateway, which matters when
a hackathon hands you a proxy URL instead of api.openai.com.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from openai import OpenAI

from . import tools
from .config import settings

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — copy .env.example to .env")
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _client = OpenAI(**kwargs)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    response = client().embeddings.create(
        model=settings.embedding_model, input=texts
    )
    return [item.embedding for item in response.data]


def _merge_tool_call_deltas(
    accumulator: dict[int, dict[str, Any]], deltas: list[Any]
) -> None:
    """Streaming delivers tool calls in fragments; stitch them back together."""
    for delta in deltas:
        entry = accumulator.setdefault(
            delta.index, {"id": "", "name": "", "arguments": ""}
        )
        if delta.id:
            entry["id"] = delta.id
        if delta.function and delta.function.name:
            entry["name"] += delta.function.name
        if delta.function and delta.function.arguments:
            entry["arguments"] += delta.function.arguments


def run_agent(messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Drive the tool-calling loop, yielding events for the UI.

    Event shapes:
      {"type": "token",     "text": "..."}          incremental answer text
      {"type": "tool_call", "name": ..., "arguments": ...}
      {"type": "tool_result", "name": ..., "result": "..."}
      {"type": "done"}
      {"type": "error",     "message": "..."}
    """
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": settings.system_prompt},
        *messages,
    ]

    for _ in range(settings.max_tool_iterations):
        stream = client().chat.completions.create(
            model=settings.model,
            messages=conversation,
            tools=tools.schemas(),
            stream=True,
        )

        content = ""
        pending: dict[int, dict[str, Any]] = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                yield {"type": "token", "text": delta.content}
            if delta.tool_calls:
                _merge_tool_call_deltas(pending, delta.tool_calls)

        if not pending:
            yield {"type": "done"}
            return

        calls = [pending[i] for i in sorted(pending)]
        conversation.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            }
        )

        for call in calls:
            yield {
                "type": "tool_call",
                "name": call["name"],
                "arguments": call["arguments"],
            }
            result = tools.call(call["name"], call["arguments"])
            yield {"type": "tool_result", "name": call["name"], "result": result}
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )

    yield {
        "type": "error",
        "message": f"stopped after {settings.max_tool_iterations} tool iterations",
    }


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
