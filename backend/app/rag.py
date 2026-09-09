"""Minimal retrieval: chunk, embed, cosine search, persist to a JSON file.

Deliberately dependency-light. It is fast enough for a few hundred pages,
which is all a hackathon demo ever needs. Swap in pgvector later if you
actually ship the thing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass
class Chunk:
    source: str
    text: str
    embedding: list[float]


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split on paragraph boundaries, then pack into ~`size`-char windows."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= size:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= size:
            buffer = paragraph
            continue
        step = max(size - overlap, 1)
        for start in range(0, len(paragraph), step):
            piece = paragraph[start : start + size]
            if piece:
                chunks.append(piece)
        buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class DocumentStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (settings.data_dir / "documents.json")
        self.chunks: list[Chunk] = []
        self._load()

    # ---------------------------------------------------------------- io
    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.chunks = [Chunk(**c) for c in raw]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------- public
    def sources(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.source not in seen:
                seen.append(chunk.source)
        return seen

    def add(self, source: str, text: str) -> int:
        from .llm import embed

        pieces = chunk_text(text)
        if not pieces:
            return 0
        vectors = embed(pieces)
        self.chunks.extend(
            Chunk(source=source, text=p, embedding=v) for p, v in zip(pieces, vectors)
        )
        self._save()
        return len(pieces)

    def search(self, query: str, top_k: int = 4) -> list[Chunk]:
        if not self.chunks:
            return []
        from .llm import embed

        query_vector = embed([query])[0]
        ranked = sorted(
            self.chunks, key=lambda c: cosine(query_vector, c.embedding), reverse=True
        )
        return ranked[:top_k]

    def clear(self) -> None:
        self.chunks = []
        self._save()


store = DocumentStore()
