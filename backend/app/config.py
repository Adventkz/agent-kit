"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    model: str = field(default_factory=lambda: _env("MODEL", "gpt-4.1-mini"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", "./data")).resolve()
    )
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o for o in _env("CORS_ORIGINS", "http://localhost:5173").split(",") if o
        )
    )
    max_tool_iterations: int = field(
        default_factory=lambda: int(_env("MAX_TOOL_ITERATIONS", "6"))
    )
    system_prompt: str = field(
        default_factory=lambda: _env(
            "SYSTEM_PROMPT",
            "You are a helpful assistant built for a hackathon demo. "
            "Answer in the language the user writes in (Kazakh, Russian or English). "
            "When the user asks about uploaded documents, call search_docs first and "
            "cite the source file names you used.",
        )
    )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
