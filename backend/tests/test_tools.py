"""Tests that run without an API key or network access."""

from __future__ import annotations

import json

import pytest

from app import tools
from app.rag import chunk_text, cosine


def test_calculator_evaluates_arithmetic():
    assert tools.call("calculator", json.dumps({"expression": "2 + 2 * 10"})).endswith("22")


def test_calculator_rejects_code_execution():
    result = json.loads(tools.call("calculator", json.dumps({"expression": "__import__('os')"})))
    assert "error" in result


def test_unknown_tool_returns_error_instead_of_raising():
    assert "error" in json.loads(tools.call("does_not_exist", "{}"))


def test_bad_arguments_are_reported_back_to_the_model():
    assert "error" in json.loads(tools.call("calculator", "{not json"))


def test_schema_is_generated_from_the_signature():
    schema = next(s for s in tools.schemas() if s["function"]["name"] == "search_docs")
    params = schema["function"]["parameters"]
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["top_k"]["type"] == "integer"
    assert params["required"] == ["query"]  # top_k has a default


def test_now_is_iso_utc():
    assert tools.call("now", "{}").endswith("+00:00")


@pytest.mark.parametrize("size", [200, 900])
def test_chunking_respects_size_and_keeps_content(size):
    text = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
    chunks = chunk_text(text, size=size, overlap=50)
    assert chunks
    assert all(len(c) <= size for c in chunks)
    assert "Paragraph 9" in " ".join(chunks)


def test_chunking_empty_text():
    assert chunk_text("   \n\n  ") == []


def test_cosine_similarity():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
