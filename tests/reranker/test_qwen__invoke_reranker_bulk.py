from unittest.mock import MagicMock

import pytest

from qwen_reranker.models.document import Chunk
from qwen_reranker.reranker.qwen import QwenReranker


class FakeResponse:
    """Simulated LLM response."""
    def __init__(self, text="yes"):
        self.content = text
        self.response_metadata = {}


@pytest.fixture
def reranker():
    """Create instance with mocked LLM to avoid real API calls."""
    r = QwenReranker(base_url="http://localhost", model="dummy", api_key="fake")
    r._reranker_llm = MagicMock()
    return r


def test_invoke_reranker_bulk_success(reranker: QwenReranker):
    # Arrange
    query = "What is the capital of France?"
    docs: list[Chunk] = [
        {"content": "Paris is the capital."},
        {"content": "Bananas are yellow."},
        {"content": "The capital of France is Paris."},
    ]

    # Mock response order: yes → no → yes
    responses = [
        FakeResponse("yes"),
        FakeResponse("no"),
        FakeResponse("yes")
    ]

    # The mocked invoke will return responses in sequence
    reranker._reranker_llm.invoke.side_effect = responses

    # Act
    results = reranker.invoke_reranker_bulk(query, docs, max_workers=3)

    # Assert order preserved
    assert len(results) == 3
    assert results[0]["reranker_score"] > 0.5  # yes
    assert results[1]["reranker_score"] < 0.5  # no
    assert results[2]["reranker_score"] > 0.5  # yes

    # Ensure content didn't change
    for original, scored in zip(docs, results):
        assert scored["content"] == original["content"]


def test_invoke_reranker_bulk_with_error_fallback(reranker: QwenReranker):
    """Simulate one invocation failing — score should default to 0.0."""
    query = "What is the capital of France?"
    docs: list[Chunk] = [
        {"content": "Paris is capital."},
        {"content": "Bananas are yellow."},
    ]

    # First response is fine, second triggers an error
    reranker._reranker_llm.invoke.side_effect = [
        FakeResponse("yes"),
        Exception("Mock failure"),
    ]

    # Act
    results = reranker.invoke_reranker_bulk(query, docs, max_workers=2)

    # Assert default 0.0 on failure
    assert results[0]["reranker_score"] > 0.5
    assert results[1]["reranker_score"] == 0.0


def test_bulk_preserves_document_order(reranker: QwenReranker):
    """Even with concurrency, results must match original document ordering."""
    query = "Example query."
    docs = [{"content": f"Doc {i}"} for i in range(5)]

    reranker._reranker_llm.invoke.side_effect = [FakeResponse("yes")] * 5

    results = reranker.invoke_reranker_bulk(query, docs, max_workers=5)

    assert [r["content"] for r in results] == [f"Doc {i}" for i in range(5)]