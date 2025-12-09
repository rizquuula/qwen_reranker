from unittest.mock import MagicMock

import pytest

from qwen_reranker.models.document import Chunk
from qwen_reranker.reranker.qwen import QwenReranker


class FakeResponse:
    """Simulated LLM response for tests."""

    def __init__(self, text="yes", yes_logprob=-0.1, no_logprob=-2.0):
        self.content = text
        self.response_metadata = {
            "logprobs": {
                "content": [
                    {
                        "top_logprobs": [
                            {"token": "yes", "logprob": yes_logprob},
                            {"token": "no", "logprob": no_logprob},
                        ]
                    }
                ]
            }
        }


@pytest.fixture
def reranker():
    """Create a QwenReranker instance with mocked LLM."""
    r = QwenReranker(base_url="http://localhost", model="dummy", api_key="fake")
    r._reranker_llm = MagicMock()  # Stub actual LLM calls
    return r


def test_invoke_reranker_yes(reranker: QwenReranker):
    # Arrange
    query = "What is the capital of France?"
    document: Chunk = {"content": "Paris is the capital city of France."}

    fake_response = FakeResponse(text="yes", yes_logprob=-0.05, no_logprob=-3.0)
    reranker._reranker_llm.invoke.return_value = fake_response

    # Act
    result = reranker.invoke_reranker(query, document)

    # Assert
    assert "reranker_score" in result
    assert result["reranker_score"] > 0.5  # higher confidence for yes
    assert result["content"] == document["content"]


def test_invoke_reranker_no(reranker: QwenReranker):
    # Arrange
    query = "What is the capital of France?"
    document: Chunk = {"content": "Bananas are yellow."}

    fake_response = FakeResponse(text="no", yes_logprob=-5.0, no_logprob=-0.02)
    reranker._reranker_llm.invoke.return_value = fake_response

    # Act
    result = reranker.invoke_reranker(query, document)

    # Assert
    assert "reranker_score" in result
    assert result["reranker_score"] < 0.5  # confidence for no


def test_invoke_reranker_fallback_no_logprobs(reranker: QwenReranker):
    """Test behavior if no logprobs exist and fallback is used."""
    query = "What is the capital of France?"
    document: Chunk = {"content": "Paris is the capital."}

    fake_response = FakeResponse(text="yes")
    fake_response.response_metadata = {}  # Remove logprobs
    reranker._reranker_llm.invoke.return_value = fake_response

    # Act
    result = reranker.invoke_reranker(query, document)

    # Assert
    assert result["reranker_score"] == 1.0  # fallback to yes
