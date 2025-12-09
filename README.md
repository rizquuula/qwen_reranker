# Simple Qwen Reranker Implementation

This project provides a lightweight and modular implementation of a document reranking system using Qwen models through the OpenAI-compatible API. It includes a base reranker interface and a concrete implementation using threaded bulk execution.

## Features

* Abstract base interface for reranker implementations
* Qwen-based reranker using `ChatOpenAI`
* Bulk reranking with optional multithreading
* Logprob-based confidence score extraction
* Fallback scoring when logprobs are unavailable

## Project Structure

```
qwen_reranker/
├── base
│   └── reranker.py
├── models
│   └── document.py
├── reranker
│   └── qwen.py
└── __init__.py
```

## Example Usage

```python
from qwen_reranker.reranker.qwen import QwenReranker

reranker = QwenReranker(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-Reranker-4B",
    api_key="your_api_key",
)

query = "What is Python?"
docs = [
    {"content": "Python is a programming language."},
    {"content": "Bananas are yellow and sweet."}
]

results = reranker.invoke_reranker_bulk(query, docs, max_workers=4)

for r in results:
    print(r)
```

## Requirements

* Python 3.12+
* Dependencies listed in `pyproject.toml`

Core dependencies include:

* `langchain-openai`
* `tqdm`
* `pytest` (optional for testing)

## Installation

Install using pip:

```bash
pip install -e .
```

or for development:

```bash
pip install -r requirements.txt
```

## Design Notes

* `RerankerBase` provides a standard interface for plug-and-play reranker implementation.
* `QwenReranker` handles:

  * message formatting
  * logprob parsing
  * yes/no scoring normalization
* The scoring system outputs a probability between `0.0` and `1.0` for each document.

## License

MIT License
