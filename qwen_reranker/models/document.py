from typing import TypedDict


class Chunk(TypedDict):
    content: str


class ChunkWithScore(Chunk):
    reranker_score: float | None = None
