from abc import ABC, abstractmethod

from models.document import Chunk, ChunkWithScore


class RerankerBase(ABC):
    @abstractmethod
    def invoke_reranker_bulk(self, query: str, docs: list[Chunk], max_workers: int = 1) -> list[ChunkWithScore]:
        raise NotImplementedError()

    @abstractmethod
    def invoke_reranker(self, query: str, doc: Chunk) -> ChunkWithScore:
        raise NotImplementedError()
