from typing import Protocol, runtime_checkable

from src.types.document import DocumentChunk
from src.types.search_strategy import SearchStrategy


@runtime_checkable
class Retriever(Protocol):
    """
    Retriever contract.
    """

    def search(
        self,
        query: str,
        top_k: int,
        top_kr: int,
        strategy: SearchStrategy,
    ) -> list[DocumentChunk]:
        """
        Retrieve and rerank chunks for the given query.

        :param query: User query string.
        :param top_k: Number of candidates to pull from the vector store.
        :param top_kr: Number of candidates to keep after reranking.
        :param strategy: Strategy selector.
        :return: Reranked chunks.
        """
        ...
