from typing import Protocol, runtime_checkable

from src.types.document import DocumentChunk


@runtime_checkable
class Reranker(Protocol):
    """
    Reranker contract.
    """

    def rerank(
        self,
        query: str,
        documents: list[DocumentChunk],
        top_k: int,
    ) -> list[DocumentChunk]:
        """
        Reorder candidates by relevance to the query.

        :param query: Query string.
        :param documents: Candidate chunks.
        :param top_k: Maximum number of chunks to keep.
        :return: Reranked chunks sorted by score descending.
        """
        ...
