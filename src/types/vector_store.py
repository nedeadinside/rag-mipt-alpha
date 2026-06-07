from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from src.types.document import DocumentChunk


@runtime_checkable
class VectorStore(Protocol):
    """
    Vector store contract.
    """

    def upsert(self, collection: str, chunks: Iterable[DocumentChunk]) -> None:
        """
        Write chunks into the collection.

        :param collection: Collection name.
        :param chunks: Chunks to write.
        """
        ...

    def search(self, collection: str, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """
        Search the collection by a query string.

        :param collection: Collection name.
        :param query: Query string.
        :param top_k: Number of results to return.
        :return: Matched chunks.
        """
        ...

    def search_multiquery(
        self, collection: str, queries: list[str], top_k: int = 5
    ) -> list[DocumentChunk]:
        """
        Search the collection by a batch of query strings.

        :param collection: Collection name.
        :param queries: Query strings.
        :param top_k: Number of results to return.
        :return: Matched chunks.
        """
        ...
