from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

from src.types.document import DocumentChunk

VectorT = TypeVar("VectorT")


@runtime_checkable
class Embedder[VectorT](Protocol):
    """
    Embedder contract.
    """

    def embed_text(self, texts: list[str]) -> list[VectorT]:
        """
        Embed a batch of document texts.

        :param texts: Document texts to embed.
        :return: Document-side embeddings in input order.
        """
        ...

    def embed_query(self, text: str) -> VectorT:
        """
        Embed a single query string.

        :param text: Query text to embed.
        :return: Query-side embedding.
        """
        ...


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
