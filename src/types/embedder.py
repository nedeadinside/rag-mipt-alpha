from typing import Protocol, runtime_checkable


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

    def embed_queries(self, queries: list[str]) -> list[VectorT]:
        """
        Embed a batch of query list.

        :param queries: Queries to embed.
        :return: Query-side embeddings in input order.
        """
        ...
