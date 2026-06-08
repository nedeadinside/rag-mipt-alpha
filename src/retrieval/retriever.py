from src.types.document import DocumentChunk
from src.types.query_expander import QueryExpanderProtocol
from src.types.reranker import Reranker
from src.types.search_strategy import SearchStrategy
from src.types.vector_store import VectorStore


class HybridRetriever:
    """
    Hybrid retriever.
    """

    def __init__(
        self,
        store: VectorStore,
        reranker: Reranker,
        collection: str,
        query_expander: QueryExpanderProtocol | None = None,
    ) -> None:
        """
        Initialize the retriever.

        :param store: Vector store used for first-stage recall.
        :param reranker: Reranker used for second-stage scoring.
        :param collection: Collection name to search.
        :param query_expander: Query expander used by the multiquery strategy.
        """
        self._store = store
        self._reranker = reranker
        self._collection = collection
        self._expander = query_expander

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
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_kr <= 0:
            raise ValueError("top_kr must be positive")
        if top_kr > top_k:
            raise ValueError("top_kr must be <= top_k")

        match strategy:
            case SearchStrategy.DEFAULT:
                candidates = self._store.search(self._collection, query, top_k)
                return self._reranker.rerank(query, candidates, top_kr)
            case SearchStrategy.MULTIQUERY:
                if self._expander is None:
                    raise ValueError("MULTIQUERY strategy requires a query_expander")
                queries = self._expander.expand(query)
                candidates = self._store.search_multiquery(
                    self._collection, queries, top_k
                )
                return self._reranker.rerank(query, candidates, top_kr)
