from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    PointStruct,
    Prefetch,
    ScoredPoint,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.types.document import DocumentChunk
from src.types.embedder import Embedder


class LocalQdrantBase:
    """
    Local Qdrant base.
    """

    _DENSE = "dense"
    _SPARSE = "sparse"

    def __init__(self, path: str) -> None:
        """
        Initialize the client.

        :param path: Storage directory path.
        """
        self._client = QdrantClient(path=path)

    def _exists(self, collection: str) -> bool:
        """
        Check whether the collection exists.

        :param collection: Collection name.
        :return: True if the collection exists.
        """
        return self._client.collection_exists(collection)

    def _assert_exists(self, collection: str) -> None:
        """
        Raise if the collection does not exist.

        :param collection: Collection name.
        """
        if not self._exists(collection):
            raise RuntimeError(f"No collection found for: {collection}")

    def _dense_prefetch(self, vector: list[float], limit: int) -> Prefetch:
        """
        Build a prefetch over the dense vector.

        :param vector: Dense query vector.
        :param limit: Prefetch limit.
        :return: Prefetch.
        """
        return Prefetch(query=vector, using=self._DENSE, limit=limit)

    def _sparse_prefetch(self, vector: SparseVector, limit: int) -> Prefetch:
        """
        Build a prefetch over the sparse vector.

        :param vector: Sparse query vector.
        :param limit: Prefetch limit.
        :return: Prefetch.
        """
        return Prefetch(query=vector, using=self._SPARSE, limit=limit)

    def _rrf_search(
        self,
        collection: str,
        prefetches: list[Prefetch],
        top_k: int,
    ) -> list[DocumentChunk]:
        """
        Run an RRF fusion query over the given prefetches.

        :param collection: Collection name.
        :param prefetches: Prefetch branches to fuse.
        :param top_k: Number of results to return.
        :return: Matched chunks.
        """
        response = self._client.query_points(
            collection_name=collection,
            prefetch=prefetches,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return self._to_chunks(response.points)

    @staticmethod
    def _to_chunks(points: list[ScoredPoint]) -> list[DocumentChunk]:
        """
        Map scored points to chunks.

        :param points: Scored points.
        :return: Chunks with score populated.
        """
        return [DocumentChunk(**p.payload, score=p.score) for p in points]


class LocalQdrantStore(LocalQdrantBase):
    """
    Local on-disk dense vector store.
    """

    def __init__(self, path: str, embedder: Embedder[list[float]]) -> None:
        """
        Initialize the store.

        :param path: Storage directory path.
        :param embedder: Dense embedder.
        """
        super().__init__(path)
        self._embedder = embedder

    def _ensure_collection(self, name: str) -> None:
        """
        Create the collection if missing.

        :param name: Collection name.
        """
        if self._exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                self._DENSE: VectorParams(
                    size=len(self._embedder.embed_text(["probe"])[0]),
                    distance=Distance.COSINE,
                ),
            },
        )

    def upsert(self, collection: str, chunks: Iterable[DocumentChunk]) -> None:
        """
        Write chunks into the collection.

        :param collection: Collection name.
        :param chunks: Chunks to write.
        """
        self._ensure_collection(collection)

        materialized = list(chunks)
        if not materialized:
            return
        vectors = self._embedder.embed_text([c.text for c in materialized])
        points = [
            PointStruct(
                id=c.id,
                vector={self._DENSE: v},
                payload=c.model_dump(exclude={"score"}),
            )
            for c, v in zip(materialized, vectors, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """
        Search the collection by a query string.

        :param collection: Collection name.
        :param query: Query string.
        :param top_k: Number of results to return.
        :return: Matched chunks.
        """
        self._assert_exists(collection)

        vector = self._embedder.embed_query(query)
        response = self._client.query_points(
            collection_name=collection,
            query=vector,
            using=self._DENSE,
            limit=top_k,
            with_payload=True,
        )
        return self._to_chunks(response.points)

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
        self._assert_exists(collection)
        prefetches = [self._dense_prefetch(q, top_k) for q in self._embedder.embed_queries(queries)]
        return self._rrf_search(collection, prefetches, top_k)


class LocalHybridQdrantStore(LocalQdrantBase):
    """
    Local on-disk hybrid vector store.
    """

    def __init__(
        self,
        path: str,
        dense: Embedder[list[float]],
        sparse: Embedder[SparseVector],
        prefetch_limit: int,
    ) -> None:
        """
        Initialize the store.

        :param path: Storage directory path.
        :param dense: Dense embedder.
        :param sparse: Sparse embedder.
        :param prefetch_limit: Per-vector prefetch candidate cap used in fusion search.
        """
        super().__init__(path)
        self._dense = dense
        self._sparse = sparse
        self._prefetch_limit = prefetch_limit

    def _ensure_collection(self, name: str) -> None:
        """
        Create the collection with named dense and sparse vectors if missing.

        :param name: Collection name.
        """
        if self._exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                self._DENSE: VectorParams(
                    size=len(self._dense.embed_text(["probe"])[0]),
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self._SPARSE: SparseVectorParams(modifier=Modifier.IDF),
            },
        )

    def upsert(self, collection: str, chunks: Iterable[DocumentChunk]) -> None:
        """
        Write chunks into the collection with both dense and sparse vectors.

        :param collection: Collection name.
        :param chunks: Chunks to write.
        """
        self._ensure_collection(collection)

        materialized = list(chunks)
        if not materialized:
            return
        texts = [c.text for c in materialized]
        dense_vectors = self._dense.embed_text(texts)
        sparse_vectors = self._sparse.embed_text(texts)
        points = [
            PointStruct(
                id=c.id,
                vector={self._DENSE: dv, self._SPARSE: sv},
                payload=c.model_dump(exclude={"score"}),
            )
            for c, dv, sv in zip(materialized, dense_vectors, sparse_vectors, strict=True)
        ]
        self._client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """
        Search the collection by a query string via RRF fusion of dense and sparse prefetches.

        :param collection: Collection name.
        :param query: Query string.
        :param top_k: Number of results to return.
        :return: Matched chunks.
        """
        self._assert_exists(collection)

        prefetches = [
            self._dense_prefetch(self._dense.embed_query(query), self._prefetch_limit),
            self._sparse_prefetch(self._sparse.embed_query(query), self._prefetch_limit),
        ]
        return self._rrf_search(collection, prefetches, top_k)

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
        self._assert_exists(collection)

        prefetches: list[Prefetch] = []

        embedded_sparse = self._sparse.embed_queries(queries)
        embedded_dense = self._dense.embed_queries(queries)

        for eqs, eqd in zip(embedded_sparse, embedded_dense, strict=True):
            prefetches.append(self._sparse_prefetch(eqs, self._prefetch_limit))
            prefetches.append(self._dense_prefetch(eqd, self._prefetch_limit))

        return self._rrf_search(collection, prefetches, top_k)
