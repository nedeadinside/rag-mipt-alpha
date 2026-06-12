import logging
from collections.abc import Iterable, Iterator
from itertools import batched

from src.types.chunker import Chunker
from src.types.document import DocumentChunk
from src.types.source import SourceDocument
from src.types.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Ingestion pipeline.
    """

    def __init__(
        self,
        chunker: Chunker,
        store: VectorStore,
        collection_name: str,
        upsert_batch_size: int,
    ) -> None:
        """
        Initialize the pipeline.

        :param chunker: Chunker.
        :param store: Vector store.
        :param collection_name: Target collection for upserts.
        :param upsert_batch_size: Number of chunks written per upsert batch.
        """
        self._chunker = chunker
        self._store = store
        self._collection_name = collection_name
        self._upsert_batch_size = upsert_batch_size

    def run(self, documents: Iterable[SourceDocument]) -> int:
        """
        Ingest the input documents.

        :param documents: Source documents to ingest.
        :return: Number of chunks written.
        """

        docs_seen = 0
        total = 0

        def chunks() -> Iterator[DocumentChunk]:
            nonlocal docs_seen
            for doc in documents:
                docs_seen += 1
                yield from self._chunker.chunk(doc)

        for batch in batched(chunks(), self._upsert_batch_size, strict=False):
            self._store.upsert(self._collection_name, batch)
            total += len(batch)
            logger.info(
                "upserted batch=%d total_chunks=%d docs_seen=%d",
                len(batch),
                total,
                docs_seen,
            )
        logger.info("ingestion done: docs_seen=%d total_chunks=%d", docs_seen, total)
        return total
