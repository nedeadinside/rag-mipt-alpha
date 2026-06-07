import logging
from collections.abc import Iterable

from src.config import IngestionSettings
from src.ingestion.types import Chunker
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
        settings: IngestionSettings,
    ) -> None:
        """
        Initialize the pipeline.

        :param chunker: Chunker.
        :param store: Vector store.
        :param settings: Ingestion settings.
        """
        self._chunker = chunker
        self._store = store
        self._settings = settings

    def run(self, documents: Iterable[SourceDocument]) -> int:
        """
        Ingest the input documents.

        :param documents: Source documents to ingest.
        :return: Number of chunks written.
        """
        collection = self._settings.collection_name
        batch_size = self._settings.upsert_batch_size

        buffer: list[DocumentChunk] = []
        docs_seen = 0
        total = 0
        for doc in documents:
            docs_seen += 1
            buffer.extend(self._chunker.chunk(doc))
            while len(buffer) >= batch_size:
                head = buffer[:batch_size]
                buffer = buffer[batch_size:]
                self._store.upsert(collection, head)
                total += len(head)
                logger.info(
                    "upserted batch=%d total_chunks=%d docs_seen=%d",
                    len(head),
                    total,
                    docs_seen,
                )
        if buffer:
            self._store.upsert(collection, buffer)
            total += len(buffer)
            logger.info(
                "upserted batch=%d total_chunks=%d docs_seen=%d (tail)",
                len(buffer),
                total,
                docs_seen,
            )
        logger.info("ingestion done: docs_seen=%d total_chunks=%d", docs_seen, total)
        return total
