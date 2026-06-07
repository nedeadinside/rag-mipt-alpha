import argparse
import logging
from collections.abc import Iterable, Iterator
from itertools import islice

from src.clients import (
    FastEmbedBM25Embedder,
    FastEmbedE5DenseEmbedder,
    LocalHybridQdrantStore,
)
from src.config import (
    get_chunking_settings,
    get_embedding_settings,
    get_ingestion_settings,
    get_retrieval_settings,
)
from src.ingestion import IngestionPipeline, SemanticChunker, load_websites
from src.types.source import SourceDocument

logger = logging.getLogger(__name__)


def _take(documents: Iterable[SourceDocument], limit: int | None) -> Iterator[SourceDocument]:
    """
    Take at most the first N items, or all of them if no limit is given.

    :param documents: Input stream.
    :param limit: Optional item cap.
    :return: Possibly truncated iterator.
    """
    if limit is None:
        yield from documents
        return
    yield from islice(documents, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ingestion pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N source docs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    ingestion = get_ingestion_settings()
    chunking = get_chunking_settings()
    embedding = get_embedding_settings()
    retrieval = get_retrieval_settings()

    dense = FastEmbedE5DenseEmbedder(embedding.dense_model, embedding.cache_dir)
    sparse = FastEmbedBM25Embedder(embedding.sparse_model, embedding.cache_dir)
    store = LocalHybridQdrantStore(
        path=ingestion.qdrant_path,
        dense=dense,
        sparse=sparse,
        retrieval=retrieval,
    )
    chunker = SemanticChunker(chunking)
    pipeline = IngestionPipeline(chunker=chunker, store=store, settings=ingestion)

    documents = _take(load_websites(ingestion.websites_csv_path), args.limit)
    total = pipeline.run(documents)
    logger.info("upserted %d chunks into %r", total, ingestion.collection_name)


if __name__ == "__main__":
    main()
