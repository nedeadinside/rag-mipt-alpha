import argparse
import logging
from collections.abc import Iterable, Iterator
from itertools import islice

from cmd.overrides import (
    apply_chunking_overrides,
    apply_embedding_overrides,
    apply_ingestion_overrides,
    apply_retrieval_overrides,
)
from src.clients import (
    FastEmbedE5DenseEmbedder,
    FastEmbedSparseEmbedder,
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


def run(args: argparse.Namespace) -> None:
    """
    Execute the ingestion pipeline using CLI overrides on top of settings.

    :param args: Parsed CLI namespace.
    """
    ingestion = apply_ingestion_overrides(args, get_ingestion_settings())
    chunking = apply_chunking_overrides(args, get_chunking_settings())
    embedding = apply_embedding_overrides(args, get_embedding_settings())
    retrieval = apply_retrieval_overrides(args, get_retrieval_settings())

    dense = FastEmbedE5DenseEmbedder(embedding.dense_model, embedding.use_cuda, embedding.cache_dir)
    sparse = FastEmbedSparseEmbedder(embedding.sparse_model, embedding.use_cuda, embedding.cache_dir)
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
