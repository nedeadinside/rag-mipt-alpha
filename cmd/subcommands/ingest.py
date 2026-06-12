import argparse
import logging
from collections.abc import Iterable, Iterator
from itertools import islice

from src.clients import (
    FastEmbedE5DenseEmbedder,
    FastEmbedSparseEmbedder,
    LocalHybridQdrantStore,
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
    Execute the ingestion pipeline using settings from CLI arguments.

    :param args: Parsed CLI namespace.
    """
    dense = FastEmbedE5DenseEmbedder(args.dense_model, args.use_cuda, args.cache_dir)
    sparse = FastEmbedSparseEmbedder(args.sparse_model, args.use_cuda, args.cache_dir)
    store = LocalHybridQdrantStore(
        path=args.qdrant_path,
        dense=dense,
        sparse=sparse,
        prefetch_limit=args.prefetch_limit,
    )
    chunker = SemanticChunker(
        embedding_model=args.chunker_embedding_model,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        min_sentences_per_chunk=args.min_sentences_per_chunk,
        min_characters_per_sentence=args.min_characters_per_sentence,
        skip_window=args.skip_window,
        filter_tolerance=args.filter_tolerance,
        overlap_size=args.overlap_size,
        overlap_method=args.overlap_method,
    )
    pipeline = IngestionPipeline(
        chunker=chunker,
        store=store,
        collection_name=args.collection_name,
        upsert_batch_size=args.upsert_batch_size,
    )

    documents = _take(load_websites(args.websites_csv_path), args.limit)
    total = pipeline.run(documents)
    logger.info("upserted %d chunks into %r", total, args.collection_name)
