import argparse

from cmd.overrides import (
    apply_embedding_overrides,
    apply_ingestion_overrides,
    apply_rag_overrides,
    apply_retrieval_overrides,
)
from cmd.questions import iter_questions
from src.clients import (
    CrossEncoderReranker,
    FastEmbedE5DenseEmbedder,
    FastEmbedSparseEmbedder,
    LocalHybridQdrantStore,
)
from src.config import (
    get_embedding_settings,
    get_ingestion_settings,
    get_rag_settings,
    get_retrieval_settings,
)
from src.rag import ChunkRetrievalStage
from src.retrieval import HybridRetriever


def run(args: argparse.Namespace) -> None:
    """
    Run stage one: retrieve chunks for each question and dump them to JSONL.

    :param args: Parsed CLI namespace.
    """
    ingestion = apply_ingestion_overrides(args, get_ingestion_settings())
    embedding = apply_embedding_overrides(args, get_embedding_settings())
    retrieval = apply_retrieval_overrides(args, get_retrieval_settings())
    rag = apply_rag_overrides(args, get_rag_settings())

    dense = FastEmbedE5DenseEmbedder(embedding.dense_model, embedding.cache_dir)
    sparse = FastEmbedSparseEmbedder(embedding.sparse_model, embedding.cache_dir)
    store = LocalHybridQdrantStore(
        path=ingestion.qdrant_path,
        dense=dense,
        sparse=sparse,
        retrieval=retrieval,
    )
    reranker = CrossEncoderReranker(retrieval.reranker_model, retrieval.reranker_device)
    retriever = HybridRetriever(
        store=store,
        reranker=reranker,
        collection=ingestion.collection_name,
    )

    stage = ChunkRetrievalStage(retriever=retriever, settings=rag)
    stage.run(iter_questions(args.questions, args.limit), args.output)
