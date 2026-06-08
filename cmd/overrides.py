import argparse

from src.config import (
    ChunkingSettings,
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    RAGSettings,
    RetrievalSettings,
)


def _collect(args: argparse.Namespace, mapping: dict[str, str]) -> dict[str, object]:
    """
    Pull non-None values from the namespace into a base-field dict.

    :param args: Parsed CLI namespace.
    :param mapping: Map of argparse attribute name to settings field name.
    :return: Dict suitable for pydantic model_copy update.
    """
    return {
        field: getattr(args, attr)
        for attr, field in mapping.items()
        if getattr(args, attr, None) is not None
    }


def apply_ingestion_overrides(
    args: argparse.Namespace, base: IngestionSettings
) -> IngestionSettings:
    """
    Return ingestion settings with CLI overrides merged in.

    :param args: Parsed CLI namespace.
    :param base: Settings loaded from disk.
    :return: Possibly updated settings.
    """
    updates = _collect(
        args,
        {
            "collection_name": "collection_name",
            "qdrant_path": "qdrant_path",
            "websites_csv_path": "websites_csv_path",
            "upsert_batch_size": "upsert_batch_size",
        },
    )
    return base.model_copy(update=updates) if updates else base


def apply_chunking_overrides(
    args: argparse.Namespace, base: ChunkingSettings
) -> ChunkingSettings:
    """
    Return chunking settings with CLI overrides merged in.

    :param args: Parsed CLI namespace.
    :param base: Settings loaded from disk.
    :return: Possibly updated settings.
    """
    updates = _collect(
        args,
        {
            "chunk_size": "chunk_size",
            "threshold": "threshold",
            "chunker_embedding_model": "embedding_model",
            "overlap_size": "overlap_size",
            "overlap_method": "overlap_method",
        },
    )
    return base.model_copy(update=updates) if updates else base


def apply_embedding_overrides(
    args: argparse.Namespace, base: EmbeddingSettings
) -> EmbeddingSettings:
    """
    Return embedding settings with CLI overrides merged in.

    :param args: Parsed CLI namespace.
    :param base: Settings loaded from disk.
    :return: Possibly updated settings.
    """
    updates = _collect(
        args,
        {
            "dense_model": "dense_model",
            "sparse_model": "sparse_model",
            "cache_dir": "cache_dir",
        },
    )
    return base.model_copy(update=updates) if updates else base


def apply_retrieval_overrides(
    args: argparse.Namespace, base: RetrievalSettings
) -> RetrievalSettings:
    """
    Return retrieval settings with CLI overrides merged in.

    :param args: Parsed CLI namespace.
    :param base: Settings loaded from disk.
    :return: Possibly updated settings.
    """
    updates = _collect(
        args,
        {
            "prefetch_limit": "prefetch_limit",
            "reranker_model": "reranker_model",
            "reranker_device": "reranker_device",
        },
    )
    return base.model_copy(update=updates) if updates else base


def apply_llm_overrides(args: argparse.Namespace, base: LLMSettings) -> LLMSettings:
    """
    Return LLM settings with CLI overrides merged in.

    :param args: Parsed CLI namespace.
    :param base: Settings loaded from disk.
    :return: Possibly updated settings.
    """
    updates = _collect(
        args,
        {
            "llm_model_name": "model_name",
            "llm_base_url": "base_url",
            "llm_temperature": "temperature",
            "llm_top_p": "top_p",
            "llm_top_k": "top_k",
        },
    )
    return base.model_copy(update=updates) if updates else base


def apply_rag_overrides(args: argparse.Namespace, base: RAGSettings) -> RAGSettings:
    """
    Return RAG pipeline settings with CLI overrides merged in.

    :param args: Parsed CLI namespace.
    :param base: Settings loaded from disk.
    :return: Possibly updated settings.
    """
    updates = _collect(
        args,
        {
            "top_k": "top_k",
            "top_kr": "top_kr",
            "strategy": "strategy",
            "prompt_name": "prompt_name",
        },
    )
    return base.model_copy(update=updates) if updates else base
