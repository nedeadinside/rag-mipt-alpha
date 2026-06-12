import argparse

from src.config import (
    get_chunking_settings,
    get_embedding_settings,
    get_ingestion_settings,
    get_llm_settings,
    get_rag_settings,
    get_retrieval_settings,
)
from src.types.search_strategy import SearchStrategy


def add_ingestion_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register ingestion flags defaulting to the loaded settings.

    :param parser: Subparser receiving the flags.
    """
    cfg = get_ingestion_settings()
    group = parser.add_argument_group("ingestion options")
    group.add_argument("--collection-name", type=str, default=cfg.collection_name)
    group.add_argument("--qdrant-path", type=str, default=cfg.qdrant_path)
    group.add_argument("--websites-csv-path", type=str, default=cfg.websites_csv_path)
    group.add_argument("--upsert-batch-size", type=int, default=cfg.upsert_batch_size)


def add_chunking_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register chunking flags defaulting to the loaded settings.

    :param parser: Subparser receiving the flags.
    """
    cfg = get_chunking_settings()
    group = parser.add_argument_group("chunking options")
    group.add_argument("--chunk-size", type=int, default=cfg.chunk_size)
    group.add_argument("--threshold", type=float, default=cfg.threshold)
    group.add_argument("--chunker-embedding-model", type=str, default=cfg.embedding_model)
    group.add_argument(
        "--min-sentences-per-chunk", type=int, default=cfg.min_sentences_per_chunk
    )
    group.add_argument(
        "--min-characters-per-sentence", type=int, default=cfg.min_characters_per_sentence
    )
    group.add_argument("--skip-window", type=int, default=cfg.skip_window)
    group.add_argument("--filter-tolerance", type=float, default=cfg.filter_tolerance)
    group.add_argument("--overlap-size", type=float, default=cfg.overlap_size)
    group.add_argument(
        "--overlap-method",
        type=str,
        choices=("suffix", "prefix", "justified"),
        default=cfg.overlap_method,
    )


def add_embedding_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register embedding flags defaulting to the loaded settings.

    :param parser: Subparser receiving the flags.
    """
    cfg = get_embedding_settings()
    group = parser.add_argument_group("embedding options")
    group.add_argument("--dense-model", type=str, default=cfg.dense_model)
    group.add_argument("--sparse-model", type=str, default=cfg.sparse_model)
    group.add_argument("--cache-dir", type=str, default=cfg.cache_dir)
    group.add_argument(
        "--use-cuda", action=argparse.BooleanOptionalAction, default=cfg.use_cuda
    )


def add_retrieval_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register retrieval flags defaulting to the loaded settings.

    :param parser: Subparser receiving the flags.
    """
    cfg = get_retrieval_settings()
    group = parser.add_argument_group("retrieval options")
    group.add_argument("--prefetch-limit", type=int, default=cfg.prefetch_limit)
    group.add_argument("--reranker-model", type=str, default=cfg.reranker_model)
    group.add_argument("--reranker-device", type=str, default=cfg.reranker_device)


def add_llm_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register LLM flags defaulting to the loaded settings.

    :param parser: Subparser receiving the flags.
    """
    cfg = get_llm_settings()
    group = parser.add_argument_group("llm options")
    group.add_argument("--llm-model-name", type=str, default=cfg.model_name)
    group.add_argument("--llm-base-url", type=str, default=cfg.base_url)
    group.add_argument("--llm-temperature", type=float, default=cfg.temperature)
    group.add_argument("--llm-top-p", type=float, default=cfg.top_p)
    group.add_argument("--llm-top-k", type=int, default=cfg.top_k)


def add_rag_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register RAG pipeline flags defaulting to the loaded settings.

    :param parser: Subparser receiving the flags.
    """
    cfg = get_rag_settings()
    group = parser.add_argument_group("rag options")
    group.add_argument("--top-k", type=int, default=cfg.top_k)
    group.add_argument("--top-kr", type=int, default=cfg.top_kr)
    group.add_argument(
        "--strategy",
        type=SearchStrategy,
        choices=list(SearchStrategy),
        default=cfg.strategy,
    )
    group.add_argument("--prompt-name", type=str, default=cfg.prompt_name)
    group.add_argument("--verifier-prompt-name", type=str, default=cfg.verifier_prompt_name)
    group.add_argument("--refusal-text-name", type=str, default=cfg.refusal_text_name)
