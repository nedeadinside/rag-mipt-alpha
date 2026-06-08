import argparse

from src.types.search_strategy import SearchStrategy


def add_ingestion_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register optional overrides for ingestion settings.

    :param parser: Subparser receiving the flags.
    """
    group = parser.add_argument_group("ingestion overrides")
    group.add_argument("--collection-name", type=str, default=None)
    group.add_argument("--qdrant-path", type=str, default=None)
    group.add_argument("--websites-csv-path", type=str, default=None)
    group.add_argument("--upsert-batch-size", type=int, default=None)


def add_chunking_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register optional overrides for chunking settings.

    :param parser: Subparser receiving the flags.
    """
    group = parser.add_argument_group("chunking overrides")
    group.add_argument("--chunk-size", type=int, default=None)
    group.add_argument("--threshold", type=float, default=None)
    group.add_argument("--chunker-embedding-model", type=str, default=None)
    group.add_argument("--overlap-size", type=float, default=None)
    group.add_argument(
        "--overlap-method",
        type=str,
        choices=("suffix", "prefix", "justified"),
        default=None,
    )


def add_embedding_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register optional overrides for embedding settings.

    :param parser: Subparser receiving the flags.
    """
    group = parser.add_argument_group("embedding overrides")
    group.add_argument("--dense-model", type=str, default=None)
    group.add_argument("--sparse-model", type=str, default=None)
    group.add_argument("--cache-dir", type=str, default=None)


def add_retrieval_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register optional overrides for retrieval settings.

    :param parser: Subparser receiving the flags.
    """
    group = parser.add_argument_group("retrieval overrides")
    group.add_argument("--prefetch-limit", type=int, default=None)
    group.add_argument("--reranker-model", type=str, default=None)
    group.add_argument("--reranker-device", type=str, default=None)


def add_llm_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register optional overrides for LLM settings.

    :param parser: Subparser receiving the flags.
    """
    group = parser.add_argument_group("llm overrides")
    group.add_argument("--llm-model-name", type=str, default=None)
    group.add_argument("--llm-base-url", type=str, default=None)
    group.add_argument("--llm-temperature", type=float, default=None)
    group.add_argument("--llm-top-p", type=float, default=None)
    group.add_argument("--llm-top-k", type=int, default=None)


def add_rag_flags(parser: argparse.ArgumentParser) -> None:
    """
    Register optional overrides for RAG pipeline settings.

    :param parser: Subparser receiving the flags.
    """
    group = parser.add_argument_group("rag overrides")
    group.add_argument("--top-k", type=int, default=None)
    group.add_argument("--top-kr", type=int, default=None)
    group.add_argument(
        "--strategy",
        type=SearchStrategy,
        choices=list(SearchStrategy),
        default=None,
    )
    group.add_argument("--prompt-name", type=str, default=None)
