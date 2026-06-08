import argparse
import logging
from pathlib import Path

from cmd.flags import (
    add_chunking_flags,
    add_embedding_flags,
    add_ingestion_flags,
    add_llm_flags,
    add_rag_flags,
    add_retrieval_flags,
)
from cmd.subcommands import generate, ingest, rag, retrieve


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser with all subcommands.

    :return: Configured parser.
    """
    parser = argparse.ArgumentParser(prog="cmd", description="RAG pipeline entry point.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index source documents into the vector store.")
    add_ingestion_flags(p_ingest)
    add_chunking_flags(p_ingest)
    add_embedding_flags(p_ingest)
    add_retrieval_flags(p_ingest)
    p_ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N source docs.",
    )
    p_ingest.set_defaults(func=ingest.run)

    p_retrieve = sub.add_parser(
        "retrieve", help="Stage 1: retrieve chunks per question into JSONL."
    )
    add_ingestion_flags(p_retrieve)
    add_embedding_flags(p_retrieve)
    add_retrieval_flags(p_retrieve)
    add_rag_flags(p_retrieve)
    p_retrieve.add_argument(
        "--questions", type=Path, required=True, help="Input CSV with q_id,query."
    )
    p_retrieve.add_argument("--output", type=Path, required=True, help="Destination JSONL.")
    p_retrieve.add_argument(
        "--limit", type=int, default=None, help="Process only first N questions."
    )
    p_retrieve.set_defaults(func=retrieve.run)

    p_generate = sub.add_parser(
        "generate", help="Stage 2: generate answers from a chunks JSONL."
    )
    add_llm_flags(p_generate)
    add_rag_flags(p_generate)
    p_generate.add_argument("--input", type=Path, required=True, help="JSONL from stage one.")
    p_generate.add_argument(
        "--output", type=Path, required=True, help="Destination submission JSONL."
    )
    p_generate.add_argument("--batch-size", type=int, default=1, help="LLM batch size.")
    p_generate.set_defaults(func=generate.run)

    p_rag = sub.add_parser("rag", help="One-pass end-to-end RAG over a questions CSV.")
    add_ingestion_flags(p_rag)
    add_embedding_flags(p_rag)
    add_retrieval_flags(p_rag)
    add_llm_flags(p_rag)
    add_rag_flags(p_rag)
    p_rag.add_argument("--questions", type=Path, required=True, help="Input CSV with q_id,query.")
    p_rag.add_argument(
        "--output", type=Path, required=True, help="Destination submission JSONL."
    )
    p_rag.add_argument(
        "--limit", type=int, default=None, help="Dev: process only first N questions."
    )
    p_rag.add_argument("--batch-size", type=int, default=1, help="LLM batch size.")
    p_rag.set_defaults(func=rag.run)

    return parser


def main() -> None:
    """
    Parse CLI arguments and dispatch to the selected subcommand.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)
