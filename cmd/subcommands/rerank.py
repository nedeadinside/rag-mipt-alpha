import argparse

from hack_optimization import RerankStage

from cmd.overrides import apply_rag_overrides, apply_retrieval_overrides
from src.clients import CrossEncoderReranker
from src.config import get_rag_settings, get_retrieval_settings


def run(args: argparse.Namespace) -> None:
    """
    Run the rerank stage: reorder candidate chunks from a JSONL into a JSONL.

    :param args: Parsed CLI namespace.
    """
    retrieval = apply_retrieval_overrides(args, get_retrieval_settings())
    rag = apply_rag_overrides(args, get_rag_settings())

    reranker = CrossEncoderReranker(retrieval.reranker_model, retrieval.reranker_device)
    stage = RerankStage(reranker=reranker, top_kr=rag.top_kr)
    stage.run(args.input, args.output)
