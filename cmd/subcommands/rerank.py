import argparse

from hack_optimization import RerankStage

from src.clients import CrossEncoderReranker


def run(args: argparse.Namespace) -> None:
    """
    Run the rerank stage: reorder candidate chunks from a JSONL into a JSONL.

    :param args: Parsed CLI namespace.
    """
    reranker = CrossEncoderReranker(args.reranker_model, args.reranker_device)
    stage = RerankStage(reranker=reranker, top_kr=args.top_kr)
    stage.run(args.input, args.output, batch_size=args.batch_size)
