import argparse

from hack_optimization import RetrieveStage

from cmd.questions import iter_questions
from src.clients import (
    FastEmbedE5DenseEmbedder,
    FastEmbedSparseEmbedder,
    LocalHybridQdrantStore,
    OllamaLLM,
)
from src.retrieval import MultiQueryExpander
from src.types.search_strategy import SearchStrategy


def run(args: argparse.Namespace) -> None:
    """
    Run the recall stage: pull candidate chunks for each question into JSONL.

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

    expander: MultiQueryExpander | None = None
    if args.strategy == SearchStrategy.MULTIQUERY:
        llm = OllamaLLM(
            args.llm_model_name,
            args.llm_base_url,
            args.llm_temperature,
            top_p=args.llm_top_p,
            top_k=args.llm_top_k,
        )
        expander = MultiQueryExpander(llm=llm)

    stage = RetrieveStage(
        store=store,
        collection=args.collection_name,
        top_k=args.top_k,
        strategy=args.strategy,
        query_expander=expander,
    )
    stage.run(iter_questions(args.questions, args.limit), args.output, batch_size=args.batch_size)
