import argparse

from hack_optimization import RetrieveStage

from cmd.overrides import (
    apply_embedding_overrides,
    apply_ingestion_overrides,
    apply_llm_overrides,
    apply_rag_overrides,
    apply_retrieval_overrides,
)
from cmd.questions import iter_questions
from src.clients import (
    FastEmbedE5DenseEmbedder,
    FastEmbedSparseEmbedder,
    LocalHybridQdrantStore,
    OllamaLLM,
)
from src.config import (
    get_embedding_settings,
    get_ingestion_settings,
    get_llm_settings,
    get_rag_settings,
    get_retrieval_settings,
)
from src.retrieval import MultiQueryExpander
from src.types.search_strategy import SearchStrategy


def run(args: argparse.Namespace) -> None:
    """
    Run the recall stage: pull candidate chunks for each question into JSONL.

    :param args: Parsed CLI namespace.
    """
    ingestion = apply_ingestion_overrides(args, get_ingestion_settings())
    embedding = apply_embedding_overrides(args, get_embedding_settings())
    retrieval = apply_retrieval_overrides(args, get_retrieval_settings())
    rag = apply_rag_overrides(args, get_rag_settings())

    dense = FastEmbedE5DenseEmbedder(embedding.dense_model, embedding.use_cuda, embedding.cache_dir)
    sparse = FastEmbedSparseEmbedder(embedding.sparse_model, embedding.use_cuda, embedding.cache_dir)
    store = LocalHybridQdrantStore(
        path=ingestion.qdrant_path,
        dense=dense,
        sparse=sparse,
        retrieval=retrieval,
    )

    expander: MultiQueryExpander | None = None
    if rag.strategy == SearchStrategy.MULTIQUERY:
        llm_cfg = apply_llm_overrides(args, get_llm_settings())
        llm = OllamaLLM(
            llm_cfg.model_name,
            llm_cfg.base_url,
            llm_cfg.temperature,
            top_p=llm_cfg.top_p,
            top_k=llm_cfg.top_k,
        )
        expander = MultiQueryExpander(llm=llm)

    stage = RetrieveStage(
        store=store,
        collection=ingestion.collection_name,
        top_k=rag.top_k,
        strategy=rag.strategy,
        query_expander=expander,
    )
    stage.run(iter_questions(args.questions, args.limit), args.output)
