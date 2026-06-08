import argparse
import json
import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from cmd.overrides import (
    apply_embedding_overrides,
    apply_ingestion_overrides,
    apply_llm_overrides,
    apply_rag_overrides,
    apply_retrieval_overrides,
)
from cmd.questions import iter_questions
from src.clients import (
    CrossEncoderReranker,
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
from src.rag.utils import build_messages
from src.retrieval import HybridRetriever, MultiQueryExpander
from src.types.document import DocumentChunk
from src.types.search_strategy import SearchStrategy

logger = logging.getLogger(__name__)


def _batched(
    questions: Iterator[tuple[int, str]], size: int
) -> Iterator[list[tuple[int, str]]]:
    """
    Group an iterator of questions into fixed-size chunks.

    :param questions: Iterator of question id and query pairs.
    :param size: Maximum batch size.
    :return: Iterator of non-empty batches.
    """
    while True:
        batch = list(islice(questions, size))
        if not batch:
            return
        yield batch


def run(args: argparse.Namespace) -> None:
    """
    Run the end-to-end RAG pipeline in one pass and write a submission JSONL.

    :param args: Parsed CLI namespace.
    """
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ingestion = apply_ingestion_overrides(args, get_ingestion_settings())
    embedding = apply_embedding_overrides(args, get_embedding_settings())
    retrieval = apply_retrieval_overrides(args, get_retrieval_settings())
    llm_cfg = apply_llm_overrides(args, get_llm_settings())
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
    llm = OllamaLLM(
        llm_cfg.model_name,
        llm_cfg.base_url,
        llm_cfg.temperature,
        top_p=llm_cfg.top_p,
        top_k=llm_cfg.top_k,
    )
    expander = (
        MultiQueryExpander(llm=llm) if rag.strategy == SearchStrategy.MULTIQUERY else None
    )
    retriever = HybridRetriever(
        store=store,
        reranker=reranker,
        collection=ingestion.collection_name,
        query_expander=expander,
    )

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    questions = iter_questions(args.questions, args.limit)
    with output.open("w", encoding="utf-8") as fp:
        for batch in _batched(questions, args.batch_size):
            triples: list[tuple[int, str, list[DocumentChunk]]] = []
            for q_id, query in batch:
                chunks = retriever.search(
                    query=query,
                    top_k=rag.top_k,
                    top_kr=rag.top_kr,
                    strategy=rag.strategy,
                )
                triples.append((q_id, query, chunks))

            prompts = [build_messages(q, c, rag.prompt_name) for _, q, c in triples]
            answers = (
                [llm.invoke(prompts[0])] if len(prompts) == 1 else llm.invoke_batch(prompts)
            )

            for (q_id, query, chunks), answer in zip(triples, answers, strict=True):
                record = {
                    "index": q_id,
                    "question": query,
                    "answer": answer,
                    "chunk_ids": [c.id for c in chunks],
                    "links": list(
                        dict.fromkeys(c.source_url for c in chunks if c.source_url)
                    ),
                }
                fp.write(json.dumps(record, ensure_ascii=False))
                fp.write("\n")
                logger.info("rag done: q_id=%d answer_len=%d", q_id, len(answer))
