import argparse
import json
import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

from cmd.questions import iter_questions
from src.clients import (
    CrossEncoderReranker,
    FastEmbedE5DenseEmbedder,
    FastEmbedSparseEmbedder,
    LocalHybridQdrantStore,
    OllamaLLM,
)
from src.prompts import load_text
from src.rag import LLMVerifier
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

    dense = FastEmbedE5DenseEmbedder(args.dense_model, args.use_cuda, args.cache_dir)
    sparse = FastEmbedSparseEmbedder(args.sparse_model, args.use_cuda, args.cache_dir)
    store = LocalHybridQdrantStore(
        path=args.qdrant_path,
        dense=dense,
        sparse=sparse,
        prefetch_limit=args.prefetch_limit,
    )
    reranker = CrossEncoderReranker(args.reranker_model, args.reranker_device)
    llm = OllamaLLM(
        args.llm_model_name,
        args.llm_base_url,
        args.llm_temperature,
        top_p=args.llm_top_p,
        top_k=args.llm_top_k,
    )
    expander = (
        MultiQueryExpander(llm=llm) if args.strategy == SearchStrategy.MULTIQUERY else None
    )
    retriever = HybridRetriever(
        store=store,
        reranker=reranker,
        collection=args.collection_name,
        query_expander=expander,
    )
    verifier = LLMVerifier(llm=llm, prompt_name=args.verifier_prompt_name)
    refusal = load_text(args.refusal_text_name)

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    questions = iter_questions(args.questions, args.limit)
    with output.open("w", encoding="utf-8") as fp:
        for batch in _batched(questions, args.batch_size):
            triples: list[tuple[int, str, list[DocumentChunk]]] = []
            for q_id, query in batch:
                chunks = retriever.search(
                    query=query,
                    top_k=args.top_k,
                    top_kr=args.top_kr,
                    strategy=args.strategy,
                )
                triples.append((q_id, query, chunks))

            passes: list[bool] = []
            for _, query, chunks in triples:
                verdict = verifier.verify(query, chunks)
                if not verdict.is_relevant:
                    logger.info(
                        "verifier reject: query_len=%d reason=%s",
                        len(query),
                        verdict.reason,
                    )
                passes.append(verdict.is_relevant)

            accepted_indices = [i for i, ok in enumerate(passes) if ok]
            accepted_prompts = [
                build_messages(triples[i][1], triples[i][2], args.prompt_name)
                for i in accepted_indices
            ]
            if len(accepted_prompts) == 0:
                accepted_answers: list[str] = []
            elif len(accepted_prompts) == 1:
                accepted_answers = [llm.invoke(accepted_prompts[0])]
            else:
                accepted_answers = llm.invoke_batch(accepted_prompts)

            cursor = 0
            for (q_id, query, chunks), ok in zip(triples, passes, strict=True):
                if ok:
                    answer = accepted_answers[cursor]
                    cursor += 1
                else:
                    answer = refusal
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
                logger.info(
                    "rag done: q_id=%d answer_len=%d refused=%s",
                    q_id,
                    len(answer),
                    not ok,
                )
