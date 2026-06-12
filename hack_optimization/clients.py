from typing import TypeVar

import numpy as np
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.clients.ollama import OllamaLLM
from src.clients.reranker import CrossEncoderReranker
from src.clients.utils import DEFAULT_RETRY_EXCEPTIONS, llm_retry
from src.prompts import load_prompt
from src.prompts.loader import load_prompt_group
from src.rag import LLMVerifier
from src.retrieval import MultiQueryExpander
from src.types.document import DocumentChunk
from src.types.verifier import VerificationResult

T = TypeVar("T", bound=BaseModel)

_CONTEXT_SEP = "\n---\n"


class BatchOllamaLLM(OllamaLLM):
    """
    Ollama chat client extended with a structured batch call.
    """

    @classmethod
    def upgrade(cls, base: OllamaLLM) -> "BatchOllamaLLM":
        """
        Wrap an existing client, reusing its already-built chat model.

        :param base: Client whose loaded chat model is reused.
        :return: Batch-capable client over the same chat model.
        """
        upgraded = cls.__new__(cls)
        upgraded._model = base._model
        return upgraded

    @llm_retry(exceptions=(*DEFAULT_RETRY_EXCEPTIONS, OutputParserException))
    def invoke_structured_batch(
        self, batches: list[list[BaseMessage]], schema: type[T]
    ) -> list[T]:
        """
        Generate typed replies for several conversations in one call.

        :param batches: Conversations to run, each a list of chat messages.
        :param schema: Model that defines the expected reply shape.
        :return: Parsed instances in input order.
        """
        runnable = self._model.with_structured_output(schema)
        return list(runnable.batch(batches))


class BatchLLMVerifier(LLMVerifier):
    """
    Verifier extended with a cross-question batch call.
    """

    _llm: BatchOllamaLLM

    @classmethod
    def upgrade(cls, base: LLMVerifier) -> "BatchLLMVerifier":
        """
        Wrap an existing verifier, reusing its prompt and loaded chat model.

        :param base: Verifier whose configuration and chat model are reused.
        :return: Batch-capable verifier over the same chat model.
        """
        upgraded = cls.__new__(cls)
        upgraded._llm = BatchOllamaLLM.upgrade(base._llm)
        upgraded._prompt_name = base._prompt_name
        return upgraded

    def verify_many(
        self, queries: list[str], chunks: list[list[DocumentChunk]]
    ) -> list[VerificationResult]:
        """
        Decide fragment sufficiency for several questions in one call.

        :param queries: User queries, one per fragment list.
        :param chunks: Retrieved supporting fragments per query.
        :return: Relevance decisions in input order.
        """
        template = load_prompt(self._prompt_name)
        batches = [
            template.format_messages(
                context=_CONTEXT_SEP.join(c.text for c in fragments),
                question=query,
            )
            for query, fragments in zip(queries, chunks, strict=True)
        ]
        if not batches:
            return []
        return self._llm.invoke_structured_batch(batches, VerificationResult)


class BatchMultiQueryExpander(MultiQueryExpander):
    """
    Query expander extended with a cross-question batch call.
    """

    @classmethod
    def upgrade(cls, base: MultiQueryExpander) -> "BatchMultiQueryExpander":
        """
        Wrap an existing expander, reusing its prompt and chat model.

        :param base: Expander whose configuration and chat model are reused.
        :return: Batch-capable expander over the same chat model.
        """
        upgraded = cls.__new__(cls)
        upgraded._llm = base._llm
        upgraded._prompt_name = base._prompt_name
        return upgraded

    def expand_many(self, queries: list[str]) -> list[list[str]]:
        """
        Produce alternative search queries for several queries in one call.

        :param queries: Original user queries.
        :return: Expanded search queries per input query in input order.
        """
        templates = load_prompt_group(self._prompt_name)
        width = len(templates)
        batches = [
            template.format_messages(question=query)
            for query in queries
            for template in templates
        ]
        if not batches:
            return []
        outputs = self._llm.invoke_batch(batches)
        return [
            [output.strip() for output in outputs[start : start + width]]
            for start in range(0, len(outputs), width)
        ]


class BatchCrossEncoderReranker(CrossEncoderReranker):
    """
    Cross-encoder reranker extended with a cross-question batch call.
    """

    @classmethod
    def upgrade(cls, base: CrossEncoderReranker) -> "BatchCrossEncoderReranker":
        """
        Wrap an existing reranker, reusing its already-loaded model.

        :param base: Reranker whose loaded model is reused.
        :return: Batch-capable reranker over the same model.
        """
        upgraded = cls.__new__(cls)
        upgraded._model = base._model
        return upgraded

    def rerank_many(
        self,
        queries: list[str],
        documents: list[list[DocumentChunk]],
        top_k: int,
    ) -> list[list[DocumentChunk]]:
        """
        Reorder candidates for several queries in one scoring pass.

        :param queries: Query strings, one per candidate list.
        :param documents: Candidate chunks per query.
        :param top_k: Maximum number of chunks to keep per query.
        :return: Reranked chunks per query in input order.
        """
        pairs: list[tuple[str, str]] = []
        spans: list[int] = []
        for query, candidates in zip(queries, documents, strict=True):
            spans.append(len(candidates))
            pairs.extend((query, candidate.text) for candidate in candidates)
        if not pairs:
            return [[] for _ in documents]

        raw = self._model.predict(pairs)
        scores: list[float] = np.asarray(raw, dtype=float).reshape(-1).tolist()

        results: list[list[DocumentChunk]] = []
        cursor = 0
        for candidates, span in zip(documents, spans, strict=True):
            chunk_scores = scores[cursor : cursor + span]
            cursor += span
            ranked = sorted(
                zip(candidates, chunk_scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            results.append(
                [doc.model_copy(update={"score": score}) for doc, score in ranked[:top_k]]
            )
        return results
