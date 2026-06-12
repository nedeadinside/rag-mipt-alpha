from typing import Protocol, TypeVar, runtime_checkable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.types.document import DocumentChunk
from src.types.llm import LLM
from src.types.query_expander import QueryExpanderProtocol
from src.types.reranker import Reranker
from src.types.verifier import VerificationResult, VerifierProtocol

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class BatchLLM(LLM, Protocol):
    """
    Chat model contract extended with a structured batch call.
    """

    def invoke_structured_batch(
        self, batches: list[list[BaseMessage]], schema: type[T]
    ) -> list[T]:
        """
        Generate typed replies for several conversations in one call.

        :param batches: Conversations to run, each a list of chat messages.
        :param schema: Model that defines the expected reply shape.
        :return: Parsed instances in input order.
        """
        ...


@runtime_checkable
class BatchReranker(Reranker, Protocol):
    """
    Reranker contract extended with a cross-question batch call.
    """

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
        ...


@runtime_checkable
class BatchQueryExpander(QueryExpanderProtocol, Protocol):
    """
    Query expander contract extended with a cross-question batch call.
    """

    def expand_many(self, queries: list[str]) -> list[list[str]]:
        """
        Produce alternative search queries for several queries in one call.

        :param queries: Original user queries.
        :return: Expanded search queries per input query in input order.
        """
        ...


@runtime_checkable
class BatchVerifier(VerifierProtocol, Protocol):
    """
    Verifier contract extended with a cross-question batch call.
    """

    def verify_many(
        self, queries: list[str], chunks: list[list[DocumentChunk]]
    ) -> list[VerificationResult]:
        """
        Decide fragment sufficiency for several questions in one call.

        :param queries: User queries, one per fragment list.
        :param chunks: Retrieved supporting fragments per query.
        :return: Relevance decisions in input order.
        """
        ...
