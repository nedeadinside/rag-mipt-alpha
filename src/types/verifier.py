from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from src.types.document import DocumentChunk


class VerificationResult(BaseModel):
    """
    Verifier decision on whether retrieved fragments cover the question.
    """

    is_relevant: bool
    reason: str


@runtime_checkable
class VerifierProtocol(Protocol):
    """
    Verifier contract.
    """

    def verify(self, query: str, chunks: list[DocumentChunk]) -> VerificationResult:
        """
        Decide whether the fragments are sufficient to answer the question.

        :param query: User query.
        :param chunks: Retrieved supporting fragments.
        :return: Relevance decision with a short rationale.
        """
        ...
