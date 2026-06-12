from typing import ClassVar

from pydantic import BaseModel

from src.types.document import DocumentChunk


class ChunkRecord(BaseModel):
    """
    Per-question artifact carrying the candidate chunks between stages.
    """

    key_field: ClassVar[str] = "q_id"

    q_id: int
    query: str
    chunks: list[DocumentChunk]


class VerifiedRecord(ChunkRecord):
    """
    Per-question artifact augmenting the candidate chunks with a relevance verdict.
    """

    is_relevant: bool
    reason: str


class SubmissionRecord(BaseModel):
    """
    Final per-question submission artifact.
    """

    key_field: ClassVar[str] = "index"

    index: int
    question: str
    answer: str
    chunk_ids: list[str]
    links: list[str]
