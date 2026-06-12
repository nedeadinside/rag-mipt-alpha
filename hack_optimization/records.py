from pydantic import BaseModel

from src.types.document import DocumentChunk


class ChunkRecord(BaseModel):
    """
    Per-question artifact carrying the candidate chunks between stages.
    """

    q_id: int
    query: str
    chunks: list[DocumentChunk]

    @property
    def record_key(self) -> int:
        """
        Return the stable key identifying this record across resumes.

        :return: Question identifier.
        """
        return self.q_id


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

    index: int
    question: str
    answer: str
    chunk_ids: list[str]
    links: list[str]

    @property
    def record_key(self) -> int:
        """
        Return the stable key identifying this record across resumes.

        :return: Question identifier.
        """
        return self.index
