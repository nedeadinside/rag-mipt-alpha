from pydantic import BaseModel

from src.types.document import DocumentChunk


class RAGAnswer(BaseModel):
    """
    Generated answer paired with the chunks used as its context.
    """

    text: str
    chunks: list[DocumentChunk]
