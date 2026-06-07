from typing import Protocol, runtime_checkable

from src.types.document import DocumentChunk
from src.types.source import SourceDocument


@runtime_checkable
class Chunker(Protocol):
    """
    Chunker contract.
    """

    def chunk(self, doc: SourceDocument) -> list[DocumentChunk]:
        """
        Split a document into chunks.

        :param doc: Source document.
        :return: Chunks.
        """
        ...
