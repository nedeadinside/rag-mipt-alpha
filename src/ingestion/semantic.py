from uuid import NAMESPACE_URL, uuid5

from chonkie import OverlapRefinery
from chonkie import SemanticChunker as ChonkieSemanticChunker

from src.config import ChunkingSettings
from src.types.document import DocumentChunk
from src.types.source import SourceDocument


class SemanticChunker:
    """
    Semantic splitter with overlap refinement over source documents.
    """

    def __init__(self, settings: ChunkingSettings) -> None:
        """
        Initialize the chunker.

        :param settings: Chunking settings.
        """
        self._chunker = ChonkieSemanticChunker(
            embedding_model=settings.embedding_model,
            threshold=settings.threshold,
            chunk_size=settings.chunk_size,
        )
        self._refiner = OverlapRefinery(
            tokenizer="character",
            context_size=settings.overlap_size,
            method=settings.overlap_method,
            merge=True,
        )

    def chunk(self, doc: SourceDocument) -> list[DocumentChunk]:
        """
        Split a source document into refined chunks.

        :param doc: Source document.
        :return: Chunks in document order.
        """
        if not doc.text or not doc.text.strip():
            return []

        raw_chunks = self._chunker.chunk(doc.text)
        refined_chunks = self._refiner.refine(raw_chunks)

        metadata = {"kind": doc.kind} if doc.kind else {}
        return [
            DocumentChunk(
                id=str(uuid5(NAMESPACE_URL, f"{doc.source_id}:{c.start_index}:{c.end_index}")),
                text=c.text,
                source_id=doc.source_id,
                source_url=doc.url,
                source_title=doc.title,
                start_index=c.start_index,
                end_index=c.end_index,
                length=c.token_count,
                metadata=metadata,
            )
            for c in refined_chunks
        ]
