from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from chonkie import OverlapRefinery
from chonkie import SemanticChunker as ChonkieSemanticChunker

from src.types.document import DocumentChunk
from src.types.source import SourceDocument


class SemanticChunker:
    """
    Semantic splitter with overlap refinement over source documents.
    """

    def __init__(
        self,
        embedding_model: str,
        threshold: float,
        chunk_size: int,
        min_sentences_per_chunk: int,
        min_characters_per_sentence: int,
        skip_window: int,
        filter_tolerance: float,
        overlap_size: float,
        overlap_method: Literal["suffix", "prefix", "justified"],
    ) -> None:
        """
        Initialize the chunker.

        :param embedding_model: Model used to embed sentences for semantic splitting.
        :param threshold: Semantic similarity threshold for splitting.
        :param chunk_size: Target chunk size in tokens.
        :param min_sentences_per_chunk: Minimum sentences per chunk.
        :param min_characters_per_sentence: Minimum characters per sentence.
        :param skip_window: Window of chunks to skip when merging similar segments.
        :param filter_tolerance: Tolerance for the similarity filter.
        :param overlap_size: Fraction of chunk used as overlap context.
        :param overlap_method: Overlap placement strategy.
        """
        self._chunker = ChonkieSemanticChunker(
            embedding_model=embedding_model,
            threshold=threshold,
            chunk_size=chunk_size,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            skip_window=skip_window,
            filter_tolerance=filter_tolerance,
        )
        self._refiner = OverlapRefinery(
            tokenizer="character",
            context_size=overlap_size,
            method=overlap_method,
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
                token_count=c.token_count,
                metadata=metadata,
            )
            for c in refined_chunks
        ]
