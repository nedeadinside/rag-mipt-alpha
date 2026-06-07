import numpy as np
from FlagEmbedding import FlagReranker

from src.types.document import DocumentChunk


class FlagEmbeddingReranker:
    """
    FlagEmbedding reranker.
    """

    def __init__(
        self,
        model_name: str,
        use_fp16: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        """
        Initialize the reranker.

        :param model_name: Reranker model identifier.
        :param use_fp16: Whether to load weights in fp16.
        :param cache_dir: Local cache directory for downloaded model weights.
        """
        self._model = FlagReranker(model_name, use_fp16=use_fp16, cache_dir=cache_dir)

    def rerank(
        self,
        query: str,
        documents: list[DocumentChunk],
        top_k: int,
    ) -> list[DocumentChunk]:
        """
        Reorder candidates by relevance to the query.

        :param query: Query string.
        :param documents: Candidate chunks.
        :param top_k: Maximum number of chunks to keep.
        :return: Reranked chunks sorted by score descending.
        """
        if not documents:
            return []
        pairs = [[query, d.text] for d in documents]
        raw = self._model.compute_score(pairs, normalize=True)
        scores: list[float] = np.asarray(raw, dtype=float).reshape(-1).tolist()
        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [doc.model_copy(update={"score": score}) for doc, score in ranked[:top_k]]
