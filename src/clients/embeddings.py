from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.models import SparseVector


class FastEmbedE5DenseEmbedder:
    """
    Dense embedder backed by FastEmbed.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        """
        Initialize the embedder.

        :param model_name: FastEmbed dense model identifier.
        :param cache_dir: Local cache directory for downloaded model weights.
        """
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single string.

        :param text: Text to embed.
        :return: Dense embedding.
        """
        [vector] = list(self._model.query_embed([text]))
        return vector.tolist()

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings.

        :param texts: Texts to embed.
        :return: Dense embeddings in input order.
        """
        return [vector.tolist() for vector in self._model.passage_embed(texts)]


class FastEmbedSparseEmbedder:
    """
    Sparse embedder backed by FastEmbed.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        """
        Initialize the embedder.

        :param model_name: FastEmbed sparse model identifier.
        :param cache_dir: Local cache directory for downloaded model weights.
        """
        self._model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_query(self, text: str) -> SparseVector:
        """
        Embed a single string into a sparse vector.

        :param text: Text to embed.
        :return: Sparse embedding.
        """
        [sparse] = list(self._model.query_embed([text]))
        return SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())

    def embed_text(self, texts: list[str]) -> list[SparseVector]:
        """
        Embed a batch of strings into sparse vectors.

        :param texts: Texts to embed.
        :return: Sparse embeddings in input order.
        """
        return [
            SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())
            for sparse in self._model.embed(texts)
        ]
