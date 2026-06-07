from functools import lru_cache
from pathlib import Path
from typing import Literal

from dynaconf import Dynaconf
from pydantic import BaseModel, Field


class IngestionSettings(BaseModel):
    """
    Ingestion settings.
    """

    collection_name: str
    qdrant_path: str
    websites_csv_path: str
    upsert_batch_size: int = Field(64, gt=0)


class ChunkingSettings(BaseModel):
    """
    Chunking settings.
    """

    chunk_size: int = Field(512, gt=0)
    threshold: float = Field(0.8, gt=0, lt=1)
    embedding_model: str = "minishlab/potion-base-32M"
    overlap_size: float = Field(0.1, gt=0, lt=1)
    overlap_method: Literal["suffix", "prefix", "justified"]


class EmbeddingSettings(BaseModel):
    """
    Embedding settings.
    """

    dense_model: str = "intfloat/multilingual-e5-large"
    sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    cache_dir: str | None = None


class RetrievalSettings(BaseModel):
    """
    Retrieval settings.
    """

    prefetch_limit: int = Field(50, gt=0)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"


@lru_cache(maxsize=1)
def _load_raw() -> Dynaconf:
    """
    Load the raw settings from disk once.

    :return: Dynaconf object.
    """
    project_root = Path(__file__).parent.parent
    return Dynaconf(
        settings_files=[project_root / "settings.yaml"],
        envvar_prefix="RAG",
        load_dotenv=True,
    )


@lru_cache(maxsize=1)
def get_ingestion_settings() -> IngestionSettings:
    """
    Return validated ingestion settings.

    :return: Ingestion settings.
    """
    return IngestionSettings(**_load_raw().ingestion.to_dict())


@lru_cache(maxsize=1)
def get_chunking_settings() -> ChunkingSettings:
    """
    Return validated chunking settings.

    :return: Chunking settings.
    """
    return ChunkingSettings(**_load_raw().chunking.to_dict())


@lru_cache(maxsize=1)
def get_embedding_settings() -> EmbeddingSettings:
    """
    Return validated embedding settings.

    :return: Embedding settings.
    """
    return EmbeddingSettings(**_load_raw().embedding.to_dict())


@lru_cache(maxsize=1)
def get_retrieval_settings() -> RetrievalSettings:
    """
    Return validated retrieval settings.

    :return: Retrieval settings.
    """
    return RetrievalSettings(**_load_raw().retrieval.to_dict())
