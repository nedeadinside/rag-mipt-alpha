from functools import lru_cache
from pathlib import Path
from typing import Literal

from dynaconf import Dynaconf
from pydantic import BaseModel, Field

from src.types.search_strategy import SearchStrategy


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

    chunk_size: int = Field(496, gt=0)
    threshold: float = Field(0.5, gt=0, lt=1)
    embedding_model: str = "minishlab/potion-base-32M"
    min_sentences_per_chunk: int = Field(4, gt=0)
    min_characters_per_sentence: int = Field(80, gt=0)
    skip_window: int = Field(1, ge=0)
    filter_tolerance: float = Field(0.05, gt=0, lt=1)
    overlap_size: float = Field(0.25, gt=0, lt=1)
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


class LLMSettings(BaseModel):
    """
    LLM settings.
    """

    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    top_k: int | None = Field(None, gt=0)


class RAGSettings(BaseModel):
    """
    RAG pipeline settings.
    """

    top_k: int = Field(100, gt=0)
    top_kr: int = Field(20, gt=0)
    strategy: SearchStrategy = SearchStrategy.DEFAULT
    prompt_name: str = "qa_rag"


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


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """
    Return validated LLM settings.

    :return: LLM settings.
    """
    return LLMSettings(**_load_raw().llm.to_dict())


@lru_cache(maxsize=1)
def get_rag_settings() -> RAGSettings:
    """
    Return validated RAG settings.

    :return: RAG settings.
    """
    return RAGSettings(**_load_raw().rag.to_dict())
