from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Document chunk.
    """

    id: str
    text: str
    source_id: str
    source_url: str | None = None
    source_title: str | None = None
    start_index: int
    end_index: int
    token_count: int
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
