from pydantic import BaseModel


class SourceDocument(BaseModel):
    """
    Source document.
    """

    source_id: str
    url: str | None = None
    title: str | None = None
    kind: str | None = None
    text: str
