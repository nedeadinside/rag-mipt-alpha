import json
import logging
from collections.abc import Iterable
from pathlib import Path

from src.config import RAGSettings
from src.types.retriever import Retriever

logger = logging.getLogger(__name__)


class ChunkRetrievalStage:
    """
    First stage of the two-phase pipeline: fetch supporting chunks per question and dump them to JSONL.
    """

    def __init__(self, retriever: Retriever, settings: RAGSettings) -> None:
        """
        Initialize the stage.

        :param retriever: Retriever used to fetch supporting chunks.
        :param settings: RAG pipeline settings.
        """
        self._retriever = retriever
        self._settings = settings

    def run(self, questions: Iterable[tuple[int, str]], output_jsonl: Path) -> None:
        """
        Run retrieval for each question and write one JSONL line per question.

        :param questions: Pairs of question id and query text.
        :param output_jsonl: Destination JSONL file.
        """
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as fp:
            for q_id, query in questions:
                chunks = self._retriever.search(
                    query=query,
                    top_k=self._settings.top_k,
                    top_kr=self._settings.top_kr,
                    strategy=self._settings.strategy,
                )
                record = {
                    "q_id": q_id,
                    "query": query,
                    "chunks": [c.model_dump(mode="json") for c in chunks],
                }
                fp.write(json.dumps(record, ensure_ascii=False))
                fp.write("\n")
                logger.info("retrieval done: q_id=%d chunks=%d", q_id, len(chunks))
