import logging
from pathlib import Path

from hack_optimization.io import append_writer, completed_keys, read_records
from hack_optimization.records import ChunkRecord
from src.types.reranker import Reranker

logger = logging.getLogger(__name__)


class RerankStage:
    """
    Rerank stage: reorder candidate chunks per question and append the trimmed set to JSONL.
    """

    def __init__(self, reranker: Reranker, top_kr: int) -> None:
        """
        Initialize the stage.

        :param reranker: Reranker used for second-stage scoring.
        :param top_kr: Number of candidates to keep after reranking.
        """
        self._reranker = reranker
        self._top_kr = top_kr

    def run(self, input_path: Path, output_path: Path) -> None:
        """
        Rerank candidates for each pending record and append one line per record.

        :param input_path: JSONL produced by the recall stage.
        :param output_path: Destination JSONL, resumed when already partially written.
        """
        done = completed_keys(output_path, ChunkRecord)
        with append_writer(output_path) as write:
            for record in read_records(input_path, ChunkRecord):
                if record.q_id in done:
                    continue
                reranked = self._reranker.rerank(record.query, record.chunks, self._top_kr)
                write(ChunkRecord(q_id=record.q_id, query=record.query, chunks=reranked))
                logger.info("rerank done: q_id=%d chunks=%d", record.q_id, len(reranked))
