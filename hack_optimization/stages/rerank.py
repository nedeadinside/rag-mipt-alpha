import logging
from collections.abc import Callable
from pathlib import Path

from hack_optimization.clients import BatchCrossEncoderReranker
from hack_optimization.io import append_writer, completed_keys, read_records
from hack_optimization.records import ChunkRecord
from src.clients.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RerankStage:
    """
    Rerank stage: reorder candidate chunks per question and append the trimmed set to JSONL.
    """

    def __init__(self, reranker: CrossEncoderReranker, top_kr: int) -> None:
        """
        Initialize the stage.

        :param reranker: Reranker used for second-stage scoring.
        :param top_kr: Number of candidates to keep after reranking.
        """
        self._reranker = BatchCrossEncoderReranker.upgrade(reranker)
        self._top_kr = top_kr

    def run(self, input_path: Path, output_path: Path, batch_size: int = 32) -> None:
        """
        Rerank candidates for each pending record and append one line per record.

        :param input_path: JSONL produced by the recall stage.
        :param output_path: Destination JSONL, resumed when already partially written.
        :param batch_size: Number of records bundled into one scoring pass.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        done = completed_keys(output_path, ChunkRecord)
        with append_writer(output_path) as write:
            buffer: list[ChunkRecord] = []
            for record in read_records(input_path, ChunkRecord):
                if record.q_id in done:
                    continue
                buffer.append(record)
                if len(buffer) >= batch_size:
                    self._flush(buffer, write)
                    buffer.clear()
            if buffer:
                self._flush(buffer, write)

    def _flush(
        self,
        buffer: list[ChunkRecord],
        write: Callable[[ChunkRecord], None],
    ) -> None:
        """
        Rerank a buffered batch in one scoring pass and append lines in input order.

        :param buffer: Pending records awaiting reranking.
        :param write: Sink appending one line per record.
        """
        reranked_per_record = self._reranker.rerank_many(
            [record.query for record in buffer],
            [record.chunks for record in buffer],
            self._top_kr,
        )
        for record, reranked in zip(buffer, reranked_per_record, strict=True):
            write(ChunkRecord(q_id=record.q_id, query=record.query, chunks=reranked))
            logger.info("rerank done: q_id=%d chunks=%d", record.q_id, len(reranked))
