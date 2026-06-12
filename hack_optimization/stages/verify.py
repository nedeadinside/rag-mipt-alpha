import logging
from collections.abc import Callable
from pathlib import Path

from hack_optimization.clients import BatchLLMVerifier
from hack_optimization.io import append_writer, completed_keys, read_records
from hack_optimization.records import ChunkRecord, VerifiedRecord
from src.rag import LLMVerifier

logger = logging.getLogger(__name__)


class VerifyStage:
    """
    Verify stage: annotate each question with a relevance verdict and append it to JSONL.
    """

    def __init__(self, verifier: LLMVerifier) -> None:
        """
        Initialize the stage.

        :param verifier: Verifier deciding whether the chunks cover the question.
        """
        self._verifier = BatchLLMVerifier.upgrade(verifier)

    def run(self, input_path: Path, output_path: Path, batch_size: int = 16) -> None:
        """
        Verify each pending record and append one annotated line per record.

        :param input_path: JSONL produced by the rerank stage.
        :param output_path: Destination JSONL, resumed when already partially written.
        :param batch_size: Number of records bundled into one verification call.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        done = completed_keys(output_path, VerifiedRecord)
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
        write: Callable[[VerifiedRecord], None],
    ) -> None:
        """
        Verify a buffered batch and append annotated lines in input order.

        :param buffer: Pending records awaiting verification.
        :param write: Sink appending one annotated line per record.
        """
        verdicts = self._verifier.verify_many(
            [record.query for record in buffer],
            [record.chunks for record in buffer],
        )
        for record, verdict in zip(buffer, verdicts, strict=True):
            if not verdict.is_relevant:
                logger.info("verify reject: q_id=%d reason=%s", record.q_id, verdict.reason)
            write(
                VerifiedRecord(
                    q_id=record.q_id,
                    query=record.query,
                    chunks=record.chunks,
                    is_relevant=verdict.is_relevant,
                    reason=verdict.reason,
                )
            )
            logger.info("verify done: q_id=%d relevant=%s", record.q_id, verdict.is_relevant)
