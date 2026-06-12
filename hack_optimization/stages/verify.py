import logging
from pathlib import Path

from hack_optimization.io import append_writer, completed_keys, read_records
from hack_optimization.records import ChunkRecord, VerifiedRecord
from src.types.verifier import VerifierProtocol

logger = logging.getLogger(__name__)


class VerifyStage:
    """
    Verify stage: annotate each question with a relevance verdict and append it to JSONL.
    """

    def __init__(self, verifier: VerifierProtocol) -> None:
        """
        Initialize the stage.

        :param verifier: Verifier deciding whether the chunks cover the question.
        """
        self._verifier = verifier

    def run(self, input_path: Path, output_path: Path) -> None:
        """
        Verify each pending record and append one annotated line per record.

        :param input_path: JSONL produced by the rerank stage.
        :param output_path: Destination JSONL, resumed when already partially written.
        """
        done = completed_keys(output_path, VerifiedRecord)
        with append_writer(output_path) as write:
            for record in read_records(input_path, ChunkRecord):
                if record.q_id in done:
                    continue
                verdict = self._verifier.verify(record.query, record.chunks)
                if not verdict.is_relevant:
                    logger.info(
                        "verify reject: q_id=%d reason=%s", record.q_id, verdict.reason
                    )
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
