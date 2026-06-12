import logging
from collections.abc import Callable
from pathlib import Path

from hack_optimization.io import append_writer, completed_keys, read_records
from hack_optimization.records import SubmissionRecord, VerifiedRecord
from src.prompts import load_text
from src.rag.utils import build_messages
from src.types.llm import LLM

logger = logging.getLogger(__name__)


class GenerateStage:
    """
    Generate stage: answer relevant questions and append submission lines to JSONL.
    """

    def __init__(self, llm: LLM, prompt_name: str, refusal_text_name: str) -> None:
        """
        Initialize the stage.

        :param llm: LLM used to generate the final answer.
        :param prompt_name: Prompt template name used for generation.
        :param refusal_text_name: Text resource name returned for rejected questions.
        """
        self._llm = llm
        self._prompt_name = prompt_name
        self._refusal_text_name = refusal_text_name

    def run(self, input_path: Path, output_path: Path, batch_size: int = 1) -> None:
        """
        Generate answers for every pending record and append submission lines.

        :param input_path: JSONL produced by the verify stage.
        :param output_path: Destination submission JSONL, resumed when already partially written.
        :param batch_size: Number of records bundled into one LLM call.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        done = completed_keys(output_path, SubmissionRecord)
        refusal = load_text(self._refusal_text_name)
        with append_writer(output_path) as write:
            buffer: list[VerifiedRecord] = []
            for record in read_records(input_path, VerifiedRecord):
                if record.q_id in done:
                    continue
                buffer.append(record)
                if len(buffer) >= batch_size:
                    self._flush(buffer, refusal, write)
                    buffer.clear()
            if buffer:
                self._flush(buffer, refusal, write)

    def _flush(
        self,
        buffer: list[VerifiedRecord],
        refusal: str,
        write: Callable[[SubmissionRecord], None],
    ) -> None:
        """
        Run the LLM on a buffered batch and append submission lines in input order.

        :param buffer: Pending verified records awaiting generation.
        :param refusal: Text returned for records rejected by the verifier.
        :param write: Sink appending one submission line per record.
        """
        accepted_indices = [i for i, record in enumerate(buffer) if record.is_relevant]
        accepted_prompts = [
            build_messages(buffer[i].query, buffer[i].chunks, self._prompt_name)
            for i in accepted_indices
        ]
        if len(accepted_prompts) == 0:
            accepted_texts: list[str] = []
        elif len(accepted_prompts) == 1:
            accepted_texts = [self._llm.invoke(accepted_prompts[0])]
        else:
            accepted_texts = self._llm.invoke_batch(accepted_prompts)

        cursor = 0
        for record in buffer:
            if record.is_relevant:
                answer = accepted_texts[cursor]
                cursor += 1
            else:
                answer = refusal
            links = list(
                dict.fromkeys(c.source_url for c in record.chunks if c.source_url)
            )
            write(
                SubmissionRecord(
                    index=record.q_id,
                    question=record.query,
                    answer=answer,
                    chunk_ids=[c.id for c in record.chunks],
                    links=links,
                )
            )
            logger.info(
                "generate done: q_id=%d answer_len=%d refused=%s",
                record.q_id,
                len(answer),
                not record.is_relevant,
            )
