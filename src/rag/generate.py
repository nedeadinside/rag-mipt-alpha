import json
import logging
from pathlib import Path

from src.config import RAGSettings
from src.rag.utils import build_messages
from src.types.document import DocumentChunk
from src.types.llm import LLM

logger = logging.getLogger(__name__)


class AnswerGenerationStage:
    """
    Second stage of the two-phase pipeline: read retrieved chunks from JSONL and emit submission answers.
    """

    def __init__(self, llm: LLM, settings: RAGSettings) -> None:
        """
        Initialize the stage.

        :param llm: LLM used to generate the final answer.
        :param settings: RAG pipeline settings.
        """
        self._llm = llm
        self._settings = settings

    def run(self, input_jsonl: Path, output_jsonl: Path, batch_size: int = 1) -> None:
        """
        Generate answers for every record in the input JSONL and write submission lines.

        :param input_jsonl: JSONL produced by the retrieval stage.
        :param output_jsonl: Destination submission JSONL.
        :param batch_size: Number of records bundled into one LLM call.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open(
            "w", encoding="utf-8"
        ) as dst:
            buffer: list[tuple[int, str, list[DocumentChunk]]] = []
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                q_id = int(record["q_id"])
                query = record["query"]
                chunks = [DocumentChunk.model_validate(c) for c in record["chunks"]]
                buffer.append((q_id, query, chunks))
                if len(buffer) >= batch_size:
                    self._flush(buffer, dst)
                    buffer.clear()
            if buffer:
                self._flush(buffer, dst)

    def _flush(
        self,
        buffer: list[tuple[int, str, list[DocumentChunk]]],
        dst,
    ) -> None:
        """
        Run the LLM on a buffered batch and write submission lines in input order.

        :param buffer: Pending records awaiting generation.
        :param dst: Open writable text stream for the submission JSONL.
        """
        prompts = [build_messages(q, c, self._settings.prompt_name) for _, q, c in buffer]
        if len(prompts) == 1:
            answers = [self._llm.invoke(prompts[0])]
        else:
            answers = self._llm.invoke_batch(prompts)

        for (q_id, query, chunks), answer in zip(buffer, answers, strict=True):
            chunk_ids = [c.id for c in chunks]
            links = list(dict.fromkeys(c.source_url for c in chunks if c.source_url))
            record = {
                "index": q_id,
                "question": query,
                "answer": answer,
                "chunk_ids": chunk_ids,
                "links": links,
            }
            dst.write(json.dumps(record, ensure_ascii=False))
            dst.write("\n")
            logger.info("generation done: q_id=%d answer_len=%d", q_id, len(answer))
