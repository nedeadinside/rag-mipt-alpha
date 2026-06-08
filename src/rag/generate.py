import json
import logging
from pathlib import Path
from typing import TextIO

from src.config import RAGSettings
from src.prompts import load_text
from src.rag.utils import build_messages
from src.types.document import DocumentChunk
from src.types.llm import LLM
from src.types.verifier import VerifierProtocol

logger = logging.getLogger(__name__)


class AnswerGenerationStage:
    """
    Second stage of the two-phase pipeline: read retrieved chunks from JSONL and emit submission answers.
    """

    def __init__(
        self,
        llm: LLM,
        settings: RAGSettings,
        verifier: VerifierProtocol | None = None,
    ) -> None:
        """
        Initialize the stage.

        :param llm: LLM used to generate the final answer.
        :param settings: RAG pipeline settings.
        :param verifier: Optional relevance verifier; when set, gates generation.
        """
        self._llm = llm
        self._settings = settings
        self._verifier = verifier

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
        dst: TextIO,
    ) -> None:
        """
        Run the LLM on a buffered batch and write submission lines in input order.

        :param buffer: Pending records awaiting generation.
        :param dst: Open writable text stream for the submission JSONL.
        """
        passes = [self._is_relevant(q, c) for _, q, c in buffer]
        accepted_indices = [i for i, ok in enumerate(passes) if ok]
        accepted_prompts = [
            build_messages(buffer[i][1], buffer[i][2], self._settings.prompt_name)
            for i in accepted_indices
        ]
        if len(accepted_prompts) == 0:
            accepted_texts: list[str] = []
        elif len(accepted_prompts) == 1:
            accepted_texts = [self._llm.invoke(accepted_prompts[0])]
        else:
            accepted_texts = self._llm.invoke_batch(accepted_prompts)

        refusal = load_text(self._settings.refusal_text_name)
        cursor = 0
        for (q_id, query, chunks), ok in zip(buffer, passes, strict=True):
            if ok:
                answer = accepted_texts[cursor]
                cursor += 1
            else:
                answer = refusal
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
            logger.info(
                "generation done: q_id=%d answer_len=%d refused=%s",
                q_id,
                len(answer),
                not ok,
            )

    def _is_relevant(self, query: str, chunks: list[DocumentChunk]) -> bool:
        """
        Ask the verifier whether the chunks cover the question.

        :param query: User query.
        :param chunks: Retrieved supporting fragments.
        :return: True when generation should proceed; True unconditionally when no verifier is configured.
        """
        if self._verifier is None:
            return True
        result = self._verifier.verify(query, chunks)
        if not result.is_relevant:
            logger.info("verifier reject: query_len=%d reason=%s", len(query), result.reason)
        return result.is_relevant
