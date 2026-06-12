import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from hack_optimization.clients import BatchMultiQueryExpander
from hack_optimization.io import append_writer, completed_keys
from hack_optimization.records import ChunkRecord
from src.retrieval import MultiQueryExpander
from src.types.document import DocumentChunk
from src.types.search_strategy import SearchStrategy
from src.types.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrieveStage:
    """
    Recall stage: fetch candidate chunks per question and append them to JSONL.
    """

    def __init__(
        self,
        store: VectorStore,
        collection: str,
        top_k: int,
        strategy: SearchStrategy,
        query_expander: MultiQueryExpander | None = None,
    ) -> None:
        """
        Initialize the stage.

        :param store: Vector store used for first-stage recall.
        :param collection: Collection name to search.
        :param top_k: Number of candidates to pull from the store.
        :param strategy: Strategy selector.
        :param query_expander: Query expander required by the multiquery strategy.
        """
        self._store = store
        self._collection = collection
        self._top_k = top_k
        self._strategy = strategy
        self._expander = (
            BatchMultiQueryExpander.upgrade(query_expander)
            if query_expander is not None
            else None
        )

    def run(
        self, questions: Iterable[tuple[int, str]], output: Path, batch_size: int = 16
    ) -> None:
        """
        Recall candidates for each pending question and append one line per question.

        :param questions: Pairs of question id and query text.
        :param output: Destination JSONL, resumed when already partially written.
        :param batch_size: Number of questions bundled into one expansion call.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        done = completed_keys(output, ChunkRecord)
        with append_writer(output) as write:
            buffer: list[tuple[int, str]] = []
            for q_id, query in questions:
                if q_id in done:
                    continue
                buffer.append((q_id, query))
                if len(buffer) >= batch_size:
                    self._flush(buffer, write)
                    buffer.clear()
            if buffer:
                self._flush(buffer, write)

    def _flush(
        self,
        buffer: list[tuple[int, str]],
        write: Callable[[ChunkRecord], None],
    ) -> None:
        """
        Recall candidates for a buffered batch and append lines in input order.

        :param buffer: Pending question id and query pairs.
        :param write: Sink appending one line per question.
        """
        chunks_per_query = self._recall_many([query for _, query in buffer])
        for (q_id, query), chunks in zip(buffer, chunks_per_query, strict=True):
            write(ChunkRecord(q_id=q_id, query=query, chunks=chunks))
            logger.info("retrieve done: q_id=%d chunks=%d", q_id, len(chunks))

    def _recall_many(self, queries: list[str]) -> list[list[DocumentChunk]]:
        """
        Pull candidate chunks for a batch of queries according to the configured strategy.

        :param queries: User query strings.
        :return: Candidate chunks per query in input order.
        """
        match self._strategy:
            case SearchStrategy.DEFAULT:
                return [
                    self._store.search(self._collection, query, self._top_k)
                    for query in queries
                ]
            case SearchStrategy.MULTIQUERY:
                if self._expander is None:
                    raise ValueError("MULTIQUERY strategy requires a query_expander")
                expansions = self._expander.expand_many(queries)
                return [
                    self._store.search_multiquery(self._collection, expanded, self._top_k)
                    for expanded in expansions
                ]
