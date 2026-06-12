import logging
from collections.abc import Iterable
from pathlib import Path

from hack_optimization.io import append_writer, completed_keys
from hack_optimization.records import ChunkRecord
from src.types.document import DocumentChunk
from src.types.query_expander import QueryExpanderProtocol
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
        query_expander: QueryExpanderProtocol | None = None,
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
        self._expander = query_expander

    def run(self, questions: Iterable[tuple[int, str]], output: Path) -> None:
        """
        Recall candidates for each pending question and append one line per question.

        :param questions: Pairs of question id and query text.
        :param output: Destination JSONL, resumed when already partially written.
        """
        done = completed_keys(output, ChunkRecord)
        with append_writer(output) as write:
            for q_id, query in questions:
                if q_id in done:
                    continue
                chunks = self._recall(query)
                write(ChunkRecord(q_id=q_id, query=query, chunks=chunks))
                logger.info("retrieve done: q_id=%d chunks=%d", q_id, len(chunks))

    def _recall(self, query: str) -> list[DocumentChunk]:
        """
        Pull candidate chunks for a query according to the configured strategy.

        :param query: User query string.
        :return: Candidate chunks.
        """
        match self._strategy:
            case SearchStrategy.DEFAULT:
                return self._store.search(self._collection, query, self._top_k)
            case SearchStrategy.MULTIQUERY:
                if self._expander is None:
                    raise ValueError("MULTIQUERY strategy requires a query_expander")
                queries = self._expander.expand(query)
                return self._store.search_multiquery(self._collection, queries, self._top_k)
