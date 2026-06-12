import logging

from langchain_core.messages import BaseMessage

from src.prompts import load_text
from src.rag.utils import build_messages
from src.types.answer import RAGAnswer
from src.types.document import DocumentChunk
from src.types.llm import LLM
from src.types.retriever import Retriever
from src.types.search_strategy import SearchStrategy
from src.types.verifier import VerifierProtocol

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval-augmented generation pipeline.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        top_k: int,
        top_kr: int,
        strategy: SearchStrategy,
        prompt_name: str,
        refusal_text_name: str,
        verifier: VerifierProtocol | None = None,
    ) -> None:
        """
        Initialize the pipeline.

        :param retriever: Retriever used to fetch supporting chunks.
        :param llm: LLM used to generate the final answer.
        :param top_k: Number of candidates fetched before reranking.
        :param top_kr: Number of chunks kept after reranking.
        :param strategy: Search strategy passed to the retriever.
        :param prompt_name: Prompt template used to render the answer.
        :param refusal_text_name: Text returned when the verifier rejects the chunks.
        :param verifier: Optional relevance verifier; when set, gates generation.
        """
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._top_kr = top_kr
        self._strategy = strategy
        self._prompt_name = prompt_name
        self._refusal_text_name = refusal_text_name
        self._verifier = verifier

    def answer(self, query: str) -> RAGAnswer:
        """
        Retrieve supporting chunks and generate an answer for one query.

        :param query: User query.
        :return: Generated answer paired with the chunks used as its context.
        """
        chunks = self._retrieve(query)
        if not self._is_relevant(query, chunks):
            text = load_text(self._refusal_text_name)
            return RAGAnswer(text=text, chunks=chunks)
        messages = self._render_messages(query, chunks)
        text = self._llm.invoke(messages)
        logger.info("rag answer ready: query_len=%d chunks=%d", len(query), len(chunks))
        return RAGAnswer(text=text, chunks=chunks)

    def answer_batch(self, queries: list[str]) -> list[RAGAnswer]:
        """
        Retrieve supporting chunks and generate answers for several queries in one LLM call.

        :param queries: User queries.
        :return: Generated answers in input order.
        """
        chunks_per_query = [self._retrieve(q) for q in queries]
        passes = [self._is_relevant(q, c) for q, c in zip(queries, chunks_per_query, strict=True)]

        accepted_indices = [i for i, ok in enumerate(passes) if ok]
        accepted_batches = [
            self._render_messages(queries[i], chunks_per_query[i]) for i in accepted_indices
        ]
        accepted_texts = self._llm.invoke_batch(accepted_batches) if accepted_batches else []

        refusal = load_text(self._refusal_text_name)
        answers: list[RAGAnswer] = []
        cursor = 0
        for i, ok in enumerate(passes):
            if ok:
                text = accepted_texts[cursor]
                cursor += 1
            else:
                text = refusal
            answers.append(RAGAnswer(text=text, chunks=chunks_per_query[i]))
        logger.info(
            "rag batch ready: queries=%d accepted=%d refused=%d",
            len(queries),
            len(accepted_indices),
            len(queries) - len(accepted_indices),
        )
        return answers

    def _retrieve(self, query: str) -> list[DocumentChunk]:
        """
        Run the retriever with configured settings.

        :param query: User query.
        :return: Reranked chunks.
        """
        return self._retriever.search(
            query=query,
            top_k=self._top_k,
            top_kr=self._top_kr,
            strategy=self._strategy,
        )

    def _render_messages(self, query: str, chunks: list[DocumentChunk]) -> list[BaseMessage]:
        """
        Render the chat prompt with the chunks joined as a context block.

        :param query: User query.
        :param chunks: Chunks to embed into the context slot.
        :return: Chat messages ready for the LLM.
        """
        return build_messages(query, chunks, self._prompt_name)

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
