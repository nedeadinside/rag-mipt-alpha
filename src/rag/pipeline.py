import logging

from langchain_core.messages import BaseMessage

from src.config import RAGSettings
from src.prompts import load_prompt
from src.types.answer import RAGAnswer
from src.types.document import DocumentChunk
from src.types.llm import LLM
from src.types.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval-augmented generation pipeline.
    """

    def __init__(self, retriever: Retriever, llm: LLM, settings: RAGSettings) -> None:
        """
        Initialize the pipeline.

        :param retriever: Retriever used to fetch supporting chunks.
        :param llm: LLM used to generate the final answer.
        :param settings: RAG pipeline settings.
        """
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    def answer(self, query: str) -> RAGAnswer:
        """
        Retrieve supporting chunks and generate an answer for one query.

        :param query: User query.
        :return: Generated answer paired with the chunks used as its context.
        """
        chunks = self._retrieve(query)
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
        batches = [
            self._render_messages(q, c) for q, c in zip(queries, chunks_per_query, strict=True)
        ]
        texts = self._llm.invoke_batch(batches)
        logger.info("rag batch ready: queries=%d", len(queries))
        return [RAGAnswer(text=t, chunks=c) for t, c in zip(texts, chunks_per_query, strict=True)]

    def _retrieve(self, query: str) -> list[DocumentChunk]:
        """
        Run the retriever with configured settings.

        :param query: User query.
        :return: Reranked chunks.
        """
        return self._retriever.search(
            query=query,
            top_k=self._settings.top_k,
            top_kr=self._settings.top_kr,
            strategy=self._settings.strategy,
        )

    def _render_messages(self, query: str, chunks: list[DocumentChunk]) -> list[BaseMessage]:
        """
        Render the chat prompt with the chunks joined as a context block.

        :param query: User query.
        :param chunks: Chunks to embed into the context slot.
        :return: Chat messages ready for the LLM.
        """
        context = "\n---\n".join(c.text for c in chunks)
        template = load_prompt(self._settings.prompt_name)
        return template.format_messages(context=context, question=query)
