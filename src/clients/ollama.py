from typing import TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.clients.utils import DEFAULT_RETRY_EXCEPTIONS, llm_retry

T = TypeVar("T", bound=BaseModel)


class OllamaLLM:
    """
    LLM client backed by Ollama.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        """
        Initialize the client.

        :param model_name: Ollama model identifier.
        :param base_url: Ollama server base URL.
        :param temperature: Sampling temperature.
        :param top_p: Nucleus sampling cutoff; None keeps server default.
        :param top_k: Top-k sampling cutoff; None keeps server default.
        """
        kwargs: dict[str, object] = {
            "model": model_name,
            "base_url": base_url,
            "temperature": temperature,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        self._model = ChatOllama(**kwargs)

    @llm_retry()
    def invoke(self, messages: list[BaseMessage]) -> str:
        """
        Generate a reply for a single chat conversation.

        :param messages: Ordered chat messages forming the prompt.
        :return: Assistant reply text.
        """
        result = self._model.invoke(messages)
        return str(result.content)

    @llm_retry()
    def invoke_batch(self, batches: list[list[BaseMessage]]) -> list[str]:
        """
        Generate replies for several chat conversations in one call.

        :param batches: Conversations to run, each a list of chat messages.
        :return: Assistant reply texts in input order.
        """
        results = self._model.batch(batches)
        return [str(r.content) for r in results]

    @llm_retry(exceptions=(*DEFAULT_RETRY_EXCEPTIONS, OutputParserException))
    def invoke_structured(self, messages: list[BaseMessage], schema: type[T]) -> T:
        """
        Generate a typed reply parsed against the given schema.

        :param messages: Ordered chat messages forming the prompt.
        :param schema: Pydantic model that defines the expected reply shape.
        :return: Parsed instance of the provided schema.
        """
        runnable = self._model.with_structured_output(schema)
        return runnable.invoke(messages)
