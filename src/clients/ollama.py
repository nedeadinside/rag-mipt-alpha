from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama


class OllamaLLM:
    """
    LLM client backed by Ollama.
    """

    def __init__(self, model_name: str, base_url: str, temperature: float) -> None:
        """
        Initialize the client.

        :param model_name: Ollama model identifier.
        :param base_url: Ollama server base URL.
        :param temperature: Sampling temperature.
        """
        self._model = ChatOllama(model=model_name, base_url=base_url, temperature=temperature)

    def invoke(self, messages: list[BaseMessage]) -> str:
        """
        Generate a reply for a single chat conversation.

        :param messages: Ordered chat messages forming the prompt.
        :return: Assistant reply text.
        """
        result = self._model.invoke(messages)
        return str(result.content)

    def invoke_batch(self, batches: list[list[BaseMessage]]) -> list[str]:
        """
        Generate replies for several chat conversations in one call.

        :param batches: Conversations to run, each a list of chat messages.
        :return: Assistant reply texts in input order.
        """
        results = self._model.batch(batches)
        return [str(r.content) for r in results]
