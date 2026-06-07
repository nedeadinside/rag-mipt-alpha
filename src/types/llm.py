from typing import Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@runtime_checkable
class LLM(Protocol):
    """
    LLM contract.
    """

    def invoke(self, messages: list[BaseMessage]) -> str:
        """
        Generate a reply for a single chat conversation.

        :param messages: Ordered chat messages forming the prompt.
        :return: Assistant reply text.
        """
        ...

    def invoke_batch(self, batches: list[list[BaseMessage]]) -> list[str]:
        """
        Generate replies for several chat conversations in one call.

        :param batches: Conversations to run, each a list of chat messages.
        :return: Assistant reply texts in input order.
        """
        ...
