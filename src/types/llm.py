from typing import Protocol, TypeVar, runtime_checkable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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

    def invoke_structured(self, messages: list[BaseMessage], schema: type[T]) -> T:
        """
        Generate a typed reply parsed against the given schema.

        :param messages: Ordered chat messages forming the prompt.
        :param schema: Pydantic model that defines the expected reply shape.
        :return: Parsed instance of the provided schema.
        """
        ...
