from typing import Protocol, runtime_checkable


@runtime_checkable
class QueryExpanderProtocol(Protocol):
    """
    Query expander contract.
    """

    def expand(self, query: str) -> list[str]:
        """
        Produce alternative search queries from the original query.

        :param query: Original user query.
        :return: Expanded search queries.
        """
        ...
