from enum import StrEnum


class SearchStrategy(StrEnum):
    """
    Retrieval strategy selector.
    """

    DEFAULT = "default"
    MULTIQUERY = "multiquery"
