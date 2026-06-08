from src.prompts.loader import load_prompt_group
from src.types.llm import LLM


class MultiQueryExpander:
    """
    Query expander that asks the model for several reformulations of the user query.
    """

    def __init__(self, llm: LLM, prompt_name: str = "multi_query") -> None:
        """
        Initialize the expander.

        :param llm: Chat model used to generate reformulations.
        :param prompt_name: Prompt group name in the registry.
        """
        self._llm = llm
        self._prompt_name = prompt_name

    def expand(self, query: str) -> list[str]:
        """
        Generate alternative search queries for the given user query.

        :param query: Original user query.
        :return: Expanded search queries.
        """
        templates = load_prompt_group(self._prompt_name)
        batches = [t.format_messages(question=query) for t in templates]
        outputs = self._llm.invoke_batch(batches)
        return [o.strip() for o in outputs]
