from src.prompts import load_prompt
from src.types.document import DocumentChunk
from src.types.llm import LLM
from src.types.verifier import VerificationResult


class LLMVerifier:
    """
    Verifier that asks the model to judge fragment-question relevance with structured output.
    """

    def __init__(self, llm: LLM, prompt_name: str = "verifier") -> None:
        """
        Initialize the verifier.

        :param llm: Chat model used for the relevance judgement.
        :param prompt_name: Prompt name in the registry.
        """
        self._llm = llm
        self._prompt_name = prompt_name

    def verify(self, query: str, chunks: list[DocumentChunk]) -> VerificationResult:
        """
        Decide whether the fragments are sufficient to answer the question.

        :param query: User query.
        :param chunks: Retrieved supporting fragments.
        :return: Relevance decision with a short rationale.
        """
        template = load_prompt(self._prompt_name)
        context = "\n---\n".join(c.text for c in chunks)
        messages = template.format_messages(context=context, question=query)
        return self._llm.invoke_structured(messages, VerificationResult)
