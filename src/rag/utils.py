from langchain_core.messages import BaseMessage

from src.prompts import load_prompt
from src.types.document import DocumentChunk


def build_messages(
    query: str,
    chunks: list[DocumentChunk],
    prompt_name: str,
) -> list[BaseMessage]:
    """
    Render the chat prompt with the chunks joined as a context block.

    :param query: User query.
    :param chunks: Chunks to embed into the context slot.
    :param prompt_name: Prompt template name to load.
    :return: Chat messages ready for the LLM.
    """
    context = "\n---\n".join(c.text for c in chunks)
    template = load_prompt(prompt_name)
    return template.format_messages(context=context, question=query)
