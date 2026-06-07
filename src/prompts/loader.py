from functools import cache, lru_cache
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

_TEMPLATES_PATH = Path(__file__).parent / "templates.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, dict[str, str]]:
    """
    Load the raw prompt registry from disk once.

    :return: Mapping from prompt name to its system and user template strings.
    """
    with _TEMPLATES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def load_prompt(name: str) -> ChatPromptTemplate:
    """
    Build a chat template from the registry by name.

    :param name: Prompt identifier in the registry.
    :return: Chat template with system and user messages.
    """
    raw = _load_raw()
    if name not in raw:
        raise KeyError(f"Unknown prompt: {name}")
    block = raw[name]
    return ChatPromptTemplate.from_messages(
        [("system", block["system"]), ("user", block["user"])],
    )
