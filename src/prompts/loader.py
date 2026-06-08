from functools import cache, lru_cache
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

_TEMPLATES_PATH = Path(__file__).parent / "templates.yaml"


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, dict]:
    """
    Load the raw prompt registry from disk once.

    :return: Mapping from prompt name to either a system/user block or a nested group.
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


@cache
def load_text(name: str) -> str:
    """
    Load a plain text entry from the registry by name.

    :param name: Text identifier in the registry.
    :return: Plain string content.
    """
    raw = _load_raw()
    if name not in raw:
        raise KeyError(f"Unknown text: {name}")
    value = raw[name]
    if not isinstance(value, str):
        raise TypeError(f"Registry entry {name!r} is not a string")
    return value


@cache
def load_prompt_group(name: str) -> list[ChatPromptTemplate]:
    """
    Build a list of chat templates from a nested registry entry.

    :param name: Group identifier in the registry.
    :return: Chat templates in registry-declared order.
    """
    raw = _load_raw()
    if name not in raw:
        raise KeyError(f"Unknown prompt group: {name}")
    group = raw[name]
    return [
        ChatPromptTemplate.from_messages(
            [("system", block["system"]), ("user", block["user"])],
        )
        for block in group.values()
    ]
