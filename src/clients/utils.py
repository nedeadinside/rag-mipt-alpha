from collections.abc import Callable
from typing import ParamSpec, TypeVar

import httpx
from ollama._types import ResponseError
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
)

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
    ResponseError,
)


def llm_retry(
    initial_wait: float = 3.0,
    max_wait: float = 20.0,
    exceptions: tuple[type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Build a retry decorator for transient transport errors on chat calls.

    :param initial_wait: Base seconds for exponential backoff.
    :param max_wait: Upper bound for backoff sleep in seconds.
    :param exceptions: Exception types that trigger a retry.
    :return: Decorator that wraps a callable with retry behavior.
    """
    return retry(
        retry=retry_if_exception_type(exceptions),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        reraise=True,
    )
