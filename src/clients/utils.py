from collections.abc import Callable
from typing import ParamSpec, TypeVar

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)


def llm_retry(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: tuple[type[BaseException], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Build a retry decorator for transient transport errors on chat calls.

    :param max_attempts: Total attempts before giving up.
    :param initial_wait: Base seconds for exponential backoff.
    :param max_wait: Upper bound for backoff sleep in seconds.
    :param exceptions: Exception types that trigger a retry.
    :return: Decorator that wraps a callable with retry behavior.
    """
    return retry(
        retry=retry_if_exception_type(exceptions),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        reraise=True,
    )
