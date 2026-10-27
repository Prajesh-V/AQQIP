"""
AQQIP Retry & Backoff Configuration

Centralizes retry policies using Tenacity.
"""
from __future__ import annotations

import requests
from tenacity import (
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    RetryCallState,
    retry,
    retry_if_result
)

from src.utils.logging import get_logger

logger = get_logger("retry")


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts."""
    if retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        logger.warning(
            "Retrying after exception",
            attempt=retry_state.attempt_number,
            func=retry_state.fn.__name__,
            exc=str(exc)
        )
    else:
        logger.warning(
            "Retrying after unexpected result",
            attempt=retry_state.attempt_number,
            func=retry_state.fn.__name__,
            result=retry_state.outcome.result()
        )


def is_retryable_request_error(exception: BaseException) -> bool:
    """Check if the exception is a retryable requests exception."""
    if isinstance(exception, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exception, requests.HTTPError):
        status = exception.response.status_code
        return status in {429, 500, 502, 503, 504}
    return False


# Standard decorator for network requests (e.g., SEC EDGAR, yfinance fallback)
# Stops after 5 attempts, waits 2^x * 1 second between retries, max 60s
with_network_retry = retry(
    retry=retry_if_exception_type(Exception) & retry_if_exception_type(requests.RequestException) ,
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    after=_log_retry,
    reraise=True
)

# Specifically for HTTP errors that should be retried based on our function above
with_http_retry = retry(
    retry=retry_if_exception_type(Exception),  # We'll handle check in retry arg
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    after=_log_retry,
    reraise=True
)

# Custom decorator logic for HTTP where we can pass the predicate
def create_network_retry():
    return retry(
        retry=retry_if_exception_type(requests.RequestException) | retry_if_exception_type(TimeoutError),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        stop=stop_after_attempt(5),
        after=_log_retry,
        reraise=True
    )

network_retry = create_network_retry()
