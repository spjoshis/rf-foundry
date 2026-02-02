"""Retry logic with exponential backoff for API calls."""

import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, Tuple, Type

from loguru import logger


class RetryableException(Exception):
    """
    Exception that indicates a failed operation should be retried.

    Raised for transient failures like network errors, timeouts,
    or temporary API unavailability (503, 504 errors).

    Example:
        >>> if response.status_code == 503:
        ...     raise RetryableException("Service unavailable")
    """

    pass


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay_seconds: Initial delay before first retry (default: 1.0)
        max_delay_seconds: Maximum delay between retries (default: 30.0)
        backoff_multiplier: Exponential backoff multiplier (default: 2.0)
        jitter: Add random jitter to delays (default: True)

    Example:
        >>> config = RetryConfig(max_retries=5, initial_delay_seconds=2.0)
        >>> # Retry delays: 2s, 4s, 8s, 16s, 30s (capped at max_delay)
    """

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


class RetryHandler:
    """
    Handles retry logic with exponential backoff.

    Can be used as a decorator or called directly.

    Example:
        >>> retry_handler = RetryHandler(RetryConfig(max_retries=3))
        >>>
        >>> @retry_handler.retry()
        >>> def fetch_data():
        ...     # This will retry on RetryableException
        ...     response = api.get("/data")
        ...     if response.status_code == 503:
        ...         raise RetryableException("Service unavailable")
        ...     return response.json()
    """

    def __init__(self, config: Optional[RetryConfig] = None) -> None:
        """
        Initialize retry handler.

        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()

    def retry(
        self,
        retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableException,),
    ) -> Callable:
        """
        Decorator for adding retry logic to a function.

        Args:
            retryable_exceptions: Tuple of exception types to retry on

        Returns:
            Decorated function with retry logic

        Example:
            >>> handler = RetryHandler()
            >>> @handler.retry()
            >>> def my_function():
            ...     # Function implementation
            ...     pass
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(
                    func,
                    retryable_exceptions,
                    *args,
                    **kwargs,
                )

            return wrapper

        return decorator

    def _execute_with_retry(
        self,
        func: Callable,
        retryable_exceptions: Tuple[Type[Exception], ...],
        *args,
        **kwargs,
    ):
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            retryable_exceptions: Exceptions to retry on
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Function result

        Raises:
            RetryableException: After all retries exhausted
            Exception: If non-retryable exception occurs
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Execute function
                result = func(*args, **kwargs)

                # Success
                if attempt > 0:
                    logger.info(
                        f"{func.__name__} succeeded on attempt {attempt + 1}"
                    )

                return result

            except retryable_exceptions as e:
                last_exception = e

                # Check if we should retry
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/"
                        f"{self.config.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)
                else:
                    # All retries exhausted
                    logger.error(
                        f"{func.__name__} failed after {attempt + 1} attempts: {e}"
                    )
                    raise RetryableException(
                        f"Failed after {attempt + 1} attempts: {e}"
                    ) from e

            except Exception as e:
                # Non-retryable exception - propagate immediately
                logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                raise

        # Should never reach here, but just in case
        raise RetryableException(
            f"Failed after {self.config.max_retries + 1} attempts"
        ) from last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: delay = initial * (multiplier ^ attempt)
        delay = self.config.initial_delay_seconds * (
            self.config.backoff_multiplier**attempt
        )

        # Cap at max delay
        delay = min(delay, self.config.max_delay_seconds)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)

    @staticmethod
    def with_retry(
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> Callable:
        """
        Convenience decorator for simple retry logic.

        Args:
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Decorator function

        Example:
            >>> @RetryHandler.with_retry(max_retries=5)
            >>> def my_api_call():
            ...     # Implementation
            ...     pass
        """
        config = RetryConfig(
            max_retries=max_retries,
            initial_delay_seconds=initial_delay,
            max_delay_seconds=max_delay,
        )

        handler = RetryHandler(config)
        return handler.retry()
