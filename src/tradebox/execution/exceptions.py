"""Custom exceptions for execution module."""


class BrokerException(Exception):
    """
    Base exception for broker errors.

    All broker-related exceptions inherit from this class.
    """

    pass


class OrderPlacementException(BrokerException):
    """
    Exception raised when order placement fails.

    Example:
        >>> if insufficient_funds:
        ...     raise OrderPlacementException("Insufficient cash for order")
    """

    pass


class OrderCancellationException(BrokerException):
    """
    Exception raised when order cancellation fails.

    Example:
        >>> if order_already_filled:
        ...     raise OrderCancellationException("Cannot cancel filled order")
    """

    pass


class InsufficientFundsException(BrokerException):
    """
    Exception raised when there are insufficient funds for an operation.

    Example:
        >>> if order_value > available_cash:
        ...     raise InsufficientFundsException(
        ...         f"Need ₹{order_value}, have ₹{available_cash}"
        ...     )
    """

    pass


class RateLimitException(BrokerException):
    """
    Exception raised when rate limit is exceeded.

    Example:
        >>> if orders_per_second > max_orders_per_second:
        ...     raise RateLimitException("Exceeded rate limit: 10 orders/second")
    """

    pass


class ReconciliationException(BrokerException):
    """
    Exception raised during reconciliation.

    Example:
        >>> if critical_discrepancy:
        ...     raise ReconciliationException(
        ...         "Position quantity mismatch: cache=100, broker=50"
        ...     )
    """

    pass


class CircuitBreakerTriggered(BrokerException):
    """
    Exception raised when circuit breaker is triggered.

    Example:
        >>> if drawdown > max_drawdown:
        ...     raise CircuitBreakerTriggered("Trading halted: 10% drawdown limit")
    """

    pass
