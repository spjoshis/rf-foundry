"""
Execution module for order placement and broker integration.

Provides paper trading and live trading broker implementations
with a unified interface.
"""

from tradebox.execution.base_broker import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
    Position,
)
from tradebox.execution.config import load_kite_broker_config
from tradebox.execution.exceptions import (
    BrokerException,
    CircuitBreakerTriggered,
    InsufficientFundsException,
    OrderCancellationException,
    OrderPlacementException,
    RateLimitException,
    ReconciliationException,
)
from tradebox.execution.paper_broker import PaperBroker
from tradebox.execution.retry import RetryConfig, RetryHandler, RetryableException

# Optional Kite broker (requires kiteconnect package)
try:
    from tradebox.execution.kite_broker import KiteBroker, KiteBrokerConfig
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    KiteBroker = None
    KiteBrokerConfig = None

__all__ = [
    # Base classes and models
    "BaseBroker",
    "Order",
    "OrderSide",
    "OrderStatus",
    "Portfolio",
    "Position",
    # Broker implementations
    "PaperBroker",
    # Configuration
    "load_kite_broker_config",
    # Retry logic
    "RetryHandler",
    "RetryConfig",
    "RetryableException",
    # Exceptions
    "BrokerException",
    "OrderPlacementException",
    "OrderCancellationException",
    "InsufficientFundsException",
    "RateLimitException",
    "ReconciliationException",
    "CircuitBreakerTriggered",
]

# Add Kite broker to exports if available
if KITE_AVAILABLE:
    __all__.extend(["KiteBroker", "KiteBrokerConfig"])
