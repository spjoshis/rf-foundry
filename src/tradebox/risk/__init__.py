"""
Risk management module for trading system.

Provides pre-trade validators, circuit breakers, and position sizing
strategies to manage trading risk and prevent catastrophic losses.
"""

from tradebox.risk.circuit_breakers import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    DrawdownCircuitBreaker,
    DailyLossCircuitBreaker,
    ConsecutiveLossCircuitBreaker,
    APIFailureCircuitBreaker,
)
from tradebox.risk.position_sizers import (
    PositionSizer,
    PositionSizerConfig,
    KellyEstimator,
    FixedPositionSizer,
    VolatilityPositionSizer,
    RiskParityPositionSizer,
    KellyPositionSizer,
    create_position_sizer,
)
from tradebox.risk.validators import (
    RiskConfig,
    RiskManager,
    RiskValidator,
    MaxPositionSizeValidator,
    MaxStockAllocationValidator,
    MaxDailyLossValidator,
    LiquidityValidator,
    MaxLeverageValidator,
)

__all__ = [
    # Circuit Breakers
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerManager",
    "DrawdownCircuitBreaker",
    "DailyLossCircuitBreaker",
    "ConsecutiveLossCircuitBreaker",
    "APIFailureCircuitBreaker",
    # Position Sizers
    "PositionSizer",
    "PositionSizerConfig",
    "KellyEstimator",
    "FixedPositionSizer",
    "VolatilityPositionSizer",
    "RiskParityPositionSizer",
    "KellyPositionSizer",
    "create_position_sizer",
    # Validators
    "RiskConfig",
    "RiskManager",
    "RiskValidator",
    "MaxPositionSizeValidator",
    "MaxStockAllocationValidator",
    "MaxDailyLossValidator",
    "LiquidityValidator",
    "MaxLeverageValidator",
]
