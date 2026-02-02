"""Configuration for backtesting."""

from dataclasses import dataclass
from typing import Literal, Optional

from loguru import logger


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters.

    Attributes:
        initial_capital: Starting capital for backtest (default: ₹100,000)
        commission_pct: Commission percentage per trade (default: 0.0015)
        slippage_pct: Slippage percentage (default: 0.001)
        position_sizing: Position sizing method ('fixed', 'kelly', 'volatility')
        max_position_size_pct: Maximum position size as % of portfolio (default: 20%)
        benchmark: Benchmark to compare against ('buy_and_hold', 'nifty50', 'sma_crossover')
        risk_free_rate: Annual risk-free rate for Sharpe calculation (default: 6%)
        trading_days_per_year: Trading days for annualization (default: 252)

    Example:
        >>> config = BacktestConfig(
        ...     initial_capital=500000,
        ...     commission_pct=0.001,
        ...     position_sizing='kelly'
        ... )
    """

    initial_capital: float = 100000.0
    commission_pct: float = 0.0015
    slippage_pct: float = 0.001
    position_sizing: Literal["fixed", "kelly", "volatility"] = "fixed"
    max_position_size_pct: float = 0.20
    benchmark: Literal["buy_and_hold", "nifty50", "sma_crossover"] = "buy_and_hold"
    risk_free_rate: float = 0.06
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be positive, got {self.initial_capital}"
            )
        if not 0 <= self.commission_pct < 1:
            raise ValueError(
                f"commission_pct must be in [0, 1), got {self.commission_pct}"
            )
        if not 0 <= self.slippage_pct < 1:
            raise ValueError(
                f"slippage_pct must be in [0, 1), got {self.slippage_pct}"
            )
        if not 0 < self.max_position_size_pct <= 1:
            raise ValueError(
                f"max_position_size_pct must be in (0, 1], got {self.max_position_size_pct}"
            )
        if not 0 <= self.risk_free_rate < 1:
            raise ValueError(
                f"risk_free_rate must be in [0, 1), got {self.risk_free_rate}"
            )

        logger.debug(f"BacktestConfig initialized: {self}")
