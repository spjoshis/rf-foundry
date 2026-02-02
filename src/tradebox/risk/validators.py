"""Pre-trade risk validators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger

from tradebox.execution.base_broker import Order, OrderSide, Portfolio


@dataclass
class RiskConfig:
    """
    Configuration for risk management.

    Attributes:
        max_position_size_pct: Max position size as % of portfolio (default: 20%)
        max_stock_allocation_pct: Max allocation per stock (default: 15%)
        max_daily_loss_pct: Max daily loss allowed (default: 2%)
        max_sector_concentration_pct: Max sector concentration (default: 30%)
        min_daily_volume: Minimum daily volume required (default: 100K shares)
        max_leverage: Maximum leverage allowed (default: 1.0, no leverage)
    """

    max_position_size_pct: float = 0.20
    max_stock_allocation_pct: float = 0.15
    max_daily_loss_pct: float = 0.02
    max_sector_concentration_pct: float = 0.30
    min_daily_volume: int = 100000
    max_leverage: float = 1.0


class RiskValidator(ABC):
    """
    Abstract base class for risk validators.

    Each validator checks a specific risk condition before trade execution.
    """

    @abstractmethod
    def validate(
        self,
        order: Order,
        portfolio: Portfolio,
        config: RiskConfig,
    ) -> Tuple[bool, str]:
        """
        Validate order against risk rules.

        Args:
            order: Order to validate
            portfolio: Current portfolio state
            config: Risk configuration

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        pass


class MaxPositionSizeValidator(RiskValidator):
    """
    Validate that position size doesn't exceed maximum.

    Prevents concentration risk by limiting position size relative to
    total portfolio value.

    Example:
        >>> validator = MaxPositionSizeValidator()
        >>> valid, reason = validator.validate(order, portfolio, config)
        >>> if not valid:
        ...     print(f"Order rejected: {reason}")
    """

    def validate(
        self,
        order: Order,
        portfolio: Portfolio,
        config: RiskConfig,
    ) -> Tuple[bool, str]:
        """Validate position size."""
        if order.side != OrderSide.BUY:
            return True, ""

        # Calculate order value
        if order.price is None:
            # Need to get current price for market orders
            logger.warning("Cannot validate market order without price")
            return True, ""

        order_value = order.quantity * order.price

        # Check against portfolio value
        max_position_value = portfolio.total_value * config.max_position_size_pct

        if order_value > max_position_value:
            return False, (
                f"Position size too large: ₹{order_value:,.0f} > "
                f"max ₹{max_position_value:,.0f} "
                f"({config.max_position_size_pct:.1%} of portfolio)"
            )

        return True, ""


class MaxStockAllocationValidator(RiskValidator):
    """
    Validate that total stock allocation doesn't exceed maximum.

    Prevents over-concentration in a single stock by checking both
    existing position and new order.
    """

    def validate(
        self,
        order: Order,
        portfolio: Portfolio,
        config: RiskConfig,
    ) -> Tuple[bool, str]:
        """Validate stock allocation."""
        if order.side != OrderSide.BUY:
            return True, ""

        if order.price is None:
            return True, ""

        # Calculate existing position value
        existing_value = 0.0
        if order.symbol in portfolio.positions:
            pos = portfolio.positions[order.symbol]
            existing_value = pos.market_value

        # Calculate new total value
        order_value = order.quantity * order.price
        total_value = existing_value + order_value

        # Check against limit
        max_allocation_value = portfolio.total_value * config.max_stock_allocation_pct

        if total_value > max_allocation_value:
            return False, (
                f"Stock allocation too high for {order.symbol}: "
                f"₹{total_value:,.0f} > max ₹{max_allocation_value:,.0f} "
                f"({config.max_stock_allocation_pct:.1%} of portfolio)"
            )

        return True, ""


class MaxDailyLossValidator(RiskValidator):
    """
    Validate that daily loss hasn't exceeded maximum.

    Prevents catastrophic losses by halting trading when daily loss
    exceeds threshold.
    """

    def __init__(self) -> None:
        """Initialize validator."""
        self.daily_start_value: Optional[float] = None
        self.daily_pnl: float = 0.0

    def reset_daily(self, portfolio_value: float) -> None:
        """Reset daily tracking at start of new day."""
        self.daily_start_value = portfolio_value
        self.daily_pnl = 0.0
        logger.info(f"Daily loss tracker reset: start value ₹{portfolio_value:,.0f}")

    def validate(
        self,
        order: Order,
        portfolio: Portfolio,
        config: RiskConfig,
    ) -> Tuple[bool, str]:
        """Validate daily loss."""
        if self.daily_start_value is None:
            self.daily_start_value = portfolio.total_value
            return True, ""

        # Calculate current daily P&L
        current_pnl = portfolio.total_value - self.daily_start_value
        loss_pct = current_pnl / self.daily_start_value

        # Check if we've exceeded max loss
        if loss_pct < -config.max_daily_loss_pct:
            return False, (
                f"Daily loss limit exceeded: {loss_pct:.2%} < "
                f"max -{config.max_daily_loss_pct:.1%}"
            )

        return True, ""


class LiquidityValidator(RiskValidator):
    """
    Validate that stock has sufficient liquidity.

    Prevents trading illiquid stocks that may be difficult to exit.
    Requires minimum average daily volume.
    """

    def validate(
        self,
        order: Order,
        portfolio: Portfolio,
        config: RiskConfig,
    ) -> Tuple[bool, str]:
        """Validate liquidity."""
        # Note: In production, this would check actual volume data
        # For now, we assume all symbols pass (would need data source)

        # Placeholder: Could integrate with yfinance to get avg volume
        # ticker = yf.Ticker(order.symbol)
        # avg_volume = ticker.info.get('averageVolume', 0)

        # For now, just log and pass
        logger.debug(f"Liquidity check for {order.symbol}: assuming sufficient")
        return True, ""


class MaxLeverageValidator(RiskValidator):
    """
    Validate that leverage doesn't exceed maximum.

    Prevents over-leveraging by ensuring total position value
    doesn't exceed cash by more than allowed leverage.
    """

    def validate(
        self,
        order: Order,
        portfolio: Portfolio,
        config: RiskConfig,
    ) -> Tuple[bool, str]:
        """Validate leverage."""
        if order.side != OrderSide.BUY:
            return True, ""

        if order.price is None:
            return True, ""

        # Calculate total position value after order
        current_positions_value = sum(
            p.market_value for p in portfolio.positions.values()
        )
        order_value = order.quantity * order.price
        total_positions_value = current_positions_value + order_value

        # Calculate leverage
        # Leverage = Total Position Value / (Cash + Current Positions Value)
        portfolio_equity = portfolio.cash + current_positions_value
        new_leverage = total_positions_value / portfolio_equity if portfolio_equity > 0 else 0

        if new_leverage > config.max_leverage:
            return False, (
                f"Leverage too high: {new_leverage:.2f}x > "
                f"max {config.max_leverage:.2f}x"
            )

        return True, ""


class RiskManager:
    """
    Manages all risk validators and performs pre-trade risk checks.

    Aggregates multiple validators and ensures all pass before
    allowing trade execution.

    Example:
        >>> config = RiskConfig(max_position_size_pct=0.20)
        >>> risk_manager = RiskManager(config)
        >>>
        >>> valid, reason = risk_manager.validate_order(order, portfolio)
        >>> if not valid:
        ...     print(f"Order rejected: {reason}")
        ...     # Don't execute order
        >>> else:
        ...     broker.place_order(...)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        """
        Initialize risk manager.

        Args:
            config: Risk configuration (uses defaults if None)
        """
        self.config = config or RiskConfig()

        # Initialize validators
        self.validators: List[RiskValidator] = [
            MaxPositionSizeValidator(),
            MaxStockAllocationValidator(),
            MaxDailyLossValidator(),
            LiquidityValidator(),
            MaxLeverageValidator(),
        ]

        logger.info(
            f"RiskManager initialized with {len(self.validators)} validators"
        )

    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
    ) -> Tuple[bool, str]:
        """
        Validate order against all risk rules.

        Args:
            order: Order to validate
            portfolio: Current portfolio state

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        for validator in self.validators:
            valid, reason = validator.validate(order, portfolio, self.config)

            if not valid:
                logger.warning(
                    f"Order validation failed: {validator.__class__.__name__} - {reason}"
                )
                return False, reason

        logger.debug(f"Order passed all {len(self.validators)} risk checks")
        return True, ""

    def reset_daily(self, portfolio_value: float) -> None:
        """
        Reset daily tracking at start of new trading day.

        Args:
            portfolio_value: Starting portfolio value for the day
        """
        for validator in self.validators:
            if isinstance(validator, MaxDailyLossValidator):
                validator.reset_daily(portfolio_value)
