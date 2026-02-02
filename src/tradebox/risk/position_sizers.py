"""Position sizing strategies for risk management in trading.

This module implements various position sizing strategies including:
- Fixed position sizing (baseline)
- Kelly Criterion (optimal growth)
- Volatility-based sizing (ATR method)
- Risk Parity (volatility targeting)

All strategies are designed to work with the Discrete(3) action space
{Hold, Buy, Sell} and provide fractional position sizes when buying.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class PositionSizerConfig:
    """
    Configuration for position sizing strategies.

    Attributes:
        strategy: Position sizing strategy ('fixed', 'kelly', 'volatility', 'risk_parity')

        # Universal parameters
        min_position: Minimum position size as fraction of portfolio (default: 0.10 = 10%)
        max_position: Maximum position size as fraction of portfolio (default: 0.20 = 20%)

        # Fixed strategy parameters
        fixed_fraction: Fixed position size fraction (default: 0.20 = 20%)

        # Kelly strategy parameters
        kelly_fraction: Fractional Kelly multiplier (default: 0.25 = quarter-Kelly)
        kelly_alpha: EMA smoothing factor for Kelly estimation (default: 0.05)
        kelly_min_trades: Minimum trades before trusting Kelly estimates (default: 20)
        kelly_prior_win_rate: Bayesian prior for win rate (default: 0.50 = 50%)
        kelly_prior_payoff: Bayesian prior for payoff ratio (default: 1.5)

        # Volatility strategy parameters
        vol_risk_per_trade: Risk per trade as fraction of portfolio (default: 0.02 = 2%)
        vol_atr_period: ATR period for volatility calculation (default: 14)
        vol_atr_multiplier: Stop loss distance in ATR units (default: 2.0)

        # Risk Parity strategy parameters
        rp_target_volatility: Target portfolio volatility (default: 0.12 = 12% annual)
        rp_volatility_window: Rolling window for volatility calculation (default: 60)
    """

    strategy: str = "fixed"

    # Universal parameters
    min_position: float = 0.10
    max_position: float = 0.20

    # Fixed strategy
    fixed_fraction: float = 0.20

    # Kelly strategy
    kelly_fraction: float = 0.25
    kelly_alpha: float = 0.05
    kelly_min_trades: int = 20
    kelly_prior_win_rate: float = 0.50
    kelly_prior_payoff: float = 1.5

    # Volatility strategy
    vol_risk_per_trade: float = 0.02
    vol_atr_period: int = 14
    vol_atr_multiplier: float = 2.0

    # Risk Parity strategy
    rp_target_volatility: float = 0.12
    rp_volatility_window: int = 60


class KellyEstimator:
    """
    Estimates win rate and payoff ratio for Kelly Criterion using EMA.

    Uses exponential moving average to track:
    - Win rate (probability of winning)
    - Average win size
    - Average loss size

    Incorporates Bayesian priors for robustness with limited trade history.

    Example:
        >>> estimator = KellyEstimator(alpha=0.05, min_trades=20)
        >>> # After each trade, update with P&L percentage
        >>> estimator.update(pnl_pct=0.05)  # 5% gain
        >>> estimator.update(pnl_pct=-0.02)  # 2% loss
        >>> kelly_f = estimator.get_kelly_fraction(kelly_fraction=0.25)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_trades: int = 20,
        prior_win_rate: float = 0.50,
        prior_payoff: float = 1.5,
    ) -> None:
        """
        Initialize Kelly Estimator.

        Args:
            alpha: EMA smoothing factor (0.05 = ~40 trade half-life)
            min_trades: Minimum trades before trusting estimates
            prior_win_rate: Bayesian prior for win rate (0.0 to 1.0)
            prior_payoff: Bayesian prior for payoff ratio (avg_win / avg_loss)

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if min_trades < 1:
            raise ValueError(f"min_trades must be >= 1, got {min_trades}")
        if not 0 < prior_win_rate < 1:
            raise ValueError(f"prior_win_rate must be in (0, 1), got {prior_win_rate}")
        if prior_payoff <= 0:
            raise ValueError(f"prior_payoff must be > 0, got {prior_payoff}")

        self.alpha = alpha
        self.min_trades = min_trades
        self.prior_win_rate = prior_win_rate
        self.prior_payoff = prior_payoff

        # Running statistics (EMA)
        self.ema_win_rate = prior_win_rate
        self.ema_avg_win = 0.0
        self.ema_avg_loss = 0.0
        self.trade_count = 0

        logger.debug(
            f"Initialized KellyEstimator: alpha={alpha}, min_trades={min_trades}, "
            f"prior_win_rate={prior_win_rate}, prior_payoff={prior_payoff}"
        )

    def update(self, pnl_pct: float) -> None:
        """
        Update Kelly estimates with new trade result.

        Args:
            pnl_pct: Trade P&L as percentage (e.g., 0.05 for 5% gain, -0.02 for 2% loss)
        """
        self.trade_count += 1
        is_win = pnl_pct > 0

        # Update EMA win rate
        self.ema_win_rate = self.alpha * float(is_win) + (1 - self.alpha) * self.ema_win_rate

        # Update EMA average win/loss separately
        if is_win:
            if self.ema_avg_win == 0:
                self.ema_avg_win = pnl_pct
            else:
                self.ema_avg_win = self.alpha * pnl_pct + (1 - self.alpha) * self.ema_avg_win
        else:
            loss_magnitude = abs(pnl_pct)
            if self.ema_avg_loss == 0:
                self.ema_avg_loss = loss_magnitude
            else:
                self.ema_avg_loss = (
                    self.alpha * loss_magnitude + (1 - self.alpha) * self.ema_avg_loss
                )

        logger.debug(
            f"Kelly update (trade #{self.trade_count}): pnl={pnl_pct:.4f}, "
            f"win_rate={self.ema_win_rate:.3f}, avg_win={self.ema_avg_win:.4f}, "
            f"avg_loss={self.ema_avg_loss:.4f}"
        )

    def get_kelly_fraction(self, kelly_fraction: float = 0.25) -> float:
        """
        Calculate position size using fractional Kelly Criterion.

        Uses Bayesian blending of priors and observed data when trade history
        is insufficient (< min_trades).

        Args:
            kelly_fraction: Fractional Kelly multiplier (0.25 = quarter-Kelly)

        Returns:
            Position size as fraction of portfolio (0.0 to 1.0+, can exceed 1.0)

        Note:
            Returned value can exceed 1.0 (indicating high confidence).
            Caller should clip to max_position limit.
        """
        # Use Bayesian blending if insufficient data
        if self.trade_count < self.min_trades:
            # Weight: 0 at trade_count=0, 1.0 at trade_count=min_trades
            weight = self.trade_count / self.min_trades

            # Blend win rate
            p = weight * self.ema_win_rate + (1 - weight) * self.prior_win_rate

            # Blend payoff ratio
            if self.ema_avg_win > 0 and self.ema_avg_loss > 0:
                observed_payoff = self.ema_avg_win / self.ema_avg_loss
                b = weight * observed_payoff + (1 - weight) * self.prior_payoff
            else:
                b = self.prior_payoff
        else:
            # Sufficient data: use observed statistics
            p = self.ema_win_rate
            b = self.ema_avg_win / max(self.ema_avg_loss, 1e-6)  # Avoid division by zero

        q = 1 - p

        # Full Kelly calculation: f = (p × b - q) / b
        kelly_f = (p * b - q) / b

        # Apply fractional Kelly
        fractional_kelly_f = kelly_fraction * kelly_f

        logger.debug(
            f"Kelly calculation: p={p:.3f}, b={b:.3f}, "
            f"full_kelly={kelly_f:.3f}, fractional={fractional_kelly_f:.3f}"
        )

        return fractional_kelly_f

    def reset(self) -> None:
        """Reset estimator state for new episode."""
        self.ema_win_rate = self.prior_win_rate
        self.ema_avg_win = 0.0
        self.ema_avg_loss = 0.0
        self.trade_count = 0
        logger.debug("KellyEstimator reset")


class PositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.

    All position sizers must implement the calculate() method which returns
    a position size as a fraction of the portfolio (0.0 to 1.0).

    Attributes:
        min_position: Minimum position size (default: 0.10 = 10%)
        max_position: Maximum position size (default: 0.20 = 20%)
    """

    def __init__(self, min_position: float = 0.10, max_position: float = 0.20) -> None:
        """
        Initialize position sizer.

        Args:
            min_position: Minimum position size as fraction (0.0 to 1.0)
            max_position: Maximum position size as fraction (0.0 to 1.0)

        Raises:
            ValueError: If min_position > max_position or out of range
        """
        if not 0 <= min_position <= 1:
            raise ValueError(f"min_position must be in [0, 1], got {min_position}")
        if not 0 <= max_position <= 1:
            raise ValueError(f"max_position must be in [0, 1], got {max_position}")
        if min_position > max_position:
            raise ValueError(
                f"min_position ({min_position}) must be <= max_position ({max_position})"
            )

        self.min_position = min_position
        self.max_position = max_position

    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Calculate position size as fraction of portfolio.

        Returns:
            Position size in range [0.0, 1.0]

        Note:
            Implementations should clip return value to [min_position, max_position]
        """
        pass


class FixedPositionSizer(PositionSizer):
    """
    Fixed position sizing strategy (baseline).

    Always returns the same fixed fraction of portfolio, regardless of
    market conditions. Useful as a baseline for comparing adaptive strategies.

    Example:
        >>> sizer = FixedPositionSizer(fixed_fraction=0.20, max_position=0.20)
        >>> position = sizer.calculate()  # Returns 0.20 (20%)
    """

    def __init__(
        self, fixed_fraction: float = 0.20, min_position: float = 0.10, max_position: float = 0.20
    ) -> None:
        """
        Initialize fixed position sizer.

        Args:
            fixed_fraction: Fixed position size (0.0 to 1.0)
            min_position: Minimum position size (for clamping)
            max_position: Maximum position size (for clamping)
        """
        super().__init__(min_position, max_position)
        self.fixed_fraction = np.clip(fixed_fraction, min_position, max_position)
        logger.info(f"Initialized FixedPositionSizer: fraction={self.fixed_fraction:.2f}")

    def calculate(self, **kwargs) -> float:
        """
        Calculate fixed position size.

        Args:
            **kwargs: Ignored (for compatibility with other sizers)

        Returns:
            Fixed position size (e.g., 0.20 for 20%)
        """
        return self.fixed_fraction


class VolatilityPositionSizer(PositionSizer):
    """
    Volatility-based position sizing using ATR (Average True Range).

    Scales position size inversely with volatility:
    - Higher volatility → smaller position (reduce risk)
    - Lower volatility → larger position (increase exposure)

    Uses fixed risk per trade approach:
    position = (risk_dollars / stop_distance) × price / portfolio_value

    Example:
        >>> sizer = VolatilityPositionSizer(risk_per_trade=0.02, atr_multiplier=2.0)
        >>> position = sizer.calculate(
        ...     price=2500.0, atr=50.0, portfolio_value=100000.0
        ... )  # Returns position size based on 2% risk and 2×ATR stop
    """

    def __init__(
        self,
        risk_per_trade: float = 0.02,
        atr_multiplier: float = 2.0,
        min_position: float = 0.10,
        max_position: float = 0.20,
    ) -> None:
        """
        Initialize volatility-based position sizer.

        Args:
            risk_per_trade: Maximum portfolio % to risk per trade (default: 0.02 = 2%)
            atr_multiplier: Stop loss distance in ATR units (default: 2.0)
            min_position: Minimum position size
            max_position: Maximum position size

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__(min_position, max_position)

        if not 0 < risk_per_trade <= 0.10:
            raise ValueError(f"risk_per_trade must be in (0, 0.10], got {risk_per_trade}")
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")

        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier

        logger.info(
            f"Initialized VolatilityPositionSizer: risk_per_trade={risk_per_trade:.3f}, "
            f"atr_multiplier={atr_multiplier:.1f}"
        )

    def calculate(
        self, price: float, atr: float, portfolio_value: float, **kwargs
    ) -> float:
        """
        Calculate volatility-based position size.

        Args:
            price: Current stock price
            atr: Average True Range (volatility measure)
            portfolio_value: Total portfolio value
            **kwargs: Additional arguments (ignored)

        Returns:
            Position size as fraction of portfolio

        Raises:
            ValueError: If price, atr, or portfolio_value are invalid
        """
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        if atr < 0:
            raise ValueError(f"atr must be >= 0, got {atr}")
        if portfolio_value <= 0:
            raise ValueError(f"portfolio_value must be > 0, got {portfolio_value}")

        # Special case: zero volatility → use max position
        if atr == 0:
            logger.warning("ATR is zero, using max_position")
            return self.max_position

        # Dollar amount willing to risk
        risk_dollars = portfolio_value * self.risk_per_trade

        # Stop loss distance in currency units
        stop_distance = atr * self.atr_multiplier

        # Position size: risk_dollars / stop_distance = number of shares
        # Convert to fraction of portfolio
        shares = risk_dollars / stop_distance
        position_value = shares * price
        position_fraction = position_value / portfolio_value

        # Clamp to limits
        clamped_position = np.clip(position_fraction, self.min_position, self.max_position)

        logger.debug(
            f"Volatility sizing: price={price:.2f}, atr={atr:.2f}, "
            f"risk=${risk_dollars:.2f}, stop=${stop_distance:.2f}, "
            f"raw_position={position_fraction:.3f}, clamped={clamped_position:.3f}"
        )

        return float(clamped_position)


class RiskParityPositionSizer(PositionSizer):
    """
    Risk Parity position sizing using volatility targeting.

    Scales position to achieve target portfolio volatility:
    position = target_volatility / asset_volatility

    For single-stock environment (Phase 1-3). Full risk parity (equal risk
    contribution across multiple assets) is deferred to Phase 4.

    Example:
        >>> sizer = RiskParityPositionSizer(target_volatility=0.12)
        >>> position = sizer.calculate(volatility=0.20)  # 20% asset volatility
        >>> # Returns 0.12 / 0.20 = 0.60, clamped to max_position=0.20
    """

    def __init__(
        self,
        target_volatility: float = 0.12,
        min_position: float = 0.10,
        max_position: float = 0.20,
    ) -> None:
        """
        Initialize risk parity position sizer.

        Args:
            target_volatility: Target portfolio volatility (annualized, default: 0.12 = 12%)
            min_position: Minimum position size
            max_position: Maximum position size

        Raises:
            ValueError: If target_volatility is invalid
        """
        super().__init__(min_position, max_position)

        if not 0 < target_volatility <= 1.0:
            raise ValueError(f"target_volatility must be in (0, 1.0], got {target_volatility}")

        self.target_volatility = target_volatility

        logger.info(
            f"Initialized RiskParityPositionSizer: target_volatility={target_volatility:.3f}"
        )

    def calculate(self, volatility: float, **kwargs) -> float:
        """
        Calculate risk parity position size.

        Args:
            volatility: Annualized asset volatility (standard deviation of returns)
            **kwargs: Additional arguments (ignored)

        Returns:
            Position size as fraction of portfolio

        Raises:
            ValueError: If volatility is invalid
        """
        if volatility < 0:
            raise ValueError(f"volatility must be >= 0, got {volatility}")

        # Special case: zero volatility → use max position
        if volatility == 0:
            logger.warning("Volatility is zero, using max_position")
            return self.max_position

        # Position size to achieve target portfolio volatility
        # portfolio_vol = position_size × asset_vol
        # Solve for position_size
        position_fraction = self.target_volatility / volatility

        # Clamp to limits
        clamped_position = np.clip(position_fraction, self.min_position, self.max_position)

        logger.debug(
            f"Risk parity sizing: asset_vol={volatility:.3f}, "
            f"target_vol={self.target_volatility:.3f}, "
            f"raw_position={position_fraction:.3f}, clamped={clamped_position:.3f}"
        )

        return float(clamped_position)


class KellyPositionSizer(PositionSizer):
    """
    Kelly Criterion position sizing for optimal growth.

    Uses fractional Kelly to balance growth and risk:
    position = kelly_fraction × ((p × b - q) / b)

    where:
    - p = win rate (probability of winning)
    - b = payoff ratio (avg_win / avg_loss)
    - q = 1 - p (probability of losing)

    Requires a KellyEstimator to track win rate and payoff ratio.

    Example:
        >>> estimator = KellyEstimator(alpha=0.05)
        >>> sizer = KellyPositionSizer(kelly_fraction=0.25, estimator=estimator)
        >>> # After some trades, calculate position
        >>> position = sizer.calculate()  # Returns Kelly-optimal position size
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        estimator: Optional[KellyEstimator] = None,
        min_position: float = 0.10,
        max_position: float = 0.20,
    ) -> None:
        """
        Initialize Kelly Criterion position sizer.

        Args:
            kelly_fraction: Fractional Kelly multiplier (default: 0.25 = quarter-Kelly)
            estimator: KellyEstimator instance (if None, must be set later)
            min_position: Minimum position size
            max_position: Maximum position size

        Raises:
            ValueError: If kelly_fraction is invalid
        """
        super().__init__(min_position, max_position)

        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1.0], got {kelly_fraction}")

        self.kelly_fraction = kelly_fraction
        self.estimator = estimator

        logger.info(
            f"Initialized KellyPositionSizer: kelly_fraction={kelly_fraction:.2f}"
        )

    def calculate(self, estimator: Optional[KellyEstimator] = None, **kwargs) -> float:
        """
        Calculate Kelly Criterion position size.

        Args:
            estimator: KellyEstimator instance (overrides init estimator if provided)
            **kwargs: Additional arguments (ignored)

        Returns:
            Position size as fraction of portfolio

        Raises:
            ValueError: If estimator is not available
        """
        # Use provided estimator or instance estimator
        est = estimator if estimator is not None else self.estimator

        if est is None:
            raise ValueError("KellyPositionSizer requires a KellyEstimator")

        # Get raw Kelly fraction
        raw_kelly = est.get_kelly_fraction(self.kelly_fraction)

        # Edge case: negative Kelly (no edge) → no position
        if raw_kelly <= 0:
            logger.debug("Negative Kelly fraction, returning 0 position")
            return 0.0

        # Clamp to limits
        clamped_position = np.clip(raw_kelly, self.min_position, self.max_position)

        logger.debug(
            f"Kelly sizing: raw_kelly={raw_kelly:.3f}, clamped={clamped_position:.3f}"
        )

        return float(clamped_position)


def create_position_sizer(config: PositionSizerConfig) -> PositionSizer:
    """
    Factory function to create position sizer from config.

    Args:
        config: PositionSizerConfig instance

    Returns:
        PositionSizer instance (FixedPositionSizer, VolatilityPositionSizer,
        RiskParityPositionSizer, or KellyPositionSizer)

    Raises:
        ValueError: If strategy is not recognized

    Example:
        >>> config = PositionSizerConfig(strategy='fixed', fixed_fraction=0.20)
        >>> sizer = create_position_sizer(config)
        >>> isinstance(sizer, FixedPositionSizer)
        True
    """
    strategy_map = {
        "fixed": FixedPositionSizer,
        "volatility": VolatilityPositionSizer,
        "risk_parity": RiskParityPositionSizer,
        "kelly": KellyPositionSizer,
    }

    if config.strategy not in strategy_map:
        raise ValueError(
            f"Unknown strategy: {config.strategy}. "
            f"Must be one of {list(strategy_map.keys())}"
        )

    # Create appropriate sizer based on strategy
    if config.strategy == "fixed":
        sizer = FixedPositionSizer(
            fixed_fraction=config.fixed_fraction,
            min_position=config.min_position,
            max_position=config.max_position,
        )
    elif config.strategy == "volatility":
        sizer = VolatilityPositionSizer(
            risk_per_trade=config.vol_risk_per_trade,
            atr_multiplier=config.vol_atr_multiplier,
            min_position=config.min_position,
            max_position=config.max_position,
        )
    elif config.strategy == "risk_parity":
        sizer = RiskParityPositionSizer(
            target_volatility=config.rp_target_volatility,
            min_position=config.min_position,
            max_position=config.max_position,
        )
    elif config.strategy == "kelly":
        # Create Kelly estimator
        estimator = KellyEstimator(
            alpha=config.kelly_alpha,
            min_trades=config.kelly_min_trades,
            prior_win_rate=config.kelly_prior_win_rate,
            prior_payoff=config.kelly_prior_payoff,
        )
        sizer = KellyPositionSizer(
            kelly_fraction=config.kelly_fraction,
            estimator=estimator,
            min_position=config.min_position,
            max_position=config.max_position,
        )

    logger.info(f"Created {sizer.__class__.__name__} with strategy='{config.strategy}'")
    return sizer
