"""Circuit breakers for automatic trading halt."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger

from tradebox.execution.base_broker import Portfolio


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breakers.

    Attributes:
        max_drawdown_pct: Max portfolio drawdown before halt (default: 10%)
        max_daily_loss_pct: Max daily loss before halt (default: 5%)
        consecutive_loss_days: Consecutive losing days before halt (default: 3)
        enabled: Whether circuit breakers are active (default: True)
    """

    max_drawdown_pct: float = 0.10
    max_daily_loss_pct: float = 0.05
    consecutive_loss_days: int = 3
    enabled: bool = True


class CircuitBreaker(ABC):
    """
    Abstract base class for circuit breakers.

    Circuit breakers automatically halt trading when adverse
    conditions are detected.
    """

    def __init__(self) -> None:
        """Initialize circuit breaker."""
        self.is_triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: str = ""

    @abstractmethod
    def check(
        self,
        portfolio: Portfolio,
        config: CircuitBreakerConfig,
    ) -> bool:
        """
        Check if circuit breaker should trigger.

        Args:
            portfolio: Current portfolio state
            config: Circuit breaker configuration

        Returns:
            True if should halt trading, False otherwise
        """
        pass

    def trigger(self, reason: str) -> None:
        """
        Trigger the circuit breaker.

        Args:
            reason: Reason for triggering
        """
        self.is_triggered = True
        self.trigger_time = datetime.now()
        self.trigger_reason = reason
        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")

    def reset(self) -> None:
        """Reset the circuit breaker."""
        if self.is_triggered:
            logger.warning(f"Circuit breaker reset: {self.trigger_reason}")
        self.is_triggered = False
        self.trigger_time = None
        self.trigger_reason = ""


class DrawdownCircuitBreaker(CircuitBreaker):
    """
    Halt trading if portfolio drawdown exceeds threshold.

    Tracks peak portfolio value and triggers if current value
    falls too far below peak.

    Example:
        >>> breaker = DrawdownCircuitBreaker()
        >>> breaker.check(portfolio, config)
        >>> if breaker.is_triggered:
        ...     print(f"Trading halted: {breaker.trigger_reason}")
    """

    def __init__(self) -> None:
        """Initialize drawdown circuit breaker."""
        super().__init__()
        self.peak_value: float = 0.0

    def check(
        self,
        portfolio: Portfolio,
        config: CircuitBreakerConfig,
    ) -> bool:
        """Check for excessive drawdown."""
        if not config.enabled:
            return False

        # Update peak
        self.peak_value = max(self.peak_value, portfolio.total_value)

        if self.peak_value == 0:
            return False

        # Calculate drawdown
        drawdown = (self.peak_value - portfolio.total_value) / self.peak_value

        # Check threshold
        if drawdown >= config.max_drawdown_pct:
            self.trigger(
                f"Drawdown of {drawdown:.1%} exceeded limit of "
                f"{config.max_drawdown_pct:.1%} "
                f"(peak: ₹{self.peak_value:,.0f}, current: ₹{portfolio.total_value:,.0f})"
            )
            return True

        return False

    def reset(self) -> None:
        """Reset drawdown tracking."""
        super().reset()
        self.peak_value = 0.0


class DailyLossCircuitBreaker(CircuitBreaker):
    """
    Halt trading if daily loss exceeds threshold.

    Tracks portfolio value at start of day and triggers if
    loss exceeds maximum allowed.
    """

    def __init__(self) -> None:
        """Initialize daily loss circuit breaker."""
        super().__init__()
        self.daily_start_value: Optional[float] = None
        self.last_check_date: Optional[datetime] = None

    def reset_daily(self, portfolio_value: float) -> None:
        """
        Reset daily tracking at start of new trading day.

        Args:
            portfolio_value: Starting value for the day
        """
        self.daily_start_value = portfolio_value
        self.last_check_date = datetime.now()
        self.is_triggered = False
        logger.info(
            f"Daily loss circuit breaker reset: start value ₹{portfolio_value:,.0f}"
        )

    def check(
        self,
        portfolio: Portfolio,
        config: CircuitBreakerConfig,
    ) -> bool:
        """Check for excessive daily loss."""
        if not config.enabled:
            return False

        # Initialize if first check
        if self.daily_start_value is None:
            self.reset_daily(portfolio.total_value)
            return False

        # Calculate daily loss
        daily_loss = (portfolio.total_value - self.daily_start_value) / self.daily_start_value

        # Check threshold
        if daily_loss <= -config.max_daily_loss_pct:
            self.trigger(
                f"Daily loss of {daily_loss:.1%} exceeded limit of "
                f"-{config.max_daily_loss_pct:.1%} "
                f"(start: ₹{self.daily_start_value:,.0f}, "
                f"current: ₹{portfolio.total_value:,.0f})"
            )
            return True

        return False


class ConsecutiveLossCircuitBreaker(CircuitBreaker):
    """
    Halt trading after consecutive losing days.

    Tracks daily P&L and triggers if too many consecutive
    losing days occur.
    """

    def __init__(self) -> None:
        """Initialize consecutive loss circuit breaker."""
        super().__init__()
        self.consecutive_losses = 0
        self.daily_pnl_history: List[float] = []
        self.last_portfolio_value: Optional[float] = None
        self.last_check_date: Optional[datetime] = None

    def update_daily(self, portfolio_value: float) -> None:
        """
        Update at end of trading day.

        Args:
            portfolio_value: Ending value for the day
        """
        if self.last_portfolio_value is None:
            self.last_portfolio_value = portfolio_value
            return

        # Calculate daily P&L
        daily_pnl = portfolio_value - self.last_portfolio_value
        self.daily_pnl_history.append(daily_pnl)

        # Update consecutive losses
        if daily_pnl < 0:
            self.consecutive_losses += 1
            logger.warning(
                f"Losing day {self.consecutive_losses}: ₹{daily_pnl:,.0f}"
            )
        else:
            self.consecutive_losses = 0

        self.last_portfolio_value = portfolio_value
        self.last_check_date = datetime.now()

    def check(
        self,
        portfolio: Portfolio,
        config: CircuitBreakerConfig,
    ) -> bool:
        """Check for consecutive losing days."""
        if not config.enabled:
            return False

        if self.consecutive_losses >= config.consecutive_loss_days:
            self.trigger(
                f"{self.consecutive_losses} consecutive losing days "
                f"(limit: {config.consecutive_loss_days})"
            )
            return True

        return False

    def reset(self) -> None:
        """Reset consecutive loss tracking."""
        super().reset()
        self.consecutive_losses = 0


class APIFailureCircuitBreaker(CircuitBreaker):
    """
    Halt trading on API or network failures.

    Tracks API errors and triggers if too many failures occur
    in a short time period.
    """

    def __init__(self, max_failures: int = 3, time_window_minutes: int = 5) -> None:
        """
        Initialize API failure circuit breaker.

        Args:
            max_failures: Maximum failures allowed in time window
            time_window_minutes: Time window in minutes
        """
        super().__init__()
        self.max_failures = max_failures
        self.time_window = timedelta(minutes=time_window_minutes)
        self.failure_times: List[datetime] = []

    def record_failure(self, error: Exception) -> None:
        """
        Record an API failure.

        Args:
            error: Exception that occurred
        """
        self.failure_times.append(datetime.now())
        logger.error(f"API failure recorded: {error}")

        # Check if should trigger
        self._check_failures()

    def _check_failures(self) -> bool:
        """Check if too many recent failures."""
        now = datetime.now()

        # Remove old failures outside time window
        self.failure_times = [
            t for t in self.failure_times
            if now - t <= self.time_window
        ]

        # Check threshold
        if len(self.failure_times) >= self.max_failures:
            self.trigger(
                f"{len(self.failure_times)} API failures in "
                f"{self.time_window.total_seconds() / 60:.0f} minutes "
                f"(limit: {self.max_failures})"
            )
            return True

        return False

    def check(
        self,
        portfolio: Portfolio,
        config: CircuitBreakerConfig,
    ) -> bool:
        """Check for API failures."""
        if not config.enabled:
            return False

        return self._check_failures()

    def reset(self) -> None:
        """Reset failure tracking."""
        super().reset()
        self.failure_times = []


class CircuitBreakerManager:
    """
    Manages all circuit breakers and checks for halt conditions.

    Aggregates multiple circuit breakers and provides unified
    interface for checking and managing trading halts.

    Example:
        >>> config = CircuitBreakerConfig(max_drawdown_pct=0.10)
        >>> manager = CircuitBreakerManager(config)
        >>>
        >>> # Check before trading
        >>> should_halt, reason = manager.check_all(portfolio)
        >>> if should_halt:
        ...     print(f"Trading halted: {reason}")
        ...     # Send alert
        ...     # Stop trading
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        """
        Initialize circuit breaker manager.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()

        # Initialize all circuit breakers
        self.breakers: List[CircuitBreaker] = [
            DrawdownCircuitBreaker(),
            DailyLossCircuitBreaker(),
            ConsecutiveLossCircuitBreaker(),
            APIFailureCircuitBreaker(),
        ]

        logger.info(
            f"CircuitBreakerManager initialized with {len(self.breakers)} breakers"
        )

    def check_all(self, portfolio: Portfolio) -> tuple[bool, str]:
        """
        Check all circuit breakers.

        Args:
            portfolio: Current portfolio state

        Returns:
            Tuple of (should_halt, reason)
        """
        if not self.config.enabled:
            return False, ""

        for breaker in self.breakers:
            if breaker.check(portfolio, self.config):
                return True, breaker.trigger_reason

        return False, ""

    def is_trading_halted(self) -> bool:
        """Check if any circuit breaker is triggered."""
        return any(b.is_triggered for b in self.breakers)

    def get_triggered_breakers(self) -> List[CircuitBreaker]:
        """Get list of triggered circuit breakers."""
        return [b for b in self.breakers if b.is_triggered]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers:
            breaker.reset()
        logger.info("All circuit breakers reset")

    def reset_daily(self, portfolio_value: float) -> None:
        """
        Reset daily tracking at start of new trading day.

        Args:
            portfolio_value: Starting portfolio value
        """
        for breaker in self.breakers:
            if isinstance(breaker, DailyLossCircuitBreaker):
                breaker.reset_daily(portfolio_value)

    def update_daily(self, portfolio_value: float) -> None:
        """
        Update at end of trading day.

        Args:
            portfolio_value: Ending portfolio value
        """
        for breaker in self.breakers:
            if isinstance(breaker, ConsecutiveLossCircuitBreaker):
                breaker.update_daily(portfolio_value)

    def record_api_failure(self, error: Exception) -> None:
        """
        Record an API failure.

        Args:
            error: Exception that occurred
        """
        for breaker in self.breakers:
            if isinstance(breaker, APIFailureCircuitBreaker):
                breaker.record_failure(error)

    def get_status(self) -> dict:
        """Get status of all circuit breakers."""
        status = {
            "trading_halted": self.is_trading_halted(),
            "enabled": self.config.enabled,
            "breakers": []
        }

        for breaker in self.breakers:
            breaker_status = {
                "name": breaker.__class__.__name__,
                "triggered": breaker.is_triggered,
                "reason": breaker.trigger_reason if breaker.is_triggered else None,
                "trigger_time": breaker.trigger_time.isoformat() if breaker.trigger_time else None,
            }
            status["breakers"].append(breaker_status)

        return status
