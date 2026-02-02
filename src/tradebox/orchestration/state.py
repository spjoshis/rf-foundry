"""State management for trading orchestrator."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from tradebox.execution.base_broker import Position
from tradebox.orchestration.exceptions import StateError


@dataclass
class OrchestratorState:
    """
    Persisted state for trading orchestrator.

    Attributes:
        last_run_time: Timestamp of last successful run
        positions: Current positions (symbol -> position dict)
        daily_pnl: Today's realized P&L
        circuit_breaker_active: Whether circuit breaker is triggered
        daily_start_value: Portfolio value at start of day
        total_trades_today: Number of trades executed today
        last_error: Last error message (if any)
        error_count_today: Number of errors today

    Example:
        >>> state = OrchestratorState()
        >>> state.last_run_time = datetime.now()
        >>> state.daily_pnl = 1250.50
    """

    last_run_time: Optional[str] = None  # ISO format datetime string
    positions: Dict[str, dict] = None  # symbol -> position dict
    daily_pnl: float = 0.0
    circuit_breaker_active: bool = False
    daily_start_value: float = 0.0
    total_trades_today: int = 0
    last_error: Optional[str] = None
    error_count_today: int = 0

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.positions is None:
            self.positions = {}

    @property
    def last_run_datetime(self) -> Optional[datetime]:
        """
        Get last run time as datetime object.

        Returns:
            datetime object or None
        """
        if self.last_run_time is None:
            return None
        return datetime.fromisoformat(self.last_run_time)

    def update_position(self, symbol: str, position: Position) -> None:
        """
        Update a position in state.

        Args:
            symbol: Stock symbol
            position: Position object

        Example:
            >>> state = OrchestratorState()
            >>> pos = Position("RELIANCE.NS", 10, 2500.0, 2550.0)
            >>> state.update_position("RELIANCE.NS", pos)
        """
        self.positions[symbol] = {
            "symbol": position.symbol,
            "quantity": position.quantity,
            "avg_price": position.avg_price,
            "current_price": position.current_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
        }
        logger.debug(f"Updated position for {symbol}: {position.quantity} @ ₹{position.avg_price:.2f}")

    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from state.

        Args:
            symbol: Stock symbol
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.debug(f"Removed position for {symbol}")

    def reset_daily(self, portfolio_value: float) -> None:
        """
        Reset daily tracking at start of new day.

        Args:
            portfolio_value: Starting portfolio value for the day

        Example:
            >>> state = OrchestratorState()
            >>> state.reset_daily(100000.0)
        """
        self.daily_pnl = 0.0
        self.circuit_breaker_active = False
        self.daily_start_value = portfolio_value
        self.total_trades_today = 0
        self.error_count_today = 0
        self.last_error = None
        logger.info(f"Daily state reset: start_value=₹{portfolio_value:,.0f}")

    def record_error(self, error_message: str) -> None:
        """
        Record an error in state.

        Args:
            error_message: Error message
        """
        self.last_error = error_message
        self.error_count_today += 1
        logger.warning(f"Error recorded: {error_message} (count today: {self.error_count_today})")

    def trigger_circuit_breaker(self, reason: str) -> None:
        """
        Trigger circuit breaker.

        Args:
            reason: Reason for circuit breaker trigger
        """
        self.circuit_breaker_active = True
        logger.error(f"Circuit breaker triggered: {reason}")

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual override)."""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker reset manually")


class StateManager:
    """
    Manages orchestrator state persistence.

    Handles loading, saving, and updating orchestrator state
    to/from JSON file on disk.

    Example:
        >>> manager = StateManager("data/orchestrator_state.json")
        >>> state = manager.load()
        >>> state.daily_pnl = 1500.0
        >>> manager.save(state)
    """

    def __init__(self, state_file: str = "data/orchestrator_state.json"):
        """
        Initialize state manager.

        Args:
            state_file: Path to state persistence file
        """
        self.state_file = Path(state_file)
        self.state: OrchestratorState = self.load()
        logger.info(f"StateManager initialized with file: {state_file}")

    def load(self) -> OrchestratorState:
        """
        Load state from disk.

        Returns:
            OrchestratorState instance

        Example:
            >>> manager = StateManager()
            >>> state = manager.load()
        """
        if not self.state_file.exists():
            logger.info(f"State file not found, creating new state: {self.state_file}")
            state = OrchestratorState()
            self.save(state)  # Create initial file
            return state

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            state = OrchestratorState(**data)
            logger.info(f"Loaded state from {self.state_file}")

            # Log key state info
            if state.last_run_time:
                logger.info(f"  Last run: {state.last_run_time}")
            logger.info(f"  Active positions: {len(state.positions)}")
            logger.info(f"  Daily P&L: ₹{state.daily_pnl:,.2f}")
            if state.circuit_breaker_active:
                logger.warning("  ⚠️ Circuit breaker ACTIVE")

            return state

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to load state from {self.state_file}: {e}")
            logger.warning("Creating fresh state")
            return OrchestratorState()

    def save(self, state: Optional[OrchestratorState] = None) -> None:
        """
        Save state to disk.

        Args:
            state: State to save (uses self.state if None)

        Raises:
            StateError: If save fails

        Example:
            >>> manager = StateManager()
            >>> state = OrchestratorState()
            >>> state.daily_pnl = 2000.0
            >>> manager.save(state)
        """
        if state is None:
            state = self.state

        try:
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict
            data = asdict(state)

            # Write atomically (write to temp file, then rename)
            temp_file = self.state_file.with_suffix(".tmp")

            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_file.replace(self.state_file)

            logger.debug(f"Saved state to {self.state_file}")

        except (IOError, OSError) as e:
            raise StateError(f"Failed to save state: {e}")

    def update_last_run(self) -> None:
        """
        Update last run timestamp to now.

        Example:
            >>> manager = StateManager()
            >>> manager.update_last_run()
        """
        self.state.last_run_time = datetime.now().isoformat()
        self.save()
        logger.debug(f"Updated last_run_time: {self.state.last_run_time}")

    def update_position(self, symbol: str, position: Position) -> None:
        """
        Update a position and save state.

        Args:
            symbol: Stock symbol
            position: Position object

        Example:
            >>> manager = StateManager()
            >>> pos = Position("TCS.NS", 5, 3500.0, 3600.0)
            >>> manager.update_position("TCS.NS", pos)
        """
        self.state.update_position(symbol, position)
        self.save()

    def remove_position(self, symbol: str) -> None:
        """
        Remove a position and save state.

        Args:
            symbol: Stock symbol
        """
        self.state.remove_position(symbol)
        self.save()

    def reset_daily(self, portfolio_value: float) -> None:
        """
        Reset daily tracking and save state.

        Args:
            portfolio_value: Starting portfolio value for the day
        """
        self.state.reset_daily(portfolio_value)
        self.save()

    def record_error(self, error_message: str) -> None:
        """
        Record an error and save state.

        Args:
            error_message: Error message
        """
        self.state.record_error(error_message)
        self.save()

    def trigger_circuit_breaker(self, reason: str) -> None:
        """
        Trigger circuit breaker and save state.

        Args:
            reason: Reason for circuit breaker trigger
        """
        self.state.trigger_circuit_breaker(reason)
        self.save()

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker and save state."""
        self.state.reset_circuit_breaker()
        self.save()

    def increment_trade_count(self) -> None:
        """Increment today's trade count."""
        self.state.total_trades_today += 1
        self.save()

    def get_state(self) -> OrchestratorState:
        """
        Get current state.

        Returns:
            Current OrchestratorState
        """
        return self.state
