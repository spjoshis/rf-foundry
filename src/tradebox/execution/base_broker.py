"""Abstract base class for broker implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """
    Represents a trading order.

    Attributes:
        order_id: Unique order identifier
        symbol: Stock symbol
        side: Buy or sell
        quantity: Number of shares
        price: Order price (None for market orders)
        status: Order status
        timestamp: Order creation time
        filled_price: Actual fill price (None if not filled)
        filled_quantity: Actual filled quantity
        commission: Commission paid
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Optional[float] = None  # None for market orders
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    commission: float = 0.0

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """
    Represents a portfolio position.

    Attributes:
        symbol: Stock symbol
        quantity: Number of shares held
        avg_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss from closed trades
    """

    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return self.quantity * self.avg_price


@dataclass
class Portfolio:
    """
    Represents entire portfolio.

    Attributes:
        cash: Available cash
        positions: Dict of symbol -> Position
        total_value: Total portfolio value (cash + positions)
    """

    cash: float
    positions: Dict[str, Position]
    total_value: float = 0.0

    def update_total_value(self) -> None:
        """Update total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        self.total_value = self.cash + positions_value


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.

    Defines the interface that all brokers (paper, live) must implement.
    This enables easy swapping between paper trading and live trading.

    Example:
        >>> # Using paper broker
        >>> from tradebox.execution import PaperBroker
        >>> broker = PaperBroker(initial_capital=100000)
        >>>
        >>> # Place order
        >>> order = broker.place_order("RELIANCE.NS", OrderSide.BUY, 10)
        >>> print(f"Order status: {order.status}")
        >>>
        >>> # Check portfolio
        >>> portfolio = broker.get_portfolio()
        >>> print(f"Cash: ₹{portfolio.cash:,.0f}")
    """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Optional[float] = None,
    ) -> Order:
        """
        Place a trading order.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            side: Buy or sell
            quantity: Number of shares
            price: Limit price (None for market order)

        Returns:
            Order object with order details

        Raises:
            ValueError: If order is invalid
            RuntimeError: If order execution fails
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None if no position
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    def get_portfolio(self) -> Portfolio:
        """
        Get complete portfolio information.

        Returns:
            Portfolio object with cash, positions, and total value
        """
        pass

    @abstractmethod
    def get_cash(self) -> float:
        """
        Get available cash.

        Returns:
            Available cash amount
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order identifier

        Returns:
            Order object or None if not found
        """
        pass

    @abstractmethod
    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """
        Get orders with optional filters.

        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)

        Returns:
            List of Order objects
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order identifier

        Returns:
            True if cancelled successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get current market quote.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with price information (bid, ask, last, etc.)
        """
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close entire position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Order object for the closing trade, or None if no position
        """
        pass
