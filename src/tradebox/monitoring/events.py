"""Event dataclasses for metrics collection."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PortfolioMetricsEvent:
    """
    Portfolio metrics snapshot event.

    Attributes:
        timestamp: Event timestamp
        total_value: Total portfolio value (cash + positions)
        cash: Available cash
        positions_value: Total value of positions
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss from closed trades
        daily_return_pct: Daily return percentage (optional)
        sharpe_ratio: Sharpe ratio (optional)
        max_drawdown_pct: Maximum drawdown percentage (optional)

    Example:
        >>> event = PortfolioMetricsEvent(
        ...     timestamp=datetime.now(),
        ...     total_value=105000.0,
        ...     cash=50000.0,
        ...     positions_value=55000.0,
        ...     unrealized_pnl=5000.0,
        ...     realized_pnl=1000.0
        ... )
    """

    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_return_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None


@dataclass
class TradeMetricsEvent:
    """
    Trade execution metrics event.

    Attributes:
        trade_id: Unique trade identifier
        timestamp: Trade execution timestamp
        symbol: Stock symbol
        side: Trade side (buy/sell)
        quantity: Number of shares
        intended_price: Intended execution price
        filled_price: Actual fill price
        slippage_pct: Slippage percentage
        commission: Commission paid
        latency_ms: Execution latency in milliseconds
        order_status: Order status (filled/rejected/cancelled)

    Example:
        >>> event = TradeMetricsEvent(
        ...     trade_id="ORD12345",
        ...     timestamp=datetime.now(),
        ...     symbol="RELIANCE.NS",
        ...     side="buy",
        ...     quantity=10,
        ...     intended_price=2500.0,
        ...     filled_price=2502.0,
        ...     slippage_pct=0.08,
        ...     commission=15.0,
        ...     latency_ms=250.0,
        ...     order_status="filled"
        ... )
    """

    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # buy/sell
    quantity: int
    intended_price: float
    filled_price: float
    slippage_pct: float
    commission: float
    latency_ms: float
    order_status: str  # filled/rejected/cancelled


@dataclass
class ModelMetricsEvent:
    """
    Model prediction metrics event.

    Attributes:
        timestamp: Prediction timestamp
        symbol: Stock symbol
        action: Predicted action (0=hold, 1=buy, 2=sell)
        confidence: Prediction confidence score (optional)
        observation: Feature vector as JSON string (optional)
        reward: Realized reward (optional, recorded later)

    Example:
        >>> event = ModelMetricsEvent(
        ...     timestamp=datetime.now(),
        ...     symbol="TCS.NS",
        ...     action=1,
        ...     confidence=0.85,
        ...     observation='{"rsi": 45.2, "macd": 0.5}',
        ...     reward=None
        ... )
    """

    timestamp: datetime
    symbol: str
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: Optional[float] = None
    observation: Optional[str] = None  # JSON serialized
    reward: Optional[float] = None


@dataclass
class SystemMetricsEvent:
    """
    System health metrics event.

    Attributes:
        timestamp: Event timestamp
        metric_name: Name of the metric (e.g., "api_latency", "error_count")
        metric_value: Metric value
        metric_type: Type of metric (latency/error/count)
        component: Component name (data/agent/broker/risk)
        message: Optional message for context

    Example:
        >>> event = SystemMetricsEvent(
        ...     timestamp=datetime.now(),
        ...     metric_name="api_latency",
        ...     metric_value=120.0,
        ...     metric_type="latency",
        ...     component="broker",
        ...     message="Order placement latency"
        ... )
    """

    timestamp: datetime
    metric_name: str
    metric_value: float
    metric_type: str  # latency/error/count
    component: str  # data/agent/broker/risk
    message: Optional[str] = None
