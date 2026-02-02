"""Paper broker implementation for simulated trading."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import yfinance as yf
from loguru import logger

from tradebox.env.costs import CostConfig, TransactionCostModel
from tradebox.execution.base_broker import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
    Position,
)


class PaperBroker(BaseBroker):
    """
    Paper trading broker with simulated execution.

    Uses live prices from Yahoo Finance but simulates order execution
    without real money. Tracks virtual portfolio with realistic
    transaction costs.

    Attributes:
        initial_capital: Starting cash
        cash: Current available cash
        positions: Dict of symbol -> Position
        orders: Dict of order_id -> Order
        cost_model: Transaction cost calculator

    Example:
        >>> broker = PaperBroker(initial_capital=100000)
        >>>
        >>> # Place buy order
        >>> order = broker.place_order("RELIANCE.NS", OrderSide.BUY, 10)
        >>> print(f"Order status: {order.status}")
        >>>
        >>> # Check portfolio
        >>> portfolio = broker.get_portfolio()
        >>> print(f"Portfolio value: ₹{portfolio.total_value:,.0f}")
        >>>
        >>> # Close position
        >>> close_order = broker.close_position("RELIANCE.NS")
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_config: Optional[CostConfig] = None,
    ) -> None:
        """
        Initialize paper broker.

        Args:
            initial_capital: Starting capital in rupees
            cost_config: Transaction cost configuration
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}

        # Transaction cost model
        self.cost_model = TransactionCostModel(cost_config or CostConfig())

        logger.info(
            f"PaperBroker initialized with capital: ₹{initial_capital:,.0f}"
        )

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Optional[float] = None,
    ) -> Order:
        """
        Place a simulated order.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            side: Buy or sell
            quantity: Number of shares
            price: Limit price (None for market order)

        Returns:
            Order object

        Raises:
            ValueError: If order is invalid
        """
        # Validate
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        # Get current market price
        current_price = self._get_market_price(symbol)

        if current_price is None:
            raise ValueError(f"Unable to get price for {symbol}")

        # Use market price if no limit price specified
        fill_price = price if price is not None else current_price

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
        )

        # Simulate immediate execution for market orders
        if price is None:
            self._execute_order(order, fill_price)

        # Store order
        self.orders[order.order_id] = order

        logger.info(
            f"Order placed: {side.value} {quantity} {symbol} @ ₹{fill_price:.2f} "
            f"(status: {order.status.value})"
        )

        return order

    def _execute_order(self, order: Order, fill_price: float) -> None:
        """
        Execute order and update portfolio.

        Args:
            order: Order to execute
            fill_price: Execution price
        """
        if order.side == OrderSide.BUY:
            self._execute_buy(order, fill_price)
        else:
            self._execute_sell(order, fill_price)

    def _execute_buy(self, order: Order, fill_price: float) -> None:
        """Execute buy order."""
        # Calculate total cost including fees
        total_cost, breakdown = self.cost_model.calculate_buy_cost(
            order.quantity, fill_price
        )

        # Check if we have enough cash
        if total_cost > self.cash:
            order.status = OrderStatus.REJECTED
            logger.warning(
                f"Buy order rejected: insufficient cash "
                f"(need ₹{total_cost:.2f}, have ₹{self.cash:.2f})"
            )
            return

        # Deduct cash
        self.cash -= total_cost

        # Update position
        if order.symbol in self.positions:
            # Add to existing position
            pos = self.positions[order.symbol]
            total_quantity = pos.quantity + order.quantity
            total_cost_basis = (pos.quantity * pos.avg_price) + (
                order.quantity * fill_price
            )
            pos.avg_price = total_cost_basis / total_quantity
            pos.quantity = total_quantity
        else:
            # Create new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                avg_price=fill_price,
                current_price=fill_price,
            )

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = breakdown["total_cost"] - (order.quantity * fill_price)

        logger.debug(
            f"Buy executed: {order.quantity} {order.symbol} @ ₹{fill_price:.2f}, "
            f"cost: ₹{total_cost:.2f}"
        )

    def _execute_sell(self, order: Order, fill_price: float) -> None:
        """Execute sell order."""
        # Check if we have the position
        if order.symbol not in self.positions:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Sell order rejected: no position in {order.symbol}")
            return

        pos = self.positions[order.symbol]

        # Check if we have enough shares
        if order.quantity > pos.quantity:
            order.status = OrderStatus.REJECTED
            logger.warning(
                f"Sell order rejected: insufficient shares "
                f"(need {order.quantity}, have {pos.quantity})"
            )
            return

        # Calculate proceeds after fees
        net_proceeds, breakdown = self.cost_model.calculate_sell_proceeds(
            order.quantity, fill_price
        )

        # Add cash
        self.cash += net_proceeds

        # Calculate realized P&L
        realized_pnl = (fill_price - pos.avg_price) * order.quantity - (
            breakdown["total_cost"] - net_proceeds
        )
        pos.realized_pnl += realized_pnl

        # Update position
        if order.quantity == pos.quantity:
            # Close position completely
            del self.positions[order.symbol]
            logger.debug(f"Position closed: {order.symbol}, P&L: ₹{realized_pnl:.2f}")
        else:
            # Reduce position
            pos.quantity -= order.quantity

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.commission = (order.quantity * fill_price) - net_proceeds

        logger.debug(
            f"Sell executed: {order.quantity} {order.symbol} @ ₹{fill_price:.2f}, "
            f"proceeds: ₹{net_proceeds:.2f}, P&L: ₹{realized_pnl:.2f}"
        )

    def _get_market_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price from Yahoo Finance.

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")

            if data.empty:
                logger.warning(f"No price data available for {symbol}")
                return None

            price = float(data["Close"].iloc[-1])
            return price

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def _update_position_prices(self) -> None:
        """Update current prices for all positions."""
        for symbol, pos in self.positions.items():
            current_price = self._get_market_price(symbol)
            if current_price is not None:
                pos.current_price = current_price
                pos.unrealized_pnl = (
                    current_price - pos.avg_price
                ) * pos.quantity

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_positions(self) -> List[Position]:
        """Get all positions."""
        self._update_position_prices()
        return list(self.positions.values())

    def get_portfolio(self) -> Portfolio:
        """Get complete portfolio."""
        self._update_position_prices()

        portfolio = Portfolio(
            cash=self.cash,
            positions=self.positions.copy(),
        )
        portfolio.update_total_value()

        return portfolio

    def get_cash(self) -> float:
        """Get available cash."""
        return self.cash

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """Get orders with optional filters."""
        orders = list(self.orders.values())

        if symbol is not None:
            orders = [o for o in orders if o.symbol == symbol]

        if status is not None:
            orders = [o for o in orders if o.status == status]

        return orders

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        order = self.orders.get(order_id)

        if order is None:
            logger.warning(f"Order not found: {order_id}")
            return False

        if order.status != OrderStatus.PENDING:
            logger.warning(
                f"Cannot cancel order {order_id}: status is {order.status.value}"
            )
            return False

        order.status = OrderStatus.CANCELLED
        logger.info(f"Order cancelled: {order_id}")
        return True

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current market quote."""
        price = self._get_market_price(symbol)

        if price is None:
            return {}

        return {
            "symbol": symbol,
            "last": price,
            "bid": price * 0.999,  # Simulated bid
            "ask": price * 1.001,  # Simulated ask
            "timestamp": datetime.now().isoformat(),
        }

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position."""
        pos = self.positions.get(symbol)

        if pos is None:
            logger.warning(f"No position to close: {symbol}")
            return None

        # Place market sell order for full quantity
        return self.place_order(symbol, OrderSide.SELL, pos.quantity)

    def get_stats(self) -> Dict[str, any]:
        """Get broker statistics."""
        portfolio = self.get_portfolio()

        total_return = (portfolio.total_value - self.initial_capital) / self.initial_capital

        filled_orders = [o for o in self.orders.values() if o.status == OrderStatus.FILLED]
        total_commission = sum(o.commission for o in filled_orders)

        return {
            "initial_capital": self.initial_capital,
            "current_value": portfolio.total_value,
            "cash": self.cash,
            "total_return": total_return,
            "num_positions": len(self.positions),
            "num_orders": len(self.orders),
            "total_commission": total_commission,
        }
