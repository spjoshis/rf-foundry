"""Unit tests for KiteBroker."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

from kiteconnect.exceptions import NetworkException, TokenException

from tradebox.execution.base_broker import OrderSide, OrderStatus
from tradebox.execution.exceptions import (
    CircuitBreakerTriggered,
    OrderPlacementException,
    RateLimitException,
)
from tradebox.execution.kite_broker import KiteBroker, KiteBrokerConfig
from tradebox.execution.retry import RetryConfig, RetryableException


@pytest.fixture
def mock_kite():
    """Create a mock KiteConnect instance."""
    mock = MagicMock()

    # Setup default return values
    mock.VARIETY_REGULAR = "regular"
    mock.EXCHANGE_NSE = "NSE"
    mock.TRANSACTION_TYPE_BUY = "BUY"
    mock.TRANSACTION_TYPE_SELL = "SELL"
    mock.ORDER_TYPE_MARKET = "MARKET"
    mock.ORDER_TYPE_LIMIT = "LIMIT"
    mock.PRODUCT_CNC = "CNC"
    mock.VALIDITY_DAY = "DAY"

    return mock


@pytest.fixture
def broker_config():
    """Create a test broker configuration."""
    return KiteBrokerConfig(
        api_key="test_api_key",
        api_secret="test_api_secret",
        access_token="test_access_token",
        reconciliation_enabled=False,  # Disable for unit tests
        retry_config=RetryConfig(max_retries=2, initial_delay_seconds=0.1),
    )


@pytest.fixture
def broker(broker_config, mock_kite):
    """Create a KiteBroker instance for testing."""
    with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
        broker = KiteBroker(broker_config)
        broker.kite = mock_kite  # Inject mock
        return broker


class TestKiteBrokerInit:
    """Tests for KiteBroker initialization."""

    def test_init_with_access_token(self, broker_config, mock_kite):
        """Test initialization with access token."""
        with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
            broker = KiteBroker(broker_config)

            assert broker.config == broker_config
            mock_kite.set_access_token.assert_called_once_with("test_access_token")

    def test_init_without_access_token(self, mock_kite):
        """Test initialization without access token."""
        config = KiteBrokerConfig(
            api_key="test_key",
            api_secret="test_secret",
            access_token=None,
            reconciliation_enabled=False,
        )

        with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
            broker = KiteBroker(config)

            # Should not set access token
            mock_kite.set_access_token.assert_not_called()

    def test_init_with_reconciliation_enabled(self, broker_config, mock_kite):
        """Test initialization with reconciliation enabled."""
        broker_config.reconciliation_enabled = True

        with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
            broker = KiteBroker(broker_config)

            assert broker._reconciliation_thread is not None
            assert broker._reconciliation_thread.is_alive()

            # Cleanup
            broker.stop_reconciliation()


class TestPlaceOrder:
    """Tests for order placement."""

    def test_place_order_success(self, broker, mock_kite):
        """Test successful order placement."""
        # Mock Kite API response
        mock_kite.place_order.return_value = "ORDER123"
        mock_kite.orders.return_value = [
            {
                "order_id": "ORDER123",
                "tradingsymbol": "RELIANCE",
                "transaction_type": "BUY",
                "quantity": 10,
                "status": "COMPLETE",
                "average_price": 2500.0,
                "filled_quantity": 10,
            }
        ]

        # Place order
        order = broker.place_order("RELIANCE", OrderSide.BUY, 10)

        # Verify
        assert order.order_id == "ORDER123"
        assert order.symbol == "RELIANCE"
        assert order.side == OrderSide.BUY
        assert order.quantity == 10
        assert order.status == OrderStatus.FILLED

        # Verify API was called correctly
        mock_kite.place_order.assert_called_once()

    def test_place_order_with_price(self, broker, mock_kite):
        """Test limit order placement."""
        mock_kite.place_order.return_value = "ORDER124"
        mock_kite.orders.return_value = []

        order = broker.place_order("TCS", OrderSide.SELL, 5, price=3500.0)

        assert order.price == 3500.0

        # Verify limit order type was used
        call_kwargs = mock_kite.place_order.call_args[1]
        assert call_kwargs["order_type"] == "LIMIT"
        assert call_kwargs["price"] == 3500.0

    def test_place_order_invalid_quantity(self, broker):
        """Test order placement with invalid quantity."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            broker.place_order("INFY", OrderSide.BUY, -10)

    def test_place_order_retry_on_network_error(self, broker, mock_kite):
        """Test retry logic on network error."""
        # First call fails, second succeeds
        mock_kite.place_order.side_effect = [
            NetworkException("Connection timeout"),
            "ORDER125",
        ]
        mock_kite.orders.return_value = []

        order = broker.place_order("HDFC", OrderSide.BUY, 20)

        # Should have retried
        assert mock_kite.place_order.call_count == 2
        assert order.order_id == "ORDER125"

    def test_place_order_failure_after_retries(self, broker, mock_kite):
        """Test order placement fails after max retries."""
        # Always fail
        mock_kite.place_order.side_effect = NetworkException("Network error")

        with pytest.raises(OrderPlacementException, match="Failed to place order"):
            broker.place_order("SBIN", OrderSide.BUY, 100)

        # Should have tried max_retries + 1 times (2 + 1 = 3)
        assert mock_kite.place_order.call_count == 3

    def test_place_order_non_retryable_error(self, broker, mock_kite):
        """Test order placement with non-retryable error."""
        # Token error is not retryable
        mock_kite.place_order.side_effect = TokenException("Invalid token")

        with pytest.raises(OrderPlacementException, match="Failed to place order"):
            broker.place_order("ITC", OrderSide.BUY, 50)

        # Should only try once (no retries)
        assert mock_kite.place_order.call_count == 1

    def test_place_order_circuit_breaker_triggered(self, broker):
        """Test order placement when circuit breaker is triggered."""
        # Trigger circuit breaker
        broker.circuit_breaker._triggered = True
        broker.circuit_breaker._trigger_reason = "Test halt"

        with pytest.raises(CircuitBreakerTriggered, match="Trading halted"):
            broker.place_order("WIPRO", OrderSide.BUY, 10)


class TestGetPosition:
    """Tests for position retrieval."""

    def test_get_position_exists(self, broker, mock_kite):
        """Test getting an existing position."""
        mock_kite.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "RELIANCE",
                    "quantity": 10,
                    "average_price": 2500.0,
                    "last_price": 2550.0,
                    "pnl": 500.0,
                    "realised": 0.0,
                }
            ]
        }

        position = broker.get_position("RELIANCE")

        assert position is not None
        assert position.symbol == "RELIANCE"
        assert position.quantity == 10
        assert position.avg_price == 2500.0
        assert position.current_price == 2550.0

    def test_get_position_not_exists(self, broker, mock_kite):
        """Test getting a non-existent position."""
        mock_kite.positions.return_value = {"day": []}

        position = broker.get_position("NONEXISTENT")

        assert position is None

    def test_get_positions_cache_fresh(self, broker, mock_kite):
        """Test getting positions from fresh cache."""
        mock_kite.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "RELIANCE",
                    "quantity": 10,
                    "average_price": 2500.0,
                    "last_price": 2550.0,
                    "pnl": 500.0,
                    "realised": 0.0,
                }
            ]
        }

        # First call - fetch from API
        positions1 = broker.get_positions()
        assert len(positions1) == 1

        # Second call - should use cache (no API call)
        positions2 = broker.get_positions()
        assert len(positions2) == 1

        # Should only call API once (cache is fresh)
        assert mock_kite.positions.call_count == 1


class TestGetQuote:
    """Tests for market quote retrieval."""

    def test_get_quote_success(self, broker, mock_kite):
        """Test successful quote retrieval."""
        mock_kite.quote.return_value = {
            "NSE:RELIANCE": {
                "last_price": 2550.0,
                "volume": 1000000,
                "depth": {
                    "buy": [{"price": 2549.5}],
                    "sell": [{"price": 2550.5}],
                },
            }
        }

        quote = broker.get_quote("RELIANCE")

        assert quote["symbol"] == "RELIANCE"
        assert quote["last"] == 2550.0
        assert quote["bid"] == 2549.5
        assert quote["ask"] == 2550.5
        assert quote["volume"] == 1000000

    def test_get_quote_retry_on_network_error(self, broker, mock_kite):
        """Test retry on network error."""
        mock_kite.quote.side_effect = [
            NetworkException("Timeout"),
            {
                "NSE:TCS": {
                    "last_price": 3500.0,
                    "volume": 500000,
                    "depth": {"buy": [{}], "sell": [{}]},
                }
            },
        ]

        quote = broker.get_quote("TCS")

        assert quote["last"] == 3500.0
        assert mock_kite.quote.call_count == 2


class TestGetPortfolio:
    """Tests for portfolio retrieval."""

    def test_get_portfolio(self, broker, mock_kite):
        """Test getting complete portfolio."""
        # Mock positions
        mock_kite.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "RELIANCE",
                    "quantity": 10,
                    "average_price": 2500.0,
                    "last_price": 2550.0,
                    "pnl": 500.0,
                    "realised": 0.0,
                }
            ]
        }

        # Mock cash
        mock_kite.margins.return_value = {
            "equity": {"available": {"cash": 100000.0}}
        }

        portfolio = broker.get_portfolio()

        assert portfolio.cash == 100000.0
        assert len(portfolio.positions) == 1
        assert "RELIANCE" in portfolio.positions
        assert portfolio.total_value > 0


class TestCancelOrder:
    """Tests for order cancellation."""

    def test_cancel_order_success(self, broker, mock_kite):
        """Test successful order cancellation."""
        # Add order to cache
        from tradebox.execution.base_broker import Order

        order = Order(
            order_id="ORDER999",
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=10,
            status=OrderStatus.PENDING,
        )

        with broker._cache_lock:
            broker._orders_cache["ORDER999"] = order

        # Mock get_order to return pending order
        mock_kite.orders.return_value = [
            {
                "order_id": "ORDER999",
                "tradingsymbol": "TEST",
                "transaction_type": "BUY",
                "quantity": 10,
                "status": "PENDING",
            }
        ]

        # Mock cancel_order
        mock_kite.cancel_order.return_value = None

        # Cancel
        result = broker.cancel_order("ORDER999")

        assert result is True
        mock_kite.cancel_order.assert_called_once()

        # Verify order status updated
        with broker._cache_lock:
            assert broker._orders_cache["ORDER999"].status == OrderStatus.CANCELLED

    def test_cancel_order_not_found(self, broker, mock_kite):
        """Test cancelling non-existent order."""
        mock_kite.orders.return_value = []

        result = broker.cancel_order("NONEXISTENT")

        assert result is False


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_per_second(self, broker, mock_kite):
        """Test per-second rate limiting."""
        mock_kite.place_order.return_value = "ORDER_X"
        mock_kite.orders.return_value = []

        # Place max_orders_per_second orders rapidly
        for i in range(broker.config.max_orders_per_second):
            broker.place_order("TEST", OrderSide.BUY, 1)

        # Verify all orders were tracked
        assert len(broker._order_times) == broker.config.max_orders_per_second

    def test_rate_limit_daily_exceeded(self, broker):
        """Test daily rate limit."""
        # Set daily count to limit
        broker._daily_order_count = broker.config.max_orders_per_day

        with pytest.raises(RateLimitException, match="Daily order limit reached"):
            broker.place_order("TEST", OrderSide.BUY, 1)


class TestReconciliation:
    """Tests for order reconciliation."""

    def test_reconcile_orders_discrepancy(self, broker_config, mock_kite):
        """Test order reconciliation fixes discrepancies."""
        # Enable reconciliation for this test
        broker_config.reconciliation_enabled = False  # Manual control

        with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
            broker = KiteBroker(broker_config)
            broker.kite = mock_kite

            # Add order to cache with PENDING status
            from tradebox.execution.base_broker import Order

            order = Order(
                order_id="ORDER999",
                symbol="TEST",
                side=OrderSide.BUY,
                quantity=10,
                status=OrderStatus.PENDING,
            )

            with broker._cache_lock:
                broker._orders_cache["ORDER999"] = order

            # Mock Kite API showing order as COMPLETE
            mock_kite.orders.return_value = [
                {
                    "order_id": "ORDER999",
                    "tradingsymbol": "TEST",
                    "transaction_type": "BUY",
                    "quantity": 10,
                    "status": "COMPLETE",
                    "average_price": 100.0,
                    "filled_quantity": 10,
                }
            ]

            # Run reconciliation
            broker._reconcile_orders()

            # Verify order status was updated
            with broker._cache_lock:
                reconciled_order = broker._orders_cache["ORDER999"]
                assert reconciled_order.status == OrderStatus.FILLED
                assert reconciled_order.filled_price == 100.0

    def test_reconcile_positions_discrepancy(self, broker_config, mock_kite):
        """Test position reconciliation fixes discrepancies."""
        # Disable reconciliation thread
        broker_config.reconciliation_enabled = False

        with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
            broker = KiteBroker(broker_config)
            broker.kite = mock_kite

            # Add position to cache
            from tradebox.execution.base_broker import Position

            position = Position(
                symbol="RELIANCE",
                quantity=10,
                avg_price=2500.0,
                current_price=2550.0,
            )

            with broker._cache_lock:
                broker._positions_cache["RELIANCE"] = position

            # Mock Kite showing different quantity
            mock_kite.positions.return_value = {
                "day": [
                    {
                        "tradingsymbol": "RELIANCE",
                        "quantity": 15,  # Different from cache
                        "average_price": 2500.0,
                        "last_price": 2550.0,
                        "pnl": 750.0,
                        "realised": 0.0,
                    }
                ]
            }

            # Run reconciliation
            broker._reconcile_positions()

            # Verify position was updated
            with broker._cache_lock:
                reconciled_pos = broker._positions_cache["RELIANCE"]
                assert reconciled_pos.quantity == 15


class TestClosePosition:
    """Tests for closing positions."""

    def test_close_position_success(self, broker, mock_kite):
        """Test closing a position."""
        # Mock position
        mock_kite.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "RELIANCE",
                    "quantity": 10,
                    "average_price": 2500.0,
                    "last_price": 2550.0,
                    "pnl": 500.0,
                    "realised": 0.0,
                }
            ]
        }

        # Mock order placement
        mock_kite.place_order.return_value = "ORDER_CLOSE"
        mock_kite.orders.return_value = []

        # Close position
        order = broker.close_position("RELIANCE")

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == 10

    def test_close_position_no_position(self, broker, mock_kite):
        """Test closing non-existent position."""
        mock_kite.positions.return_value = {"day": []}

        order = broker.close_position("NONEXISTENT")

        assert order is None


class TestStatusMapping:
    """Tests for status mapping."""

    def test_map_kite_status(self, broker):
        """Test Kite status to OrderStatus mapping."""
        assert broker._map_kite_status("PENDING") == OrderStatus.PENDING
        assert broker._map_kite_status("OPEN") == OrderStatus.PENDING
        assert broker._map_kite_status("COMPLETE") == OrderStatus.FILLED
        assert broker._map_kite_status("REJECTED") == OrderStatus.REJECTED
        assert broker._map_kite_status("CANCELLED") == OrderStatus.CANCELLED

        # Unknown status defaults to PENDING
        assert broker._map_kite_status("UNKNOWN") == OrderStatus.PENDING


class TestCleanup:
    """Tests for cleanup and resource management."""

    def test_stop_reconciliation(self, broker_config, mock_kite):
        """Test stopping reconciliation thread."""
        broker_config.reconciliation_enabled = True

        with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
            broker = KiteBroker(broker_config)

            assert broker._reconciliation_thread is not None
            assert broker._reconciliation_thread.is_alive()

            # Stop reconciliation
            broker.stop_reconciliation()

            # Thread should be stopped
            assert not broker._reconciliation_thread.is_alive()

    def test_close(self, broker_config, mock_kite):
        """Test explicit broker close."""
        broker_config.reconciliation_enabled = True

        with patch("tradebox.execution.kite_broker.KiteConnect", return_value=mock_kite):
            broker = KiteBroker(broker_config)

            # Close broker
            broker.close()

            # Reconciliation should be stopped
            if broker._reconciliation_thread:
                assert not broker._reconciliation_thread.is_alive()
