"""Integration tests for KiteBroker with Kite Connect API.

These tests require real Kite API credentials and are disabled by default.
To enable:
    export KITE_API_KEY="your_key"
    export KITE_API_SECRET="your_secret"
    export KITE_ACCESS_TOKEN="your_token"
    export KITE_INTEGRATION_TESTS=1

Then run:
    pytest tests/integration/test_kite_broker_integration.py -v
"""

import os

import pytest

from tradebox.execution.base_broker import OrderSide
from tradebox.execution.config import load_kite_broker_config
from tradebox.execution.kite_broker import KiteBroker


@pytest.mark.skipif(
    not os.getenv("KITE_INTEGRATION_TESTS"),
    reason="Integration tests disabled (set KITE_INTEGRATION_TESTS=1 to run)",
)
class TestKiteBrokerIntegration:
    """
    Integration tests with Kite Connect API.

    Requires:
    - KITE_API_KEY environment variable
    - KITE_API_SECRET environment variable
    - KITE_ACCESS_TOKEN environment variable (from manual login)
    - KITE_INTEGRATION_TESTS=1 environment variable to enable

    These tests use real API calls (to mock trading environment).

    IMPORTANT: Use Kite's mock/test trading environment, NOT production!
    """

    @pytest.fixture
    def broker(self):
        """Create KiteBroker with real credentials."""
        try:
            config = load_kite_broker_config()
            return KiteBroker(config)
        except ValueError as e:
            pytest.skip(f"Missing credentials: {e}")

    def test_get_quote_real_api(self, broker):
        """Test fetching real market quote."""
        quote = broker.get_quote("RELIANCE")

        # Verify quote structure
        assert "last" in quote
        assert quote["last"] > 0
        assert "bid" in quote
        assert "ask" in quote
        assert "volume" in quote
        assert "symbol" in quote
        assert quote["symbol"] == "RELIANCE"

    def test_get_positions_real_api(self, broker):
        """Test fetching real positions."""
        positions = broker.get_positions()

        # Should return list (may be empty)
        assert isinstance(positions, list)

        # If positions exist, verify structure
        for position in positions:
            assert position.symbol
            assert isinstance(position.quantity, int)
            assert isinstance(position.avg_price, float)
            assert isinstance(position.current_price, float)

    def test_get_cash_real_api(self, broker):
        """Test fetching real cash balance."""
        cash = broker.get_cash()

        # Should return non-negative float
        assert isinstance(cash, float)
        assert cash >= 0

    def test_get_portfolio_real_api(self, broker):
        """Test fetching real portfolio."""
        portfolio = broker.get_portfolio()

        # Verify portfolio structure
        assert isinstance(portfolio.cash, float)
        assert portfolio.cash >= 0
        assert isinstance(portfolio.positions, dict)
        assert isinstance(portfolio.total_value, float)
        assert portfolio.total_value >= 0

    def test_place_and_cancel_order_real_api(self, broker):
        """Test placing and cancelling an order.

        WARNING: This places a real order! Uses limit order far from market
        to avoid accidental execution.
        """
        # Get current quote
        quote = broker.get_quote("RELIANCE")
        current_price = quote["last"]

        # Place limit order 20% below market (won't fill)
        far_price = round(current_price * 0.8, 2)

        # Place order
        order = broker.place_order(
            "RELIANCE",
            OrderSide.BUY,
            1,  # Minimum quantity
            price=far_price,
        )

        # Verify order was placed
        assert order.order_id is not None
        assert order.symbol == "RELIANCE"
        assert order.side == OrderSide.BUY
        assert order.quantity == 1
        assert order.price == far_price

        # Cancel order immediately
        cancelled = broker.cancel_order(order.order_id)

        # Verify cancellation
        assert cancelled is True

        # Verify order status updated
        updated_order = broker.get_order(order.order_id)
        if updated_order:
            # May take a moment for status to update
            import time

            time.sleep(1)
            updated_order = broker.get_order(order.order_id)
            # Order should be cancelled or still pending cancellation
            assert updated_order.status.value in ["cancelled", "pending"]

    def test_get_orders_real_api(self, broker):
        """Test fetching orders."""
        orders = broker.get_orders()

        # Should return list
        assert isinstance(orders, list)

        # If orders exist, verify structure
        for order in orders:
            assert order.order_id
            assert order.symbol
            assert order.side in [OrderSide.BUY, OrderSide.SELL]
            assert order.quantity > 0

    def test_broker_cleanup(self, broker):
        """Test broker cleanup."""
        # Close broker
        broker.close()

        # Verify reconciliation stopped
        if broker._reconciliation_thread:
            assert not broker._reconciliation_thread.is_alive()


@pytest.mark.skipif(
    not os.getenv("KITE_INTEGRATION_TESTS"),
    reason="Integration tests disabled",
)
class TestKiteBrokerConfiguration:
    """Test configuration loading."""

    def test_load_config_from_env(self):
        """Test loading configuration from environment."""
        # Requires environment variables to be set
        try:
            config = load_kite_broker_config()

            # Verify config loaded
            assert config.api_key
            assert config.api_secret
            assert config.reconciliation_enabled is not None
            assert config.max_orders_per_second > 0
            assert config.retry_config is not None

        except ValueError as e:
            pytest.skip(f"Missing credentials: {e}")


# Usage instructions printed when tests are skipped
if __name__ == "__main__":
    print(
        """
Integration Tests for KiteBroker

These tests require Kite API credentials. To run them:

1. Set environment variables:
   export KITE_API_KEY="your_api_key"
   export KITE_API_SECRET="your_api_secret"
   export KITE_ACCESS_TOKEN="your_access_token"
   export KITE_INTEGRATION_TESTS=1

2. Run tests:
   pytest tests/integration/test_kite_broker_integration.py -v

WARNING: These tests make real API calls. Use a test/mock trading account!
Never use production credentials for integration tests.
"""
    )
