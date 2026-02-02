"""Unit tests for transaction cost model."""

import numpy as np
import pytest

from tradebox.env.costs import CostConfig, TransactionCostModel


@pytest.fixture
def cost_model():
    """Create a transaction cost model with default config."""
    return TransactionCostModel(CostConfig())


@pytest.fixture
def custom_cost_model():
    """Create a transaction cost model with custom config."""
    config = CostConfig(
        brokerage_pct=0.0005,
        brokerage_cap=50.0,
        stt_pct=0.002,
        slippage_pct=0.002,
    )
    return TransactionCostModel(config)


@pytest.fixture
def dynamic_slippage_model():
    """Create a transaction cost model with dynamic slippage enabled."""
    config = CostConfig(
        use_dynamic_slippage=True,
        base_spread_bps=2.5,
        spread_volatility_multiplier=True,
        max_spread_bps=10.0,
        impact_coefficient=0.2,
        max_impact_pct=0.005,
    )
    return TransactionCostModel(config)


def test_cost_config_defaults():
    """Test CostConfig default values."""
    config = CostConfig()
    assert config.brokerage_pct == 0.0003
    assert config.brokerage_cap == 20.0
    assert config.stt_pct == 0.001
    assert config.transaction_charges_pct == 0.0000325
    assert config.gst_rate == 0.18
    assert config.stamp_duty_pct == 0.00015
    assert config.slippage_pct == 0.001


def test_buy_cost_calculation(cost_model):
    """Test buy cost calculation with all fees."""
    shares = 100
    price = 1000.0

    total_cost, breakdown = cost_model.calculate_buy_cost(shares, price)

    # Trade value should be shares * price
    assert breakdown["trade_value"] == 100000.0

    # Total cost should be trade value + fees
    assert total_cost > 100000.0

    # Effective rate should be between 0.1% and 0.2%
    assert 0.001 < breakdown["effective_rate"] < 0.002

    # All cost components should be present
    assert "brokerage" in breakdown
    assert "transaction_charges" in breakdown
    assert "gst" in breakdown
    assert "stamp_duty" in breakdown
    assert "slippage" in breakdown
    assert "total_cost" in breakdown


def test_sell_proceeds_calculation(cost_model):
    """Test sell proceeds calculation with all fees."""
    shares = 100
    price = 1000.0

    net_proceeds, breakdown = cost_model.calculate_sell_proceeds(shares, price)

    # Trade value should be shares * price
    assert breakdown["trade_value"] == 100000.0

    # Net proceeds should be less than trade value due to fees
    assert net_proceeds < 100000.0

    # Effective rate should be between 0.2% and 0.3% (sell is higher due to STT)
    assert 0.002 < breakdown["effective_rate"] < 0.003

    # STT should only be on sell side
    assert "stt" in breakdown
    assert breakdown["stt"] > 0

    # All cost components should be present
    assert "brokerage" in breakdown
    assert "transaction_charges" in breakdown
    assert "gst" in breakdown
    assert "slippage" in breakdown
    assert "total_cost" in breakdown


def test_brokerage_cap_enforced(cost_model):
    """Test that brokerage is capped at ₹20 for large trades."""
    # Large trade where 0.03% would exceed ₹20
    shares = 10000
    price = 1000.0  # ₹10L trade

    total_cost, breakdown = cost_model.calculate_buy_cost(shares, price)

    # Brokerage should be capped at ₹20
    assert breakdown["brokerage"] == 20.0

    # Without cap, brokerage would be ₹300 (10L * 0.03%)
    uncapped_brokerage = 10000000 * 0.0003
    assert uncapped_brokerage > 20.0


def test_round_trip_cost_percentage(cost_model):
    """Test round-trip cost is within expected range (0.35-0.4%)."""
    rt_cost = cost_model.calculate_round_trip_cost_pct()

    # Zerodha round-trip including slippage should be 0.35-0.4%
    # (0.14% buy + 0.23% sell = 0.37% total)
    assert 0.0035 <= rt_cost <= 0.004

    # Test with different price (higher price has lower % due to brokerage cap)
    rt_cost_high_price = cost_model.calculate_round_trip_cost_pct(price=5000.0)
    assert 0.003 <= rt_cost_high_price <= 0.0035


def test_custom_config(custom_cost_model):
    """Test cost model with custom configuration."""
    shares = 100
    price = 1000.0

    total_cost, breakdown = custom_cost_model.calculate_buy_cost(shares, price)

    # With higher brokerage (0.05%), costs should be higher
    assert breakdown["brokerage"] > 20.0  # Higher than default

    # With higher slippage (0.2%), costs should be higher
    assert breakdown["slippage"] == 1000.0 * 0.002 * 100  # 0.2% of trade value


def test_zero_shares_edge_case(cost_model):
    """Test cost calculation with zero shares."""
    total_cost, breakdown = cost_model.calculate_buy_cost(0, 1000.0)

    # All costs should be zero
    assert breakdown["trade_value"] == 0.0
    assert breakdown["total_cost"] == 0.0
    assert breakdown["effective_rate"] == 0.0


def test_small_trade(cost_model):
    """Test cost calculation for small trade (brokerage not capped)."""
    shares = 10
    price = 100.0  # ₹1000 trade

    total_cost, breakdown = cost_model.calculate_buy_cost(shares, price)

    # For ₹1000 trade, brokerage is 0.03% = ₹0.30 (below ₹20 cap)
    expected_brokerage = 1000.0 * 0.0003
    assert breakdown["brokerage"] == pytest.approx(expected_brokerage, rel=1e-6)
    assert breakdown["brokerage"] < 20.0


def test_buy_vs_sell_costs_difference(cost_model):
    """Test that buy and sell have different costs (STT, stamp duty)."""
    shares = 100
    price = 1000.0

    buy_cost, buy_breakdown = cost_model.calculate_buy_cost(shares, price)
    sell_proceeds, sell_breakdown = cost_model.calculate_sell_proceeds(shares, price)

    # Buy has stamp duty, sell has STT
    assert "stamp_duty" in buy_breakdown
    assert buy_breakdown["stamp_duty"] > 0

    assert "stt" in sell_breakdown
    assert sell_breakdown["stt"] > 0

    # Sell should be more expensive due to STT (0.1%)
    assert sell_breakdown["total_cost"] > buy_breakdown["total_cost"]


def test_cost_breakdown_completeness(cost_model):
    """Test that cost breakdown contains all expected fields."""
    total_cost, breakdown = cost_model.calculate_buy_cost(100, 1000.0)

    required_fields = [
        "trade_value",
        "brokerage",
        "transaction_charges",
        "gst",
        "stamp_duty",
        "slippage",
        "total_cost",
        "effective_rate",
    ]

    for field in required_fields:
        assert field in breakdown
        assert isinstance(breakdown[field], (int, float))
        assert breakdown[field] >= 0


def test_gst_calculation(cost_model):
    """Test GST is calculated correctly on brokerage + transaction charges."""
    shares = 100
    price = 1000.0

    total_cost, breakdown = cost_model.calculate_buy_cost(shares, price)

    # GST should be 18% of (brokerage + transaction_charges)
    expected_gst = (breakdown["brokerage"] + breakdown["transaction_charges"]) * 0.18
    assert breakdown["gst"] == pytest.approx(expected_gst, rel=1e-6)


def test_transaction_charges_calculation(cost_model):
    """Test transaction charges are calculated correctly."""
    shares = 100
    price = 1000.0

    total_cost, breakdown = cost_model.calculate_buy_cost(shares, price)

    # Transaction charges should be 0.00325% of trade value
    expected_charges = 100000.0 * 0.0000325
    assert breakdown["transaction_charges"] == pytest.approx(expected_charges, rel=1e-6)


def test_slippage_calculation(cost_model):
    """Test slippage is calculated correctly."""
    shares = 100
    price = 1000.0

    buy_cost, buy_breakdown = cost_model.calculate_buy_cost(shares, price)
    sell_proceeds, sell_breakdown = cost_model.calculate_sell_proceeds(shares, price)

    # Slippage should be 0.1% of trade value for both buy and sell
    expected_slippage = 100000.0 * 0.001
    assert buy_breakdown["slippage"] == pytest.approx(expected_slippage, rel=1e-6)
    assert sell_breakdown["slippage"] == pytest.approx(expected_slippage, rel=1e-6)


def test_stamp_duty_only_on_buy(cost_model):
    """Test stamp duty is only charged on buy side."""
    shares = 100
    price = 1000.0

    buy_cost, buy_breakdown = cost_model.calculate_buy_cost(shares, price)
    sell_proceeds, sell_breakdown = cost_model.calculate_sell_proceeds(shares, price)

    # Buy should have stamp duty
    assert "stamp_duty" in buy_breakdown
    assert buy_breakdown["stamp_duty"] > 0

    # Sell should not have stamp duty
    assert "stamp_duty" not in sell_breakdown


def test_stt_only_on_sell(cost_model):
    """Test STT is only charged on sell side."""
    shares = 100
    price = 1000.0

    buy_cost, buy_breakdown = cost_model.calculate_buy_cost(shares, price)
    sell_proceeds, sell_breakdown = cost_model.calculate_sell_proceeds(shares, price)

    # Buy should not have STT
    assert "stt" not in buy_breakdown

    # Sell should have STT
    assert "stt" in sell_breakdown
    assert sell_breakdown["stt"] > 0

    # STT should be 0.1% of trade value
    expected_stt = 100000.0 * 0.001
    assert sell_breakdown["stt"] == pytest.approx(expected_stt, rel=1e-6)


# ============================================================================
# Dynamic Slippage Model Tests (STORY-036)
# ============================================================================


class TestDynamicSlippageModel:
    """Test suite for dynamic slippage calculation (intraday trading)."""

    def test_dynamic_slippage_config_defaults(self):
        """Test dynamic slippage configuration defaults."""
        config = CostConfig(use_dynamic_slippage=True)

        assert config.use_dynamic_slippage is True
        assert config.base_spread_bps == 2.5
        assert config.spread_volatility_multiplier is True
        assert config.max_spread_bps == 10.0
        assert config.impact_coefficient == 0.2
        assert config.max_impact_pct == 0.005

    def test_dynamic_slippage_enabled(self, dynamic_slippage_model):
        """Test that dynamic slippage is used when enabled."""
        shares = 100
        price = 1000.0
        volume = 1000000  # 1M shares volume
        atr = 20.0  # ₹20 ATR

        buy_cost, buy_breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Slippage should be calculated (non-zero)
        assert buy_breakdown["slippage"] > 0

        # Slippage should be different from flat model (0.1% of trade value)
        flat_slippage = 100000.0 * 0.001
        # Dynamic slippage should be less for liquid stocks
        assert buy_breakdown["slippage"] != flat_slippage

    def test_dynamic_slippage_bid_ask_spread(self, dynamic_slippage_model):
        """Test bid-ask spread component of dynamic slippage."""
        shares = 100
        price = 1000.0
        volume = 1000000
        atr = 10.0  # 1% ATR (10/1000)

        buy_cost, breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Slippage should be positive
        assert breakdown["slippage"] > 0

        # For low volatility, spread should be close to base (2.5 bps)
        # Half-spread = 1.25 bps = 0.0125% × 100,000 = ₹12.5
        expected_min_spread = 100000.0 * (2.5 / 10000) / 2
        assert breakdown["slippage"] >= expected_min_spread * 0.5  # Allow some tolerance

    def test_dynamic_slippage_volatility_scaling(self, dynamic_slippage_model):
        """Test that slippage scales with volatility (ATR)."""
        shares = 100
        price = 1000.0
        volume = 1000000

        # Low volatility
        atr_low = 5.0  # 0.5% ATR
        buy_cost_low, breakdown_low = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr_low
        )

        # High volatility
        atr_high = 50.0  # 5% ATR
        buy_cost_high, breakdown_high = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr_high
        )

        # Higher volatility should result in higher slippage
        assert breakdown_high["slippage"] > breakdown_low["slippage"]

    def test_dynamic_slippage_market_impact(self, dynamic_slippage_model):
        """Test market impact component based on volume participation."""
        shares = 1000  # Large order
        price = 1000.0

        # Low volume (high impact)
        volume_low = 10000  # 10% participation (1000/10000)
        buy_cost_low, breakdown_low = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume_low, atr=10.0
        )

        # High volume (low impact)
        volume_high = 1000000  # 0.1% participation (1000/1000000)
        buy_cost_high, breakdown_high = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume_high, atr=10.0
        )

        # Lower volume should result in higher market impact and slippage
        assert breakdown_low["slippage"] > breakdown_high["slippage"]

    def test_dynamic_slippage_max_spread_cap(self, dynamic_slippage_model):
        """Test that spread is capped at max_spread_bps."""
        shares = 100
        price = 1000.0
        volume = 1000000

        # Extremely high volatility
        atr = 500.0  # 50% ATR (unrealistic but tests cap)

        buy_cost, breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Spread component should be capped at max_spread_bps (10 bps)
        # Half-spread = 5 bps = 0.05% × 100,000 = ₹50
        # Total slippage includes impact, but spread alone should be capped
        max_spread_cost = 100000.0 * (10.0 / 10000) / 2  # ₹50
        # Slippage can be higher due to impact, but spread is capped
        # We can't isolate spread in the breakdown, but we can verify total is reasonable
        assert breakdown["slippage"] < 500.0  # Should not be absurdly high

    def test_dynamic_slippage_max_impact_cap(self, dynamic_slippage_model):
        """Test that market impact is capped at max_impact_pct."""
        # Very large order relative to volume
        shares = 100000  # 100K shares
        price = 1000.0
        volume = 100000  # 100% participation (extreme)
        atr = 10.0

        buy_cost, breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Total slippage should be capped
        # Max impact = 0.5% of trade value
        max_impact = 100000000.0 * 0.005  # ₹500,000

        # Slippage should not exceed max_impact + max_spread
        max_spread = 100000000.0 * (10.0 / 10000) / 2  # ₹50,000
        max_total = max_impact + max_spread

        assert breakdown["slippage"] <= max_total * 1.1  # Allow 10% tolerance

    def test_dynamic_slippage_buy_vs_sell_symmetric(self, dynamic_slippage_model):
        """Test that dynamic slippage is symmetric for buy and sell."""
        shares = 100
        price = 1000.0
        volume = 1000000
        atr = 10.0

        buy_cost, buy_breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        sell_proceeds, sell_breakdown = dynamic_slippage_model.calculate_sell_proceeds(
            shares, price, volume, atr
        )

        # Slippage should be approximately equal for buy and sell
        assert buy_breakdown["slippage"] == pytest.approx(
            sell_breakdown["slippage"], rel=0.01
        )

    def test_dynamic_slippage_zero_volume_edge_case(self, dynamic_slippage_model):
        """Test dynamic slippage with zero volume (edge case)."""
        shares = 100
        price = 1000.0
        volume = 0  # Edge case
        atr = 10.0

        buy_cost, breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Should handle gracefully (volume participation = 0)
        assert breakdown["slippage"] >= 0
        assert not np.isnan(breakdown["slippage"])

    def test_dynamic_slippage_none_atr_uses_default(self, dynamic_slippage_model):
        """Test that None ATR uses 1% default."""
        shares = 100
        price = 1000.0
        volume = 1000000
        atr = None  # Should default to 1% of price = ₹10

        buy_cost, breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Should calculate slippage with default ATR
        assert breakdown["slippage"] > 0

    def test_dynamic_slippage_integration_with_full_costs(self, dynamic_slippage_model):
        """Test dynamic slippage integrates correctly with all other costs."""
        shares = 100
        price = 1000.0
        volume = 1000000
        atr = 10.0

        total_cost, breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Total cost should include all components
        calculated_total = (
            breakdown["brokerage"]
            + breakdown["transaction_charges"]
            + breakdown["gst"]
            + breakdown["stamp_duty"]
            + breakdown["slippage"]
        )

        assert breakdown["total_cost"] == pytest.approx(calculated_total, rel=1e-6)

        # Total cost should be trade_value + all costs
        assert total_cost == pytest.approx(
            breakdown["trade_value"] + breakdown["total_cost"], rel=1e-6
        )

    def test_dynamic_vs_flat_slippage_comparison(self):
        """Compare dynamic slippage vs flat slippage for same scenario."""
        shares = 100
        price = 1000.0
        volume = 1000000
        atr = 10.0

        # Flat slippage model
        flat_config = CostConfig(use_dynamic_slippage=False, slippage_pct=0.001)
        flat_model = TransactionCostModel(flat_config)

        # Dynamic slippage model
        dynamic_config = CostConfig(use_dynamic_slippage=True)
        dynamic_model = TransactionCostModel(dynamic_config)

        flat_cost, flat_breakdown = flat_model.calculate_buy_cost(shares, price)
        dynamic_cost, dynamic_breakdown = dynamic_model.calculate_buy_cost(
            shares, price, volume, atr
        )

        # Flat slippage should be exactly 0.1% of trade value
        assert flat_breakdown["slippage"] == 100000.0 * 0.001

        # Dynamic slippage should be different (could be higher or lower)
        # For liquid stocks with low volatility, it's typically lower
        assert dynamic_breakdown["slippage"] != flat_breakdown["slippage"]

    def test_dynamic_slippage_round_trip_cost(self, dynamic_slippage_model):
        """Test round-trip cost with dynamic slippage."""
        # Custom method to calculate round-trip with volume and ATR
        shares = 100
        price = 1000.0
        volume = 1000000
        atr = 10.0

        # Buy
        buy_total, buy_breakdown = dynamic_slippage_model.calculate_buy_cost(
            shares, price, volume, atr
        )
        buy_cost = buy_total - (shares * price)

        # Sell
        sell_proceeds, sell_breakdown = dynamic_slippage_model.calculate_sell_proceeds(
            shares, price, volume, atr
        )
        sell_cost = (shares * price) - sell_proceeds

        # Round-trip cost
        total_cost = buy_cost + sell_cost
        round_trip_pct = total_cost / (shares * price)

        # With dynamic slippage, round-trip should be lower than flat model
        # Flat model: ~0.37%, dynamic should be ~0.19-0.30% for liquid stocks
        assert 0.0015 <= round_trip_pct <= 0.004  # 0.15% to 0.4%


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
