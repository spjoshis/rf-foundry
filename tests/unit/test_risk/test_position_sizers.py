"""Unit tests for position sizing strategies."""

import pytest
import numpy as np

from tradebox.risk.position_sizers import (
    PositionSizerConfig,
    KellyEstimator,
    PositionSizer,
    FixedPositionSizer,
    VolatilityPositionSizer,
    RiskParityPositionSizer,
    KellyPositionSizer,
    create_position_sizer,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_config():
    """Create base position sizer config."""
    return PositionSizerConfig(min_position=0.10, max_position=0.20)


@pytest.fixture
def kelly_estimator():
    """Create Kelly estimator."""
    return KellyEstimator(alpha=0.05, min_trades=20)


# ============================================================================
# PositionSizerConfig Tests
# ============================================================================


def test_position_sizer_config_defaults():
    """Test PositionSizerConfig default values."""
    config = PositionSizerConfig()

    assert config.strategy == "fixed"
    assert config.min_position == 0.10
    assert config.max_position == 0.20
    assert config.fixed_fraction == 0.20
    assert config.kelly_fraction == 0.25
    assert config.vol_risk_per_trade == 0.02
    assert config.rp_target_volatility == 0.12


def test_position_sizer_config_custom_values():
    """Test PositionSizerConfig with custom values."""
    config = PositionSizerConfig(
        strategy="volatility",
        min_position=0.05,
        max_position=0.30,
        vol_risk_per_trade=0.03,
    )

    assert config.strategy == "volatility"
    assert config.min_position == 0.05
    assert config.max_position == 0.30
    assert config.vol_risk_per_trade == 0.03


# ============================================================================
# KellyEstimator Tests
# ============================================================================


def test_kelly_estimator_initialization():
    """Test KellyEstimator initialization."""
    estimator = KellyEstimator(alpha=0.05, min_trades=20, prior_win_rate=0.50, prior_payoff=1.5)

    assert estimator.alpha == 0.05
    assert estimator.min_trades == 20
    assert estimator.prior_win_rate == 0.50
    assert estimator.prior_payoff == 1.5
    assert estimator.trade_count == 0
    assert estimator.ema_win_rate == 0.50


def test_kelly_estimator_invalid_parameters():
    """Test KellyEstimator rejects invalid parameters."""
    # Invalid alpha
    with pytest.raises(ValueError, match="alpha must be in"):
        KellyEstimator(alpha=0)

    # Invalid min_trades
    with pytest.raises(ValueError, match="min_trades must be"):
        KellyEstimator(min_trades=0)

    # Invalid prior_win_rate
    with pytest.raises(ValueError, match="prior_win_rate must be"):
        KellyEstimator(prior_win_rate=1.5)

    # Invalid prior_payoff
    with pytest.raises(ValueError, match="prior_payoff must be"):
        KellyEstimator(prior_payoff=-1.0)


def test_kelly_estimator_update_winning_trade():
    """Test KellyEstimator update with winning trade."""
    estimator = KellyEstimator(alpha=0.05, prior_win_rate=0.50)

    # Update with 5% gain
    estimator.update(pnl_pct=0.05)

    assert estimator.trade_count == 1
    # EMA win rate should increase (0.05 × 1.0 + 0.95 × 0.50 = 0.525)
    assert estimator.ema_win_rate == pytest.approx(0.525, abs=1e-6)
    assert estimator.ema_avg_win == pytest.approx(0.05, abs=1e-6)


def test_kelly_estimator_update_losing_trade():
    """Test KellyEstimator update with losing trade."""
    estimator = KellyEstimator(alpha=0.05, prior_win_rate=0.50)

    # Update with 3% loss
    estimator.update(pnl_pct=-0.03)

    assert estimator.trade_count == 1
    # EMA win rate should decrease (0.05 × 0.0 + 0.95 × 0.50 = 0.475)
    assert estimator.ema_win_rate == pytest.approx(0.475, abs=1e-6)
    assert estimator.ema_avg_loss == pytest.approx(0.03, abs=1e-6)


def test_kelly_estimator_multiple_trades():
    """Test KellyEstimator with multiple trades."""
    estimator = KellyEstimator(alpha=0.1, prior_win_rate=0.50)  # Higher alpha for faster updates

    # Simulate 10 trades: 6 wins, 4 losses (60% win rate)
    # Interleave wins and losses for realistic EMA behavior
    for _ in range(4):
        estimator.update(pnl_pct=0.05)  # Win
        estimator.update(pnl_pct=-0.03)  # Loss
    estimator.update(pnl_pct=0.05)  # Final 2 wins
    estimator.update(pnl_pct=0.05)

    assert estimator.trade_count == 10
    # EMA win rate should be valid (between 0 and 1)
    assert 0 < estimator.ema_win_rate < 1
    # With 60% wins, should see reasonable estimates
    assert estimator.ema_avg_win > 0
    assert estimator.ema_avg_loss > 0


def test_kelly_estimator_get_kelly_fraction_insufficient_trades():
    """Test Kelly fraction with insufficient trade history (use priors)."""
    estimator = KellyEstimator(alpha=0.05, min_trades=20, prior_win_rate=0.50, prior_payoff=1.5)

    # Only 10 trades (< min_trades = 20)
    for _ in range(10):
        estimator.update(pnl_pct=0.05)

    kelly_f = estimator.get_kelly_fraction(kelly_fraction=0.25)

    # Should blend prior and observed (Bayesian)
    assert isinstance(kelly_f, float)
    assert kelly_f >= 0  # Valid Kelly fraction


def test_kelly_estimator_get_kelly_fraction_sufficient_trades():
    """Test Kelly fraction with sufficient trade history."""
    estimator = KellyEstimator(alpha=0.05, min_trades=20)

    # 25 trades: 15 wins (60%), 10 losses (40%)
    # Average win = 5%, average loss = 3%
    for _ in range(15):
        estimator.update(pnl_pct=0.05)
    for _ in range(10):
        estimator.update(pnl_pct=-0.03)

    kelly_f = estimator.get_kelly_fraction(kelly_fraction=0.25)

    # With 60% win rate, 5%/3% = 1.67 payoff, Kelly > 0
    assert kelly_f > 0


def test_kelly_estimator_negative_kelly():
    """Test Kelly fraction when edge is negative."""
    estimator = KellyEstimator(alpha=0.1, min_trades=20, prior_win_rate=0.30, prior_payoff=0.5)

    # Simulate losing strategy: 70% losses
    for _ in range(25):
        for _ in range(3):
            estimator.update(pnl_pct=-0.05)  # 70% losses
        estimator.update(pnl_pct=0.03)  # 30% wins (small)

    kelly_f = estimator.get_kelly_fraction(kelly_fraction=0.25)

    # Should be small or negative (no edge)
    assert kelly_f <= 0.05  # Very small or negative


def test_kelly_estimator_reset():
    """Test KellyEstimator reset."""
    estimator = KellyEstimator(alpha=0.05, prior_win_rate=0.50)

    # Build up state
    for _ in range(10):
        estimator.update(pnl_pct=0.05)

    assert estimator.trade_count == 10

    # Reset
    estimator.reset()

    assert estimator.trade_count == 0
    assert estimator.ema_win_rate == 0.50  # Back to prior
    assert estimator.ema_avg_win == 0.0
    assert estimator.ema_avg_loss == 0.0


# ============================================================================
# FixedPositionSizer Tests
# ============================================================================


def test_fixed_position_sizer_initialization():
    """Test FixedPositionSizer initialization."""
    sizer = FixedPositionSizer(fixed_fraction=0.20, max_position=0.20)

    assert sizer.fixed_fraction == 0.20
    assert sizer.min_position == 0.10
    assert sizer.max_position == 0.20


def test_fixed_position_sizer_calculate():
    """Test FixedPositionSizer always returns fixed fraction."""
    sizer = FixedPositionSizer(fixed_fraction=0.15)

    # Should return same value regardless of inputs
    assert sizer.calculate() == 0.15
    assert sizer.calculate(price=100, atr=5) == 0.15
    assert sizer.calculate(volatility=0.2) == 0.15


def test_fixed_position_sizer_clamping():
    """Test FixedPositionSizer clamps to min/max."""
    # Fixed fraction above max
    sizer = FixedPositionSizer(fixed_fraction=0.30, min_position=0.10, max_position=0.20)
    assert sizer.fixed_fraction == 0.20  # Clamped to max

    # Fixed fraction below min
    sizer = FixedPositionSizer(fixed_fraction=0.05, min_position=0.10, max_position=0.20)
    assert sizer.fixed_fraction == 0.10  # Clamped to min


# ============================================================================
# VolatilityPositionSizer Tests
# ============================================================================


def test_volatility_position_sizer_initialization():
    """Test VolatilityPositionSizer initialization."""
    sizer = VolatilityPositionSizer(risk_per_trade=0.02, atr_multiplier=2.0)

    assert sizer.risk_per_trade == 0.02
    assert sizer.atr_multiplier == 2.0
    assert sizer.min_position == 0.10
    assert sizer.max_position == 0.20


def test_volatility_position_sizer_invalid_parameters():
    """Test VolatilityPositionSizer rejects invalid parameters."""
    # Invalid risk_per_trade
    with pytest.raises(ValueError, match="risk_per_trade must be"):
        VolatilityPositionSizer(risk_per_trade=0)

    # Invalid atr_multiplier
    with pytest.raises(ValueError, match="atr_multiplier must be"):
        VolatilityPositionSizer(atr_multiplier=-1.0)


def test_volatility_position_sizer_calculate():
    """Test VolatilityPositionSizer calculation."""
    sizer = VolatilityPositionSizer(risk_per_trade=0.02, atr_multiplier=2.0, max_position=0.20)

    # Portfolio: ₹100,000, Risk: 2% = ₹2,000
    # Stock: ₹2,500, ATR: ₹50, Stop: 2×50 = ₹100
    # Shares: 2000/100 = 20 shares
    # Position value: 20 × 2500 = ₹50,000
    # Position fraction: 50,000 / 100,000 = 0.50, clamped to 0.20
    position = sizer.calculate(price=2500.0, atr=50.0, portfolio_value=100000.0)

    assert position == 0.20  # Clamped to max


def test_volatility_position_sizer_high_volatility():
    """Test VolatilityPositionSizer with high volatility (small position)."""
    sizer = VolatilityPositionSizer(risk_per_trade=0.02, atr_multiplier=2.0)

    # High ATR → large stop → fewer shares → smaller position
    # Portfolio: ₹100,000, Risk: ₹2,000
    # Stock: ₹2,500, ATR: ₹200 (high!), Stop: ₹400
    # Shares: 2000/400 = 5 shares
    # Position value: 5 × 2500 = ₹12,500
    # Position fraction: 12,500 / 100,000 = 0.125
    position = sizer.calculate(price=2500.0, atr=200.0, portfolio_value=100000.0)

    assert position == pytest.approx(0.125, abs=0.01)


def test_volatility_position_sizer_zero_atr():
    """Test VolatilityPositionSizer with zero ATR."""
    sizer = VolatilityPositionSizer()

    # Zero ATR should return max_position
    position = sizer.calculate(price=2500.0, atr=0.0, portfolio_value=100000.0)

    assert position == sizer.max_position


def test_volatility_position_sizer_invalid_inputs():
    """Test VolatilityPositionSizer with invalid inputs."""
    sizer = VolatilityPositionSizer()

    # Invalid price
    with pytest.raises(ValueError, match="price must be"):
        sizer.calculate(price=-100, atr=50.0, portfolio_value=100000.0)

    # Invalid ATR
    with pytest.raises(ValueError, match="atr must be"):
        sizer.calculate(price=2500.0, atr=-10.0, portfolio_value=100000.0)

    # Invalid portfolio_value
    with pytest.raises(ValueError, match="portfolio_value must be"):
        sizer.calculate(price=2500.0, atr=50.0, portfolio_value=0)


# ============================================================================
# RiskParityPositionSizer Tests
# ============================================================================


def test_risk_parity_position_sizer_initialization():
    """Test RiskParityPositionSizer initialization."""
    sizer = RiskParityPositionSizer(target_volatility=0.12)

    assert sizer.target_volatility == 0.12
    assert sizer.min_position == 0.10
    assert sizer.max_position == 0.20


def test_risk_parity_position_sizer_invalid_target_volatility():
    """Test RiskParityPositionSizer rejects invalid target volatility."""
    with pytest.raises(ValueError, match="target_volatility must be"):
        RiskParityPositionSizer(target_volatility=0)


def test_risk_parity_position_sizer_calculate():
    """Test RiskParityPositionSizer calculation."""
    sizer = RiskParityPositionSizer(target_volatility=0.12, max_position=0.20)

    # Asset volatility = 20%, target = 12%
    # Position = 12% / 20% = 0.60, clamped to 0.20
    position = sizer.calculate(volatility=0.20)

    assert position == 0.20  # Clamped to max


def test_risk_parity_position_sizer_low_volatility():
    """Test RiskParityPositionSizer with low volatility (large position)."""
    sizer = RiskParityPositionSizer(target_volatility=0.12, max_position=0.30)

    # Low volatility = 10%, target = 12%
    # Position = 12% / 10% = 1.20, clamped to 0.30
    position = sizer.calculate(volatility=0.10)

    assert position == 0.30  # Clamped to max


def test_risk_parity_position_sizer_high_volatility():
    """Test RiskParityPositionSizer with high volatility (small position)."""
    sizer = RiskParityPositionSizer(target_volatility=0.12, min_position=0.05)

    # High volatility = 30%, target = 12%
    # Position = 12% / 30% = 0.40, clamped within limits
    position = sizer.calculate(volatility=0.30)

    assert position == pytest.approx(0.20, abs=0.01)  # Clamped to max


def test_risk_parity_position_sizer_zero_volatility():
    """Test RiskParityPositionSizer with zero volatility."""
    sizer = RiskParityPositionSizer(target_volatility=0.12)

    # Zero volatility should return max_position
    position = sizer.calculate(volatility=0.0)

    assert position == sizer.max_position


def test_risk_parity_position_sizer_invalid_volatility():
    """Test RiskParityPositionSizer with invalid volatility."""
    sizer = RiskParityPositionSizer()

    with pytest.raises(ValueError, match="volatility must be"):
        sizer.calculate(volatility=-0.1)


# ============================================================================
# KellyPositionSizer Tests
# ============================================================================


def test_kelly_position_sizer_initialization():
    """Test KellyPositionSizer initialization."""
    estimator = KellyEstimator()
    sizer = KellyPositionSizer(kelly_fraction=0.25, estimator=estimator)

    assert sizer.kelly_fraction == 0.25
    assert sizer.estimator == estimator


def test_kelly_position_sizer_invalid_kelly_fraction():
    """Test KellyPositionSizer rejects invalid kelly_fraction."""
    with pytest.raises(ValueError, match="kelly_fraction must be"):
        KellyPositionSizer(kelly_fraction=0)

    with pytest.raises(ValueError, match="kelly_fraction must be"):
        KellyPositionSizer(kelly_fraction=1.5)


def test_kelly_position_sizer_calculate_with_estimator():
    """Test KellyPositionSizer calculation with estimator."""
    estimator = KellyEstimator(alpha=0.1, min_trades=20)
    sizer = KellyPositionSizer(kelly_fraction=0.25, estimator=estimator)

    # Simulate winning strategy: 60% win rate, 5%/3% payoff
    for _ in range(25):
        for _ in range(3):
            estimator.update(pnl_pct=0.05)  # 60% wins
        for _ in range(2):
            estimator.update(pnl_pct=-0.03)  # 40% losses

    position = sizer.calculate()

    # Should be positive (positive edge)
    assert position > 0
    assert position <= sizer.max_position


def test_kelly_position_sizer_negative_kelly():
    """Test KellyPositionSizer returns 0 for negative Kelly."""
    estimator = KellyEstimator(alpha=0.1, min_trades=20, prior_win_rate=0.30, prior_payoff=0.5)
    sizer = KellyPositionSizer(kelly_fraction=0.25, estimator=estimator)

    # Simulate losing strategy
    for _ in range(25):
        for _ in range(3):
            estimator.update(pnl_pct=-0.05)  # 70% losses
        estimator.update(pnl_pct=0.02)  # 30% small wins

    position = sizer.calculate()

    # Should be 0 (no edge)
    assert position == 0.0


def test_kelly_position_sizer_no_estimator():
    """Test KellyPositionSizer raises error without estimator."""
    sizer = KellyPositionSizer(kelly_fraction=0.25)

    with pytest.raises(ValueError, match="requires a KellyEstimator"):
        sizer.calculate()


def test_kelly_position_sizer_override_estimator():
    """Test KellyPositionSizer with overridden estimator in calculate()."""
    estimator1 = KellyEstimator(alpha=0.05)
    estimator2 = KellyEstimator(alpha=0.1)

    sizer = KellyPositionSizer(kelly_fraction=0.25, estimator=estimator1)

    # Provide different estimator at calculate time
    for _ in range(25):
        estimator2.update(pnl_pct=0.05)

    position = sizer.calculate(estimator=estimator2)

    # Should use estimator2
    assert position > 0


# ============================================================================
# Base PositionSizer Tests
# ============================================================================


def test_position_sizer_base_class_invalid_parameters():
    """Test PositionSizer base class validates parameters."""
    # min_position out of range
    with pytest.raises(ValueError, match="min_position must be"):
        FixedPositionSizer(fixed_fraction=0.20, min_position=-0.1)

    # max_position out of range
    with pytest.raises(ValueError, match="max_position must be"):
        FixedPositionSizer(fixed_fraction=0.20, max_position=1.5)

    # min > max
    with pytest.raises(ValueError, match="min_position.*must be <="):
        FixedPositionSizer(fixed_fraction=0.20, min_position=0.30, max_position=0.20)


# ============================================================================
# Factory Function Tests
# ============================================================================


def test_create_position_sizer_fixed():
    """Test factory creates FixedPositionSizer."""
    config = PositionSizerConfig(strategy='fixed', fixed_fraction=0.15)
    sizer = create_position_sizer(config)

    assert isinstance(sizer, FixedPositionSizer)
    assert sizer.fixed_fraction == 0.15


def test_create_position_sizer_volatility():
    """Test factory creates VolatilityPositionSizer."""
    config = PositionSizerConfig(strategy='volatility', vol_risk_per_trade=0.03)
    sizer = create_position_sizer(config)

    assert isinstance(sizer, VolatilityPositionSizer)
    assert sizer.risk_per_trade == 0.03


def test_create_position_sizer_risk_parity():
    """Test factory creates RiskParityPositionSizer."""
    config = PositionSizerConfig(strategy='risk_parity', rp_target_volatility=0.15)
    sizer = create_position_sizer(config)

    assert isinstance(sizer, RiskParityPositionSizer)
    assert sizer.target_volatility == 0.15


def test_create_position_sizer_kelly():
    """Test factory creates KellyPositionSizer with estimator."""
    config = PositionSizerConfig(strategy='kelly', kelly_fraction=0.50)
    sizer = create_position_sizer(config)

    assert isinstance(sizer, KellyPositionSizer)
    assert sizer.kelly_fraction == 0.50
    assert sizer.estimator is not None
    assert isinstance(sizer.estimator, KellyEstimator)


def test_create_position_sizer_invalid_strategy():
    """Test factory raises error for unknown strategy."""
    config = PositionSizerConfig(strategy='invalid')

    with pytest.raises(ValueError, match="Unknown strategy"):
        create_position_sizer(config)


def test_all_sizers_inherit_from_base():
    """Test that all sizers inherit from PositionSizer."""
    assert issubclass(FixedPositionSizer, PositionSizer)
    assert issubclass(VolatilityPositionSizer, PositionSizer)
    assert issubclass(RiskParityPositionSizer, PositionSizer)
    assert issubclass(KellyPositionSizer, PositionSizer)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_all_sizers_handle_extreme_values():
    """Test all sizers handle extreme but valid values."""
    config_fixed = PositionSizerConfig(strategy='fixed')
    config_vol = PositionSizerConfig(strategy='volatility')
    config_rp = PositionSizerConfig(strategy='risk_parity')

    fixed_sizer = create_position_sizer(config_fixed)
    vol_sizer = create_position_sizer(config_vol)
    rp_sizer = create_position_sizer(config_rp)

    # Fixed sizer (no inputs needed)
    assert 0 <= fixed_sizer.calculate() <= 1

    # Volatility sizer with very high price
    position = vol_sizer.calculate(price=100000.0, atr=1000.0, portfolio_value=1000000.0)
    assert 0 <= position <= vol_sizer.max_position

    # Risk parity with very low volatility
    position = rp_sizer.calculate(volatility=0.01)
    assert 0 <= position <= rp_sizer.max_position


def test_all_sizers_consistent_return_type():
    """Test all sizers return float in [0, 1]."""
    config_fixed = PositionSizerConfig(strategy='fixed')
    config_vol = PositionSizerConfig(strategy='volatility')
    config_rp = PositionSizerConfig(strategy='risk_parity')

    fixed_sizer = create_position_sizer(config_fixed)
    vol_sizer = create_position_sizer(config_vol)
    rp_sizer = create_position_sizer(config_rp)

    # All should return float in [0, 1]
    assert isinstance(fixed_sizer.calculate(), float)
    assert isinstance(vol_sizer.calculate(price=2500, atr=50, portfolio_value=100000), float)
    assert isinstance(rp_sizer.calculate(volatility=0.20), float)

    # All should be in valid range
    assert 0 <= fixed_sizer.calculate() <= 1
    assert 0 <= vol_sizer.calculate(price=2500, atr=50, portfolio_value=100000) <= 1
    assert 0 <= rp_sizer.calculate(volatility=0.20) <= 1
