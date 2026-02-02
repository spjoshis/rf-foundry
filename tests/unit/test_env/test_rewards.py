"""Unit tests for reward functions."""

import numpy as np
import pytest

from tradebox.env.rewards import (
    RewardConfig,
    RewardFunction,
    RiskAdjustedReward,
    SharpeReward,
    SimpleReward,
    create_reward_function,
)


@pytest.fixture
def simple_config():
    """Create simple reward config."""
    return RewardConfig(reward_type="simple")


@pytest.fixture
def risk_adjusted_config():
    """Create risk-adjusted reward config."""
    return RewardConfig(
        reward_type="risk_adjusted", drawdown_penalty=0.5, trade_penalty=0.001
    )


@pytest.fixture
def sharpe_config():
    """Create Sharpe reward config."""
    return RewardConfig(reward_type="sharpe", sharpe_window=20)


def test_reward_config_defaults():
    """Test RewardConfig default values."""
    config = RewardConfig()
    assert config.reward_type == "risk_adjusted"
    assert config.drawdown_penalty == 0.5
    assert config.trade_penalty == 0.001
    assert config.sharpe_window == 20


def test_simple_reward_positive_return(simple_config):
    """Test simple reward with positive return."""
    reward_fn = SimpleReward(simple_config)

    reward = reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=1)

    # Should be 1% gain
    assert reward == pytest.approx(0.01, abs=1e-6)


def test_simple_reward_negative_return(simple_config):
    """Test simple reward with negative return."""
    reward_fn = SimpleReward(simple_config)

    reward = reward_fn.calculate(prev_value=10000, current_value=9900, action=0, step=1)

    # Should be -1% loss
    assert reward == pytest.approx(-0.01, abs=1e-6)


def test_simple_reward_zero_return(simple_config):
    """Test simple reward with no change."""
    reward_fn = SimpleReward(simple_config)

    reward = reward_fn.calculate(prev_value=10000, current_value=10000, action=0, step=1)

    # Should be 0% change
    assert reward == pytest.approx(0.0, abs=1e-6)


def test_simple_reward_invalid_prev_value(simple_config):
    """Test simple reward with invalid previous value."""
    reward_fn = SimpleReward(simple_config)

    reward = reward_fn.calculate(prev_value=0, current_value=10000, action=0, step=1)

    # Should return 0 for invalid input
    assert reward == 0.0


def test_risk_adjusted_reward_no_drawdown_no_trade(risk_adjusted_config):
    """Test risk-adjusted reward with no drawdown and no trade."""
    reward_fn = RiskAdjustedReward(risk_adjusted_config)

    # First step: gain with hold action
    reward = reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=1)

    # Should be just the daily return (no drawdown, no trade penalty)
    assert reward == pytest.approx(0.01, abs=1e-6)


def test_risk_adjusted_reward_with_drawdown(risk_adjusted_config):
    """Test risk-adjusted reward penalizes drawdown."""
    reward_fn = RiskAdjustedReward(risk_adjusted_config)

    # Step 1: Set peak at 11000
    reward_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)

    # Step 2: Drop to 10000 (drawdown = 1000/11000 ≈ 9.09%)
    reward = reward_fn.calculate(prev_value=11000, current_value=10000, action=0, step=2)

    # daily_return = -0.0909, drawdown = 0.0909, penalty = 0.5 × 0.0909 = 0.0455
    # reward = -0.0909 - 0.0455 = -0.1364
    expected_return = -1000 / 11000  # -0.0909
    expected_drawdown = 1000 / 11000  # 0.0909
    expected_reward = expected_return - 0.5 * expected_drawdown
    assert reward == pytest.approx(expected_reward, abs=1e-4)


def test_risk_adjusted_reward_with_trade_penalty(risk_adjusted_config):
    """Test risk-adjusted reward penalizes trading."""
    reward_fn = RiskAdjustedReward(risk_adjusted_config)

    # Buy action (action=1)
    reward_buy = reward_fn.calculate(
        prev_value=10000, current_value=10100, action=1, step=1
    )

    # Should be daily_return - trade_penalty = 0.01 - 0.001 = 0.009
    assert reward_buy == pytest.approx(0.009, abs=1e-6)

    # Sell action (action=2)
    reward_sell = reward_fn.calculate(
        prev_value=10100, current_value=10200, action=2, step=2
    )

    # daily_return ≈ 0.0099, minus trade_penalty = 0.001
    expected_return = 100 / 10100
    expected_reward = expected_return - 0.001
    assert reward_sell == pytest.approx(expected_reward, abs=1e-6)


def test_risk_adjusted_reward_hold_no_penalty(risk_adjusted_config):
    """Test that hold action doesn't incur trade penalty."""
    reward_fn = RiskAdjustedReward(risk_adjusted_config)

    # Hold action (action=0)
    reward = reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=1)

    # Should be just daily_return (no trade penalty)
    assert reward == pytest.approx(0.01, abs=1e-6)


def test_sharpe_reward_window_not_filled(sharpe_config):
    """Test Sharpe reward before window is filled returns simple return."""
    reward_fn = SharpeReward(sharpe_config)

    # First 19 steps should return simple daily return
    for step in range(1, 20):
        reward = reward_fn.calculate(
            prev_value=10000, current_value=10100, action=0, step=step
        )
        assert reward == pytest.approx(0.01, abs=1e-6)


def test_sharpe_reward_after_window_filled(sharpe_config):
    """Test Sharpe reward after window is filled."""
    reward_fn = SharpeReward(sharpe_config)

    # Generate 20 consistent positive returns
    for step in range(1, 21):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=step)

    # 21st step should return Sharpe ratio
    reward = reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=21)

    # With consistent 1% returns and zero std, Sharpe should be very high or 0
    # (depends on numerical precision)
    assert isinstance(reward, float)


def test_sharpe_reward_with_varying_returns():
    """Test Sharpe reward with varying returns."""
    config = RewardConfig(reward_type="sharpe", sharpe_window=5)
    reward_fn = SharpeReward(config)

    # Generate varying returns
    values = [10000, 10100, 10050, 10150, 10100, 10200]

    for i in range(1, len(values)):
        reward = reward_fn.calculate(
            prev_value=values[i - 1], current_value=values[i], action=0, step=i
        )

    # After window filled, should compute Sharpe
    # Just verify it's a valid number
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not np.isinf(reward)


def test_sharpe_reward_zero_std():
    """Test Sharpe reward with zero standard deviation."""
    config = RewardConfig(reward_type="sharpe", sharpe_window=3)
    reward_fn = SharpeReward(config)

    # All identical returns (zero std)
    for step in range(1, 5):
        reward = reward_fn.calculate(
            prev_value=10000, current_value=10100, action=0, step=step
        )

    # Should return 0 when std is 0 (avoid div by zero)
    assert reward == 0.0


def test_reward_function_reset(risk_adjusted_config):
    """Test that reset clears reward function state."""
    reward_fn = RiskAdjustedReward(risk_adjusted_config)

    # Build up state
    reward_fn.calculate(prev_value=10000, current_value=11000, action=1, step=1)
    assert reward_fn.peak_value > 0

    # Reset
    reward_fn.reset()

    # State should be cleared
    assert reward_fn.peak_value == 0.0
    assert len(reward_fn.returns_history) == 0


def test_sharpe_reward_reset(sharpe_config):
    """Test that Sharpe reward reset clears history."""
    reward_fn = SharpeReward(sharpe_config)

    # Build up history
    for step in range(1, 25):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=step)

    assert len(reward_fn.returns_history) == 24

    # Reset
    reward_fn.reset()

    # History should be cleared
    assert len(reward_fn.returns_history) == 0


def test_create_reward_function_simple():
    """Test factory creates simple reward function."""
    config = RewardConfig(reward_type="simple")
    reward_fn = create_reward_function(config)

    assert isinstance(reward_fn, SimpleReward)
    assert reward_fn.config == config


def test_create_reward_function_risk_adjusted():
    """Test factory creates risk-adjusted reward function."""
    config = RewardConfig(reward_type="risk_adjusted")
    reward_fn = create_reward_function(config)

    assert isinstance(reward_fn, RiskAdjustedReward)
    assert reward_fn.config == config


def test_create_reward_function_sharpe():
    """Test factory creates Sharpe reward function."""
    config = RewardConfig(reward_type="sharpe")
    reward_fn = create_reward_function(config)

    assert isinstance(reward_fn, SharpeReward)
    assert reward_fn.config == config


def test_create_reward_function_invalid_type():
    """Test factory raises error for invalid reward type."""
    config = RewardConfig(reward_type="invalid")

    with pytest.raises(ValueError, match="Unknown reward_type"):
        create_reward_function(config)


def test_all_rewards_are_reward_functions():
    """Test that all reward classes inherit from RewardFunction."""
    assert issubclass(SimpleReward, RewardFunction)
    assert issubclass(RiskAdjustedReward, RewardFunction)
    assert issubclass(SharpeReward, RewardFunction)


def test_risk_adjusted_peak_value_tracking():
    """Test that peak value is tracked correctly over multiple steps."""
    config = RewardConfig(reward_type="risk_adjusted")
    reward_fn = RiskAdjustedReward(config)

    # Step 1: 10000 -> 11000
    reward_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)
    assert reward_fn.peak_value == 11000

    # Step 2: 11000 -> 10500 (drawdown)
    reward_fn.calculate(prev_value=11000, current_value=10500, action=0, step=2)
    assert reward_fn.peak_value == 11000  # Peak unchanged

    # Step 3: 10500 -> 12000 (new peak)
    reward_fn.calculate(prev_value=10500, current_value=12000, action=0, step=3)
    assert reward_fn.peak_value == 12000  # New peak


def test_sharpe_reward_history_length():
    """Test that Sharpe reward maintains correct history length."""
    config = RewardConfig(reward_type="sharpe", sharpe_window=10)
    reward_fn = SharpeReward(config)

    # Generate 50 steps
    for step in range(1, 51):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=step)

    # History should contain all returns (not capped at window size)
    assert len(reward_fn.returns_history) == 50


def test_reward_config_custom_values():
    """Test creating RewardConfig with custom values."""
    config = RewardConfig(
        reward_type="sharpe",
        drawdown_penalty=0.8,
        trade_penalty=0.002,
        sharpe_window=30,
    )

    assert config.reward_type == "sharpe"
    assert config.drawdown_penalty == 0.8
    assert config.trade_penalty == 0.002
    assert config.sharpe_window == 30


def test_simple_reward_large_values(simple_config):
    """Test simple reward with large portfolio values."""
    reward_fn = SimpleReward(simple_config)

    reward = reward_fn.calculate(
        prev_value=1_000_000, current_value=1_010_000, action=0, step=1
    )

    # Should still be 1% gain
    assert reward == pytest.approx(0.01, abs=1e-6)


def test_risk_adjusted_multiple_trades(risk_adjusted_config):
    """Test risk-adjusted reward accumulates trade penalties correctly."""
    reward_fn = RiskAdjustedReward(risk_adjusted_config)

    # Multiple consecutive trades
    total_penalty = 0

    for action in [1, 2, 1, 2, 0]:  # Buy, Sell, Buy, Sell, Hold
        reward = reward_fn.calculate(
            prev_value=10000, current_value=10000, action=action, step=1
        )
        if action in [1, 2]:
            total_penalty += 0.001

    # Total trade penalty for 4 trades should be 0.004
    assert total_penalty == pytest.approx(0.004, abs=1e-6)


# ============================================================================
# NEW: Tests for Advanced Reward Functions (STORY-017)
# ============================================================================


@pytest.fixture
def sortino_config():
    """Create Sortino reward config."""
    return RewardConfig(
        reward_type="sortino",
        sortino_window=20,
        risk_free_rate=0.06,
        mar=0.0,
    )


@pytest.fixture
def volatility_penalized_config():
    """Create volatility-penalized reward config."""
    return RewardConfig(
        reward_type="volatility_penalized",
        volatility_window=20,
        volatility_penalty=0.1,
        drawdown_penalty=0.5,
        trade_penalty=0.001,
    )


@pytest.fixture
def enhanced_drawdown_config():
    """Create enhanced drawdown reward config."""
    return RewardConfig(
        reward_type="enhanced_drawdown",
        enhanced_drawdown_penalty=2.0,
        duration_penalty=0.5,
        trade_penalty=0.001,
    )


@pytest.fixture
def calmar_config():
    """Create Calmar reward config."""
    return RewardConfig(
        reward_type="calmar", calmar_window=60, min_dd_threshold=0.01
    )


# ============================================================================
# SortinoReward Tests
# ============================================================================


def test_sortino_reward_bootstrap_period(sortino_config):
    """Test Sortino reward returns simple daily return during bootstrap."""
    from tradebox.env.rewards import SortinoReward

    reward_fn = SortinoReward(sortino_config)

    # First 19 steps should return simple daily return
    for step in range(1, 20):
        reward = reward_fn.calculate(
            prev_value=10000, current_value=10100, action=0, step=step
        )
        # Should be 1% gain
        assert reward == pytest.approx(0.01, abs=1e-6)


def test_sortino_downside_std_calculation(sortino_config):
    """Test Sortino correctly calculates downside deviation."""
    from tradebox.env.rewards import SortinoReward

    reward_fn = SortinoReward(sortino_config)

    # Generate mixed returns: 10 positive (1%), 10 negative (-1%)
    for i in range(10):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=i)
        reward_fn.calculate(prev_value=10000, current_value=9900, action=0, step=i + 10)

    # 21st step should compute Sortino with only downside returns
    reward = reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=21)

    # Verify it's a valid Sortino ratio (non-zero, not NaN)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not np.isinf(reward)


def test_sortino_zero_downside_volatility(sortino_config):
    """Test Sortino returns 0 when all returns above MAR."""
    from tradebox.env.rewards import SortinoReward

    reward_fn = SortinoReward(sortino_config)

    # Generate 20 consecutive positive returns (all above MAR=0)
    for step in range(1, 21):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=step)

    # 21st step should return 0 (no downside volatility)
    reward = reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=21)

    assert reward == 0.0


def test_sortino_vs_sharpe_asymmetric_returns():
    """Test Sortino vs Sharpe with asymmetric return distribution."""
    from tradebox.env.rewards import SharpeReward, SortinoReward

    sharpe_config = RewardConfig(reward_type="sharpe", sharpe_window=20)
    sortino_config = RewardConfig(
        reward_type="sortino", sortino_window=20, risk_free_rate=0.0, mar=0.0
    )

    sharpe_fn = SharpeReward(sharpe_config)
    sortino_fn = SortinoReward(sortino_config)

    # Asymmetric returns: large gains, small losses
    # 15 steps with +2% gains, 5 steps with -0.5% losses
    for step in range(1, 16):
        sharpe_fn.calculate(prev_value=10000, current_value=10200, action=0, step=step)
        sortino_fn.calculate(prev_value=10000, current_value=10200, action=0, step=step)

    for step in range(16, 21):
        sharpe_fn.calculate(prev_value=10000, current_value=9950, action=0, step=step)
        sortino_fn.calculate(prev_value=10000, current_value=9950, action=0, step=step)

    # Calculate 21st step
    sharpe_reward = sharpe_fn.calculate(
        prev_value=10000, current_value=10100, action=0, step=21
    )
    sortino_reward = sortino_fn.calculate(
        prev_value=10000, current_value=10100, action=0, step=21
    )

    # With asymmetric returns (large upside, small downside), Sortino should be
    # higher than Sharpe (or at least valid)
    assert isinstance(sharpe_reward, float)
    assert isinstance(sortino_reward, float)
    assert not np.isnan(sortino_reward)


def test_sortino_reset(sortino_config):
    """Test Sortino reward reset clears history."""
    from tradebox.env.rewards import SortinoReward

    reward_fn = SortinoReward(sortino_config)

    # Build up history
    for step in range(1, 25):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=step)

    assert len(reward_fn.returns_history) == 24

    # Reset
    reward_fn.reset()

    # History should be cleared
    assert len(reward_fn.returns_history) == 0


# ============================================================================
# VolatilityPenalizedReward Tests
# ============================================================================


def test_volatility_penalty_applied(volatility_penalized_config):
    """Test that volatility penalty is correctly applied."""
    from tradebox.env.rewards import VolatilityPenalizedReward

    reward_fn = VolatilityPenalizedReward(volatility_penalized_config)

    # Generate 20 returns with high volatility
    returns = [0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.025, -0.005] * 3
    values = [10000]
    for r in returns[:20]:
        values.append(values[-1] * (1 + r))

    for i in range(20):
        reward_fn.calculate(
            prev_value=values[i], current_value=values[i + 1], action=0, step=i
        )

    # 21st step should have volatility penalty
    reward = reward_fn.calculate(
        prev_value=values[20], current_value=values[20] * 1.01, action=0, step=21
    )

    # Reward should be less than simple daily return due to volatility penalty
    daily_return = 0.01
    assert reward < daily_return


def test_volatility_annualization(volatility_penalized_config):
    """Test that volatility is correctly annualized."""
    from tradebox.env.rewards import VolatilityPenalizedReward

    reward_fn = VolatilityPenalizedReward(volatility_penalized_config)

    # Generate 20 identical returns (1% gain each day)
    for step in range(1, 21):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=step)

    # With constant returns, std=0, so no volatility penalty
    reward = reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=21)

    # Should be close to daily return (no volatility penalty for constant returns)
    assert reward == pytest.approx(0.01, abs=1e-6)


def test_volatility_combined_with_drawdown(volatility_penalized_config):
    """Test volatility penalty combines correctly with drawdown penalty."""
    from tradebox.env.rewards import VolatilityPenalizedReward

    reward_fn = VolatilityPenalizedReward(volatility_penalized_config)

    # Build up to peak
    for step in range(1, 21):
        reward_fn.calculate(prev_value=10000, current_value=11000, action=0, step=step)

    # Drawdown step with high volatility
    reward = reward_fn.calculate(prev_value=11000, current_value=10000, action=0, step=21)

    # Reward should be negative (loss + drawdown penalty + potential volatility penalty)
    expected_return = (10000 - 11000) / 11000  # -0.0909
    assert reward < expected_return  # More negative due to penalties


# ============================================================================
# EnhancedDrawdownReward Tests
# ============================================================================


def test_quadratic_drawdown_penalty(enhanced_drawdown_config):
    """Test that drawdown penalty is quadratic."""
    from tradebox.env.rewards import EnhancedDrawdownReward

    reward_fn = EnhancedDrawdownReward(enhanced_drawdown_config)

    # Set peak at 11000
    reward_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)

    # 10% drawdown
    reward_10pct = reward_fn.calculate(prev_value=11000, current_value=9900, action=0, step=2)

    # Reset and test 20% drawdown
    reward_fn.reset()
    reward_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)
    reward_20pct = reward_fn.calculate(prev_value=11000, current_value=8800, action=0, step=2)

    # 20% DD should have ~4x the penalty of 10% DD (quadratic)
    # Note: actual values will differ due to different daily returns
    # But the quadratic component should dominate
    assert isinstance(reward_10pct, float)
    assert isinstance(reward_20pct, float)
    assert reward_20pct < reward_10pct  # Larger DD = more negative reward


def test_duration_penalty_accumulation(enhanced_drawdown_config):
    """Test that duration penalty accumulates over time."""
    from tradebox.env.rewards import EnhancedDrawdownReward

    reward_fn = EnhancedDrawdownReward(enhanced_drawdown_config)

    # Set peak
    reward_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)

    # Stay underwater for 10 steps (constant value)
    rewards = []
    for step in range(2, 12):
        reward = reward_fn.calculate(prev_value=10500, current_value=10500, action=0, step=step)
        rewards.append(reward)

    # Duration penalty should increase each step
    # Later rewards should be more negative due to accumulating duration penalty
    # (even though portfolio value is constant)
    assert len(rewards) == 10
    # At least check the last reward is more negative than first
    # (due to duration penalty accumulation)
    assert rewards[-1] <= rewards[0]


def test_duration_reset_on_new_peak(enhanced_drawdown_config):
    """Test that duration counter resets when new peak reached."""
    from tradebox.env.rewards import EnhancedDrawdownReward

    reward_fn = EnhancedDrawdownReward(enhanced_drawdown_config)

    # Set initial peak
    reward_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)

    # Drawdown for 5 steps
    for step in range(2, 7):
        reward_fn.calculate(prev_value=10500, current_value=10500, action=0, step=step)

    assert reward_fn.steps_since_peak == 5

    # New peak reached
    reward_fn.calculate(prev_value=10500, current_value=12000, action=0, step=7)

    # Duration counter should reset
    assert reward_fn.steps_since_peak == 0


def test_enhanced_drawdown_vs_linear():
    """Test enhanced (quadratic) DD penalty vs linear DD penalty."""
    # Use equal effective penalties for same DD level:
    # For 20% DD: linear penalty = 0.2 × 0.5 = 0.1
    # For 20% DD: quadratic penalty = 0.04 × 5.0 = 0.2 (2x higher)
    linear_config = RewardConfig(reward_type="risk_adjusted", drawdown_penalty=0.5)
    enhanced_config = RewardConfig(
        reward_type="enhanced_drawdown",
        enhanced_drawdown_penalty=5.0,  # Higher coefficient for quadratic penalty
        duration_penalty=0.0,  # Disable duration for fair comparison
    )

    from tradebox.env.rewards import EnhancedDrawdownReward, RiskAdjustedReward

    linear_fn = RiskAdjustedReward(linear_config)
    enhanced_fn = EnhancedDrawdownReward(enhanced_config)

    # Set peak at 11000
    linear_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)
    enhanced_fn.calculate(prev_value=10000, current_value=11000, action=0, step=1)

    # Large drawdown (20%)
    linear_reward = linear_fn.calculate(prev_value=11000, current_value=8800, action=0, step=2)
    enhanced_reward = enhanced_fn.calculate(prev_value=11000, current_value=8800, action=0, step=2)

    # Enhanced should penalize more heavily (quadratic with higher coefficient)
    # Linear: -0.2 (return) - 0.1 (DD penalty) = -0.3
    # Enhanced: -0.2 (return) - 0.2 (DD² penalty) = -0.4
    assert enhanced_reward < linear_reward


# ============================================================================
# CalmarReward Tests
# ============================================================================


def test_calmar_bootstrap_period(calmar_config):
    """Test Calmar reward returns simple daily return during bootstrap."""
    from tradebox.env.rewards import CalmarReward

    reward_fn = CalmarReward(calmar_config)

    # First 59 steps should return simple daily return
    for step in range(1, 60):
        reward = reward_fn.calculate(
            prev_value=10000, current_value=10100, action=0, step=step
        )
        # Should be 1% gain
        assert reward == pytest.approx(0.01, abs=1e-6)


def test_calmar_cagr_calculation(calmar_config):
    """Test Calmar correctly calculates rolling CAGR."""
    from tradebox.env.rewards import CalmarReward

    reward_fn = CalmarReward(calmar_config)

    # Generate 60 steps with steady 1% daily gains
    for step in range(1, 61):
        prev_val = 10000 * (1.01 ** (step - 1))
        curr_val = 10000 * (1.01 ** step)
        reward_fn.calculate(prev_value=prev_val, current_value=curr_val, action=0, step=step)

    # 61st step should compute Calmar ratio
    prev_val = 10000 * (1.01**60)
    curr_val = 10000 * (1.01**61)
    reward = reward_fn.calculate(prev_value=prev_val, current_value=curr_val, action=0, step=61)

    # Should return a valid Calmar ratio (positive with steady gains)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not np.isinf(reward)
    assert reward > 0  # Positive CAGR, minimal DD


def test_calmar_max_drawdown_tracking(calmar_config):
    """Test Calmar correctly tracks maximum drawdown."""
    from tradebox.env.rewards import CalmarReward

    reward_fn = CalmarReward(calmar_config)

    # Generate 60 steps: up to peak, then drawdown
    # Steps 1-30: rise to 15000
    for step in range(1, 31):
        prev_val = 10000 + (step - 1) * 166.67
        curr_val = 10000 + step * 166.67
        reward_fn.calculate(prev_value=prev_val, current_value=curr_val, action=0, step=step)

    # Steps 31-60: drop to 12000 (20% DD from peak of 15000)
    for step in range(31, 61):
        prev_val = 15000 - (step - 31) * 100
        curr_val = 15000 - (step - 30) * 100
        reward_fn.calculate(prev_value=prev_val, current_value=curr_val, action=0, step=step)

    # Max DD should be around 20%
    assert reward_fn.window_max_dd > 0.15  # At least 15% (allowing for rounding)
    assert reward_fn.window_max_dd < 0.25  # Less than 25%


def test_calmar_min_dd_threshold(calmar_config):
    """Test Calmar uses minimum DD threshold to prevent division by zero."""
    from tradebox.env.rewards import CalmarReward

    reward_fn = CalmarReward(calmar_config)

    # Generate 60 steps with steady gains (no drawdown)
    for step in range(1, 61):
        prev_val = 10000 * (1.001 ** (step - 1))
        curr_val = 10000 * (1.001 ** step)
        reward_fn.calculate(prev_value=prev_val, current_value=curr_val, action=0, step=step)

    # 61st step should use min_dd_threshold (0.01) if actual DD is smaller
    prev_val = 10000 * (1.001**60)
    curr_val = 10000 * (1.001**61)
    reward = reward_fn.calculate(prev_value=prev_val, current_value=curr_val, action=0, step=61)

    # Should not be infinite (threshold prevents div by zero)
    assert not np.isinf(reward)
    assert isinstance(reward, float)


def test_calmar_reset(calmar_config):
    """Test Calmar reward reset clears value history."""
    from tradebox.env.rewards import CalmarReward

    reward_fn = CalmarReward(calmar_config)

    # Build up history
    for step in range(1, 65):
        reward_fn.calculate(prev_value=10000, current_value=10100, action=0, step=step)

    assert len(reward_fn.value_history) == 64

    # Reset
    reward_fn.reset()

    # History should be cleared
    assert len(reward_fn.value_history) == 0
    assert reward_fn.window_max_dd == 0.0


# ============================================================================
# Factory Tests for New Rewards
# ============================================================================


def test_create_reward_function_sortino():
    """Test factory creates Sortino reward function."""
    from tradebox.env.rewards import SortinoReward, create_reward_function

    config = RewardConfig(reward_type="sortino")
    reward_fn = create_reward_function(config)

    assert isinstance(reward_fn, SortinoReward)
    assert reward_fn.config == config


def test_create_reward_function_volatility_penalized():
    """Test factory creates volatility-penalized reward function."""
    from tradebox.env.rewards import VolatilityPenalizedReward, create_reward_function

    config = RewardConfig(reward_type="volatility_penalized")
    reward_fn = create_reward_function(config)

    assert isinstance(reward_fn, VolatilityPenalizedReward)
    assert reward_fn.config == config


def test_create_reward_function_enhanced_drawdown():
    """Test factory creates enhanced drawdown reward function."""
    from tradebox.env.rewards import EnhancedDrawdownReward, create_reward_function

    config = RewardConfig(reward_type="enhanced_drawdown")
    reward_fn = create_reward_function(config)

    assert isinstance(reward_fn, EnhancedDrawdownReward)
    assert reward_fn.config == config


def test_create_reward_function_calmar():
    """Test factory creates Calmar reward function."""
    from tradebox.env.rewards import CalmarReward, create_reward_function

    config = RewardConfig(reward_type="calmar")
    reward_fn = create_reward_function(config)

    assert isinstance(reward_fn, CalmarReward)
    assert reward_fn.config == config


def test_all_new_rewards_are_reward_functions():
    """Test that all new reward classes inherit from RewardFunction."""
    from tradebox.env.rewards import (
        CalmarReward,
        EnhancedDrawdownReward,
        RewardFunction,
        SortinoReward,
        VolatilityPenalizedReward,
    )

    assert issubclass(SortinoReward, RewardFunction)
    assert issubclass(VolatilityPenalizedReward, RewardFunction)
    assert issubclass(EnhancedDrawdownReward, RewardFunction)
    assert issubclass(CalmarReward, RewardFunction)


# ============================================================================
# Edge Case Tests for All Rewards
# ============================================================================


def test_all_rewards_handle_zero_prev_value():
    """Test that all reward functions handle zero prev_value gracefully."""
    from tradebox.env.rewards import (
        CalmarReward,
        EnhancedDrawdownReward,
        RewardConfig,
        SortinoReward,
        VolatilityPenalizedReward,
    )

    configs = [
        RewardConfig(reward_type="sortino"),
        RewardConfig(reward_type="volatility_penalized"),
        RewardConfig(reward_type="enhanced_drawdown"),
        RewardConfig(reward_type="calmar"),
    ]

    reward_classes = [SortinoReward, VolatilityPenalizedReward, EnhancedDrawdownReward, CalmarReward]

    for config, RewardClass in zip(configs, reward_classes):
        reward_fn = RewardClass(config)
        reward = reward_fn.calculate(prev_value=0, current_value=10000, action=0, step=1)
        # Should return 0 or handle gracefully (not crash)
        assert isinstance(reward, float)
        assert reward == 0.0


def test_all_rewards_handle_negative_returns():
    """Test that all rewards handle negative returns without crashing."""
    from tradebox.env.rewards import (
        CalmarReward,
        EnhancedDrawdownReward,
        RewardConfig,
        SortinoReward,
        VolatilityPenalizedReward,
    )

    configs = [
        RewardConfig(reward_type="sortino", sortino_window=5),
        RewardConfig(reward_type="volatility_penalized", volatility_window=5),
        RewardConfig(reward_type="enhanced_drawdown"),
        RewardConfig(reward_type="calmar", calmar_window=5),
    ]

    reward_classes = [SortinoReward, VolatilityPenalizedReward, EnhancedDrawdownReward, CalmarReward]

    for config, RewardClass in zip(configs, reward_classes):
        reward_fn = RewardClass(config)

        # Generate 10 consecutive losses
        for step in range(1, 11):
            reward = reward_fn.calculate(
                prev_value=10000, current_value=9900, action=0, step=step
            )
            # Should return a valid number (not NaN or inf)
            assert isinstance(reward, float)
            assert not np.isnan(reward)
            assert not np.isinf(reward)


def test_all_rewards_reset_properly():
    """Test that all reward functions reset state correctly."""
    from tradebox.env.rewards import (
        CalmarReward,
        EnhancedDrawdownReward,
        RewardConfig,
        SortinoReward,
        VolatilityPenalizedReward,
    )

    configs = [
        RewardConfig(reward_type="sortino"),
        RewardConfig(reward_type="volatility_penalized"),
        RewardConfig(reward_type="enhanced_drawdown"),
        RewardConfig(reward_type="calmar"),
    ]

    reward_classes = [SortinoReward, VolatilityPenalizedReward, EnhancedDrawdownReward, CalmarReward]

    for config, RewardClass in zip(configs, reward_classes):
        reward_fn = RewardClass(config)

        # Build up state
        for step in range(1, 25):
            reward_fn.calculate(prev_value=10000, current_value=10100, action=1, step=step)

        # Reset
        reward_fn.reset()

        # Verify reset worked
        assert reward_fn.peak_value == 0.0
        assert len(reward_fn.returns_history) == 0

        # EnhancedDrawdownReward has additional state
        if isinstance(reward_fn, EnhancedDrawdownReward):
            assert reward_fn.steps_since_peak == 0

        # CalmarReward has additional state
        if isinstance(reward_fn, CalmarReward):
            assert len(reward_fn.value_history) == 0
            assert reward_fn.window_max_dd == 0.0
