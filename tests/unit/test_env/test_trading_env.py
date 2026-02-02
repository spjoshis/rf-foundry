"""Unit tests for trading environment."""

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from tradebox.env.costs import CostConfig
from tradebox.env.rewards import RewardConfig
from tradebox.env.trading_env import EnvConfig, TradingEnv


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")

    # Generate realistic price movements
    close_prices = 1000 + np.cumsum(np.random.randn(200) * 10)
    close_prices = np.maximum(close_prices, 100)  # Ensure positive prices

    data = pd.DataFrame(
        {
            "Open": close_prices * (1 + np.random.randn(200) * 0.01),
            "High": close_prices * (1 + abs(np.random.randn(200)) * 0.02),
            "Low": close_prices * (1 - abs(np.random.randn(200)) * 0.02),
            "Close": close_prices,
            "Volume": np.random.randint(100000, 1000000, 200),
        },
        index=dates,
    )

    # Ensure OHLC consistency
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

    return data


@pytest.fixture
def sample_features(sample_data):
    """Create sample features matching data length."""
    np.random.seed(42)
    n_features = 5  # Small number for testing
    features = pd.DataFrame(
        np.random.randn(len(sample_data), n_features),
        index=sample_data.index,
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    return features


@pytest.fixture
def env_config():
    """Create environment config for testing."""
    return EnvConfig(
        initial_capital=100000.0,
        lookback_window=60,
        max_episode_steps=100,
        cost_config=CostConfig(),
        reward_config=RewardConfig(reward_type="simple"),
    )


@pytest.fixture
def trading_env(sample_data, sample_features, env_config):
    """Create trading environment for testing."""
    return TradingEnv(sample_data, sample_features, env_config)


def test_env_initialization(trading_env, sample_features):
    """Test environment initializes correctly."""
    assert isinstance(trading_env, TradingEnv)
    assert isinstance(trading_env.action_space, spaces.Discrete)
    assert trading_env.action_space.n == 3
    assert isinstance(trading_env.observation_space, spaces.Box)

    # Check observation space shape
    n_feature_cols = len(sample_features.columns)
    expected_obs_size = 60 * n_feature_cols + 4
    assert trading_env.observation_space.shape == (expected_obs_size,)


def test_env_initialization_invalid_data_length():
    """Test environment raises error with insufficient data."""
    short_data = pd.DataFrame({"Close": [100, 101, 102]})
    short_features = pd.DataFrame(np.random.randn(3, 5))
    config = EnvConfig()

    with pytest.raises(ValueError, match="must be at least"):
        TradingEnv(short_data, short_features, config)


def test_env_initialization_mismatched_lengths():
    """Test environment raises error when data and features don't match."""
    data = pd.DataFrame({"Close": np.random.randn(200)})
    features = pd.DataFrame(np.random.randn(100, 5))
    config = EnvConfig()

    with pytest.raises(ValueError, match="must have same length"):
        TradingEnv(data, features, config)


def test_reset(trading_env):
    """Test reset initializes episode correctly."""
    obs, info = trading_env.reset()

    # Check observation shape
    assert obs.shape == trading_env.observation_space.shape
    assert obs.dtype == np.float32

    # Check portfolio state reset
    assert trading_env.cash == trading_env.config.initial_capital
    assert trading_env.position == 0
    assert trading_env.entry_price == 0.0
    assert trading_env.total_trades == 0

    # Check info dict
    assert "step" in info
    assert "portfolio_value" in info
    assert info["portfolio_value"] == trading_env.config.initial_capital


def test_reset_with_seed(trading_env):
    """Test reset with seed gives reproducible results."""
    obs1, _ = trading_env.reset(seed=42)
    step1 = trading_env.current_step

    obs2, _ = trading_env.reset(seed=42)
    step2 = trading_env.current_step

    # Same seed should give same starting position
    assert step1 == step2
    assert np.allclose(obs1, obs2)


def test_step_hold_action(trading_env):
    """Test hold action doesn't change portfolio."""
    obs, info = trading_env.reset()
    initial_cash = trading_env.cash
    initial_position = trading_env.position

    # Execute hold action
    obs, reward, terminated, truncated, info = trading_env.step(0)

    # Cash and position should be unchanged
    assert trading_env.cash == initial_cash
    assert trading_env.position == initial_position
    assert trading_env.total_trades == 0


def test_step_buy_action(trading_env):
    """Test buy action creates position."""
    obs, info = trading_env.reset()
    initial_cash = trading_env.cash

    # Execute buy action
    obs, reward, terminated, truncated, info = trading_env.step(1)

    # Should have bought shares
    assert trading_env.position > 0
    assert trading_env.cash < initial_cash
    assert trading_env.entry_price > 0
    assert trading_env.total_trades == 1

    # Portfolio value could be slightly higher or lower due to price movement
    # but should be close to initial capital (within a reasonable range)
    assert abs(info["portfolio_value"] - initial_cash) < initial_cash * 0.05  # Within 5%


def test_step_sell_action_no_position(trading_env):
    """Test sell action with no position does nothing."""
    obs, info = trading_env.reset()
    initial_cash = trading_env.cash

    # Execute sell action with no position
    obs, reward, terminated, truncated, info = trading_env.step(2)

    # Nothing should change
    assert trading_env.cash == initial_cash
    assert trading_env.position == 0
    assert trading_env.total_trades == 0


def test_step_buy_then_sell(trading_env):
    """Test buy followed by sell closes position."""
    obs, info = trading_env.reset()
    initial_capital = trading_env.cash

    # Buy
    trading_env.step(1)
    assert trading_env.position > 0

    # Move forward a few steps to see price change
    for _ in range(5):
        trading_env.step(0)  # Hold

    # Sell
    obs, reward, terminated, truncated, info = trading_env.step(2)

    # Position should be closed
    assert trading_env.position == 0
    assert trading_env.entry_price == 0.0
    assert trading_env.total_trades == 2

    # All value should be in cash now
    assert info["portfolio_value"] == trading_env.cash


def test_step_multiple_buys(trading_env):
    """Test that can't buy multiple times."""
    obs, info = trading_env.reset()

    # First buy
    trading_env.step(1)
    position_after_first = trading_env.position

    # Second buy (should do nothing since already have position)
    trading_env.step(1)
    position_after_second = trading_env.position

    # Position should not change
    assert position_after_second == position_after_first
    assert trading_env.total_trades == 1  # Only one trade executed


def test_observation_shape(trading_env):
    """Test observation has correct shape."""
    obs, _ = trading_env.reset()

    # Check shape matches observation space
    assert obs.shape == trading_env.observation_space.shape

    # Check all values are finite
    assert np.all(np.isfinite(obs))


def test_portfolio_value_calculation(trading_env):
    """Test portfolio value is calculated correctly."""
    obs, info = trading_env.reset()

    # Initially, all cash
    assert trading_env._get_portfolio_value() == trading_env.cash

    # After buying
    trading_env.step(1)
    current_price = trading_env.data.iloc[trading_env.current_step]["Close"]
    expected_value = trading_env.cash + trading_env.position * current_price
    assert trading_env._get_portfolio_value() == pytest.approx(expected_value)


def test_unrealized_pnl_calculation(trading_env):
    """Test unrealized P&L calculation."""
    obs, info = trading_env.reset()

    # No position, no P&L
    assert trading_env._get_unrealized_pnl_pct() == 0.0

    # Buy shares
    trading_env.step(1)
    entry_price = trading_env.entry_price

    # Move forward and check P&L
    trading_env.step(0)
    current_price = trading_env.data.iloc[trading_env.current_step]["Close"]
    expected_pnl = (current_price - entry_price) / entry_price
    assert trading_env._get_unrealized_pnl_pct() == pytest.approx(expected_pnl)


def test_episode_termination(trading_env):
    """Test episode terminates when reaching end of data."""
    obs, info = trading_env.reset()

    # Set current step near end
    trading_env.current_step = len(trading_env.data) - 2

    # Next step should terminate
    obs, reward, terminated, truncated, info = trading_env.step(0)

    assert terminated is True


def test_episode_truncation(trading_env):
    """Test episode truncates after max_episode_steps."""
    obs, info = trading_env.reset()
    start_step = trading_env.current_step

    # Run for max_episode_steps
    for _ in range(trading_env.config.max_episode_steps):
        obs, reward, terminated, truncated, info = trading_env.step(0)

    # Should be truncated
    assert truncated == True
    assert trading_env.current_step - start_step == trading_env.config.max_episode_steps


def test_info_dict_contents(trading_env):
    """Test info dict contains all expected keys."""
    obs, info = trading_env.reset()
    obs, reward, terminated, truncated, info = trading_env.step(0)

    required_keys = [
        "step",
        "portfolio_value",
        "cash",
        "position",
        "total_trades",
        "entry_price",
        "unrealized_pnl_pct",
    ]

    for key in required_keys:
        assert key in info


def test_action_space_sample(trading_env):
    """Test action space sampling works."""
    for _ in range(10):
        action = trading_env.action_space.sample()
        assert action in [0, 1, 2]


def test_observation_space_contains(trading_env):
    """Test observations are within observation space."""
    obs, _ = trading_env.reset()
    assert trading_env.observation_space.contains(obs)

    for _ in range(10):
        obs, _, _, _, _ = trading_env.step(trading_env.action_space.sample())
        assert trading_env.observation_space.contains(obs)


def test_reward_is_float(trading_env):
    """Test reward is always a float."""
    obs, _ = trading_env.reset()

    for _ in range(10):
        obs, reward, _, _, _ = trading_env.step(trading_env.action_space.sample())
        assert isinstance(reward, float)
        assert np.isfinite(reward)


def test_env_with_different_reward_functions():
    """Test environment works with different reward functions."""
    from tradebox.env.rewards import RewardConfig

    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    data = pd.DataFrame(
        {"Close": 1000 + np.cumsum(np.random.randn(200) * 10)}, index=dates
    )
    features = pd.DataFrame(np.random.randn(200, 5), index=dates)

    for reward_type in ["simple", "risk_adjusted", "sharpe"]:
        config = EnvConfig(
            reward_config=RewardConfig(reward_type=reward_type),
            lookback_window=60,
            max_episode_steps=50,
        )
        env = TradingEnv(data, features, config)
        obs, info = env.reset()
        obs, reward, _, _, _ = env.step(1)
        assert isinstance(reward, float)


def test_cost_model_integration(trading_env):
    """Test that transaction costs are applied correctly."""
    obs, info = trading_env.reset()
    initial_cash = trading_env.cash

    # Buy shares
    trading_env.step(1)

    # Check that we bought shares and paid costs
    assert trading_env.position > 0
    assert trading_env.cash < initial_cash  # Cash was used

    # To verify transaction costs were applied, check that:
    # 1. The cash spent is more than just (shares * entry_price)
    entry_price = trading_env.entry_price
    shares = trading_env.position
    naive_cost = shares * entry_price

    actual_cost_paid = initial_cash - trading_env.cash

    # Actual cost should be higher than naive cost due to fees
    assert actual_cost_paid > naive_cost


def test_multiple_episodes(trading_env):
    """Test running multiple episodes."""
    for episode in range(3):
        obs, info = trading_env.reset()
        done = False
        steps = 0

        while not done and steps < 20:
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            done = terminated or truncated
            steps += 1

        # Each episode should reset properly
        assert trading_env.current_step >= trading_env.config.lookback_window


def test_observation_portfolio_state_components(trading_env):
    """Test that portfolio state components are in observation."""
    obs, _ = trading_env.reset()

    # Last 4 elements should be portfolio state
    portfolio_state = obs[-4:]

    # Initial state: no position, no P&L, all cash
    assert portfolio_state[0] == pytest.approx(0.0)  # position_size (0 shares)
    assert portfolio_state[1] == pytest.approx(0.0)  # price change from entry
    assert portfolio_state[2] == pytest.approx(0.0)  # unrealized P&L
    assert portfolio_state[3] == pytest.approx(1.0)  # cash % (100%)


def test_deterministic_reset(trading_env):
    """Test that reset without seed is random."""
    steps = []
    for _ in range(5):
        obs, info = trading_env.reset()
        steps.append(trading_env.current_step)

    # Should not all be the same (random starts)
    assert len(set(steps)) > 1


def test_env_handles_edge_case_low_cash():
    """Test environment when cash is too low to buy shares."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    data = pd.DataFrame({"Close": [10000.0] * 200}, index=dates)  # Very high price
    features = pd.DataFrame(np.random.randn(200, 5), index=dates)

    config = EnvConfig(initial_capital=1000.0, lookback_window=60, max_episode_steps=50)  # Low capital
    env = TradingEnv(data, features, config)

    obs, info = env.reset()
    obs, reward, _, _, info = env.step(1)  # Try to buy

    # Should not be able to buy (price too high relative to capital)
    assert env.position == 0
    assert env.cash == config.initial_capital
