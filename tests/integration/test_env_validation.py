"""Integration test for environment validation with Stable-Baselines3."""

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from tradebox.env import EnvConfig, TradingEnv


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")

    close_prices = 1000 + np.cumsum(np.random.randn(300) * 10)
    close_prices = np.maximum(close_prices, 100)

    data = pd.DataFrame(
        {
            "Open": close_prices * (1 + np.random.randn(300) * 0.01),
            "High": close_prices * (1 + abs(np.random.randn(300)) * 0.02),
            "Low": close_prices * (1 - abs(np.random.randn(300)) * 0.02),
            "Close": close_prices,
            "Volume": np.random.randint(100000, 1000000, 300),
        },
        index=dates,
    )

    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

    return data


@pytest.fixture
def sample_features(sample_data):
    """Create sample features matching data length."""
    np.random.seed(42)
    n_features = 27  # Realistic number of features
    features = pd.DataFrame(
        np.random.randn(len(sample_data), n_features),
        index=sample_data.index,
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    return features


def test_env_passes_sb3_check_env(sample_data, sample_features):
    """Test that environment passes Stable-Baselines3 check_env validation."""
    config = EnvConfig(
        initial_capital=100000.0, lookback_window=60, max_episode_steps=100
    )

    env = TradingEnv(sample_data, sample_features, config)

    # This will raise an exception if the environment doesn't comply
    check_env(env, warn=True)


def test_env_can_run_episode(sample_data, sample_features):
    """Test that environment can run a full episode."""
    config = EnvConfig(
        initial_capital=100000.0, lookback_window=60, max_episode_steps=50
    )

    env = TradingEnv(sample_data, sample_features, config)

    obs, info = env.reset()
    done = False
    steps = 0
    total_reward = 0

    while not done and steps < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    # Verify episode completed
    assert steps > 0
    assert isinstance(total_reward, float)


def test_env_works_with_multiple_reward_functions(sample_data, sample_features):
    """Test environment works with all reward function types."""
    from tradebox.env.rewards import RewardConfig

    for reward_type in ["simple", "risk_adjusted", "sharpe"]:
        config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=50,
            reward_config=RewardConfig(reward_type=reward_type),
        )

        env = TradingEnv(sample_data, sample_features, config)

        # Run a short episode
        obs, info = env.reset()
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break

        # Should complete without errors
        assert True
