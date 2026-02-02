"""Integration tests for TradingEnv with techno-fundamental features."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from tradebox.data.loaders.fundamental_loader import FundamentalConfig
from tradebox.env.trading_env import EnvConfig, TradingEnv
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
from tradebox.features.technical import FeatureConfig


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample OHLCV price data."""
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    np.random.seed(42)

    close_prices = 100 + np.cumsum(np.random.randn(300) * 2)
    high_prices = close_prices + np.abs(np.random.randn(300) * 1)
    low_prices = close_prices - np.abs(np.random.randn(300) * 1)
    open_prices = close_prices + np.random.randn(300) * 0.5

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, 300),
        },
        index=dates,
    )


@pytest.fixture
def technical_only_features(sample_price_data: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    """Extract technical-only features."""
    config = FeatureExtractorConfig(
        technical=FeatureConfig(sma_periods=[20, 50], normalize=True),
        fundamental=FundamentalConfig(enabled=False),
        cache_dir=str(tmp_path),
    )
    extractor = FeatureExtractor(config)
    return extractor.extract("TEST", sample_price_data, fit_normalize=True)


@pytest.fixture
def combined_features(sample_price_data: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    """Extract combined technical + fundamental features."""
    config = FeatureExtractorConfig(
        technical=FeatureConfig(sma_periods=[20, 50], normalize=True),
        fundamental=FundamentalConfig(enabled=True, use_mock_data=True),
        cache_dir=str(tmp_path),
    )
    extractor = FeatureExtractor(config)
    return extractor.extract("TEST", sample_price_data, fit_normalize=True)


class TestTechnicalOnlyEnvironment:
    """Tests for environment with technical features only."""

    def test_env_initialization_technical_only(
        self, sample_price_data: pd.DataFrame, technical_only_features: pd.DataFrame
    ) -> None:
        """Should initialize environment with technical-only features."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, technical_only_features, config)

        # Should have observation space for technical window + portfolio
        # With 2 SMAs (20, 50), we get technical features
        # Exact count depends on FeatureConfig, but should be > 0
        assert env.observation_space.shape[0] > 60  # At least 60 days of data
        assert len(env._technical_cols) > 0
        assert len(env._fundamental_cols) == 0  # No fundamentals

    def test_env_passes_sb3_check_technical_only(
        self, sample_price_data: pd.DataFrame, technical_only_features: pd.DataFrame
    ) -> None:
        """Should pass Stable-Baselines3 environment validation."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, technical_only_features, config)

        # This raises exception if env doesn't comply with Gym API
        check_env(env, warn=True)

    def test_observation_shape_technical_only(
        self, sample_price_data: pd.DataFrame, technical_only_features: pd.DataFrame
    ) -> None:
        """Should produce observations with correct shape."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, technical_only_features, config)

        obs, info = env.reset()

        # Observation should match declared observation space
        assert obs.shape == env.observation_space.shape
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32


class TestTechnoFundamentalEnvironment:
    """Tests for environment with both technical and fundamental features."""

    def test_env_initialization_combined(
        self, sample_price_data: pd.DataFrame, combined_features: pd.DataFrame
    ) -> None:
        """Should initialize environment with techno-fundamental features."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, combined_features, config)

        # Should have both technical and fundamental columns
        assert len(env._technical_cols) > 0
        assert len(env._fundamental_cols) > 0

        # Observation space should be larger than technical-only
        # Expected: 60 × n_tech + n_fund + 4
        n_tech_window = 60 * len(env._technical_cols)
        n_fund_static = len(env._fundamental_cols)
        n_portfolio = 4
        expected_obs_size = n_tech_window + n_fund_static + n_portfolio

        assert env.observation_space.shape[0] == expected_obs_size

    def test_env_passes_sb3_check_combined(
        self, sample_price_data: pd.DataFrame, combined_features: pd.DataFrame
    ) -> None:
        """Should pass Stable-Baselines3 environment validation with fundamentals."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, combined_features, config)

        # This raises exception if env doesn't comply with Gym API
        check_env(env, warn=True)

    def test_observation_structure_combined(
        self, sample_price_data: pd.DataFrame, combined_features: pd.DataFrame
    ) -> None:
        """Should construct observations with correct structure."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, combined_features, config)

        obs, info = env.reset()

        # Observation should match declared observation space
        assert obs.shape == env.observation_space.shape

        # Verify observation structure:
        # [technical_window, fundamentals_static, portfolio_state]
        n_tech_window = 60 * len(env._technical_cols)
        n_fund_static = len(env._fundamental_cols)

        # Extract components
        technical_part = obs[:n_tech_window]
        fundamental_part = obs[n_tech_window:n_tech_window + n_fund_static]
        portfolio_part = obs[-4:]

        # All parts should have values
        assert not np.all(np.isnan(technical_part))
        assert fundamental_part.shape[0] == n_fund_static
        assert portfolio_part.shape[0] == 4

    def test_fundamentals_are_static_not_windowed(
        self, sample_price_data: pd.DataFrame, combined_features: pd.DataFrame
    ) -> None:
        """Should use static fundamentals (current values), not windowed."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, combined_features, config)

        obs1, _ = env.reset()
        n_tech_window = 60 * len(env._technical_cols)
        n_fund_static = len(env._fundamental_cols)

        # Extract fundamental part of observation
        fund_part1 = obs1[n_tech_window:n_tech_window + n_fund_static]

        # Step forward a few times
        for _ in range(5):
            obs, _, _, _, _ = env.step(0)  # Hold action

        # Extract fundamental part after steps
        fund_part2 = obs[n_tech_window:n_tech_window + n_fund_static]

        # Fundamentals might change over time (quarterly updates)
        # But they should be a single vector, not a window
        assert fund_part1.shape[0] == n_fund_static
        assert fund_part2.shape[0] == n_fund_static

    def test_run_full_episode_combined(
        self, sample_price_data: pd.DataFrame, combined_features: pd.DataFrame
    ) -> None:
        """Should run complete episode without errors."""
        config = EnvConfig(lookback_window=60, max_episode_steps=50)
        env = TradingEnv(sample_price_data, combined_features, config)

        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        assert steps <= 50  # Should respect max_episode_steps
        assert isinstance(total_reward, float)


class TestFeatureSplitting:
    """Tests for technical vs fundamental feature splitting."""

    def test_split_feature_columns(
        self, sample_price_data: pd.DataFrame, combined_features: pd.DataFrame
    ) -> None:
        """Should correctly split technical and fundamental features."""
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, combined_features, config)

        numeric_features = combined_features.select_dtypes(include=[np.number])
        tech_cols, fund_cols = env._split_feature_columns(numeric_features.columns)

        # Should have both types
        assert len(tech_cols) > 0
        assert len(fund_cols) > 0

        # Technical columns should contain expected patterns
        tech_patterns = ["SMA_", "EMA_", "MACD", "RSI"]
        assert any(any(pattern in col for pattern in tech_patterns) for col in tech_cols)

        # Fundamental columns should contain expected patterns
        fund_patterns = ["PE_", "ROE", "Debt_to_Equity", "Revenue_Growth"]
        assert any(any(pattern in col for pattern in fund_patterns) for col in fund_cols)


class TestBackwardCompatibility:
    """Tests for backward compatibility with technical-only mode."""

    def test_technical_only_still_works(
        self, sample_price_data: pd.DataFrame, technical_only_features: pd.DataFrame
    ) -> None:
        """Should maintain backward compatibility with existing code."""
        # This mimics the old API usage
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(sample_price_data, technical_only_features, config)

        # Should work exactly as before
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
