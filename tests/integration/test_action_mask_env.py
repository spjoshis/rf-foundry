"""Integration tests for action masking in TradingEnv."""

import numpy as np
import pandas as pd
import pytest

from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.env.action_mask import ActionMaskConfig
from tradebox.env.trading_env import EnvConfig, TradingEnv
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
from tradebox.features.regime import RegimeConfig, RegimeDetector


@pytest.fixture
def mock_ohlcv_data():
    """Create mock OHLCV data for testing."""
    np.random.seed(42)
    n_days = 200

    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    close_prices = 100 + np.cumsum(np.random.randn(n_days) * 2)

    data = pd.DataFrame({
        "Open": close_prices * 0.99,
        "High": close_prices * 1.01,
        "Low": close_prices * 0.98,
        "Close": close_prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, n_days),
    }, index=dates)

    return data


@pytest.fixture
def features_with_regime(mock_ohlcv_data):
    """Create features with regime detection enabled."""
    # Create simple technical features
    features = pd.DataFrame({
        "SMA_20": mock_ohlcv_data["Close"].rolling(20).mean(),
        "RSI": 50.0,  # Simplified
        # Add ADX and DI for regime detection
        "ADX": np.random.uniform(10, 40, len(mock_ohlcv_data)),
        "Plus_DI": np.random.uniform(10, 30, len(mock_ohlcv_data)),
        "Minus_DI": np.random.uniform(10, 30, len(mock_ohlcv_data)),
    }, index=mock_ohlcv_data.index)

    # Add regime features using RegimeDetector
    regime_config = RegimeConfig(
        trending_threshold=25.0,
        ranging_threshold=20.0,
        use_directional_bias=True,
    )
    detector = RegimeDetector(regime_config)
    regime_df = detector.detect(features)

    # Combine all features
    all_features = pd.concat([features, regime_df], axis=1)

    # Fill NaN values
    all_features = all_features.fillna(method="bfill").fillna(0.0)

    return all_features


class TestActionMaskBackwardCompatibility:
    """Test backward compatibility with action masking disabled."""

    def test_env_without_action_mask_config(self, mock_ohlcv_data, features_with_regime):
        """Test env works without action_mask_config (backward compatible)."""
        # Create env config WITHOUT action_mask_config
        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=50,
        )

        # Should not raise error (default action_mask_config created)
        env = TradingEnv(mock_ohlcv_data, features_with_regime, env_config)

        # Verify action masking is disabled by default
        assert env.config.action_mask_config.enabled is False

        # Verify env works normally
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "price" in obs
        assert "portfolio" in obs

        # Verify no action_mask in info when disabled
        assert "action_mask" not in info

    def test_env_with_action_mask_disabled(self, mock_ohlcv_data, features_with_regime):
        """Test env with action mask explicitly disabled."""
        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=50,
            action_mask_config=ActionMaskConfig(enabled=False),
        )

        env = TradingEnv(mock_ohlcv_data, features_with_regime, env_config)

        # Reset and step
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        # No action mask in info
        assert "action_mask" not in info

        # All actions should work
        for action in [0, 1, 2]:
            obs, reward, terminated, truncated, info = env.step(action)
            # Should not raise error


class TestActionMaskIntegrationEnabled:
    """Test action masking when enabled."""

    def test_env_with_action_mask_enabled(self, mock_ohlcv_data, features_with_regime):
        """Test env with action mask enabled."""
        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=50,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_with_regime, env_config)

        # Reset
        obs, info = env.reset()

        # Action mask should be in info
        assert "action_mask" in info
        assert isinstance(info["action_mask"], np.ndarray)
        assert info["action_mask"].shape == (3,)
        assert info["action_mask"].dtype == bool

    def test_action_mask_changes_with_regime(self, mock_ohlcv_data, features_with_regime):
        """Test that action mask changes based on regime."""
        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=50,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_with_regime, env_config)

        # Collect masks across multiple steps
        obs, info = env.reset()
        masks = [info["action_mask"].copy()]

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            if "action_mask" in info:
                masks.append(info["action_mask"].copy())
            if terminated or truncated:
                break

        # Verify we got multiple masks
        assert len(masks) > 1

        # Verify masks are valid (at least Hold should always be allowed)
        for mask in masks:
            assert mask[0] == True  # Hold always allowed

    def test_regime_info_extraction(self, mock_ohlcv_data, features_with_regime):
        """Test that regime info is correctly extracted from features."""
        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=50,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_with_regime, env_config)

        # Reset and extract regime info
        obs, info = env.reset()

        # Should not raise error
        regime_state, trend_bias = env._get_regime_info()

        # Verify regime state is valid
        assert regime_state in {0, 1, 2}

        # Verify trend bias is valid
        assert trend_bias in {-1, 0, 1}

    def test_action_mask_missing_columns_raises_error(self, mock_ohlcv_data):
        """Test that missing regime columns raises helpful error."""
        # Create features WITHOUT regime columns
        features = pd.DataFrame({
            "SMA_20": mock_ohlcv_data["Close"].rolling(20).mean(),
            "RSI": 50.0,
        }, index=mock_ohlcv_data.index).fillna(0.0)

        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=50,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features, env_config)

        # Should raise error when trying to get regime info
        with pytest.raises(ValueError, match="Regime column.*not found"):
            env.reset()


class TestActionMaskRealisticScenario:
    """Test action masking in realistic trading scenarios."""

    def test_ranging_market_restricts_trading(self, mock_ohlcv_data, features_with_regime):
        """Test that ranging markets restrict trading to Hold only."""
        # Force all timesteps to ranging regime
        features_ranging = features_with_regime.copy()
        features_ranging["regime_state"] = 0  # Ranging
        features_ranging["trend_bias"] = 0    # Neutral

        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=10,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_ranging, env_config)

        obs, info = env.reset()

        # In ranging market, only Hold should be allowed
        expected_mask = np.array([True, False, False])
        np.testing.assert_array_equal(info["action_mask"], expected_mask)

    def test_trending_upmarket_allows_buy(self, mock_ohlcv_data, features_with_regime):
        """Test that trending upmarket allows Buy but not Sell."""
        # Force all timesteps to trending upmarket
        features_trending_up = features_with_regime.copy()
        features_trending_up["regime_state"] = 2   # Trending
        features_trending_up["trend_bias"] = 1     # Uptrend

        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=10,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_trending_up, env_config)

        obs, info = env.reset()

        # In uptrend, Hold and Buy should be allowed
        expected_mask = np.array([True, True, False])
        np.testing.assert_array_equal(info["action_mask"], expected_mask)

    def test_trending_downmarket_allows_sell(self, mock_ohlcv_data, features_with_regime):
        """Test that trending downmarket allows Sell but not Buy."""
        # Force all timesteps to trending downmarket
        features_trending_down = features_with_regime.copy()
        features_trending_down["regime_state"] = 2   # Trending
        features_trending_down["trend_bias"] = -1    # Downtrend

        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=10,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_trending_down, env_config)

        obs, info = env.reset()

        # In downtrend, Hold and Sell should be allowed
        expected_mask = np.array([True, False, True])
        np.testing.assert_array_equal(info["action_mask"], expected_mask)

    def test_transition_market_allows_all_actions(self, mock_ohlcv_data, features_with_regime):
        """Test that transition market allows all actions."""
        # Force all timesteps to transition regime
        features_transition = features_with_regime.copy()
        features_transition["regime_state"] = 1   # Transition
        features_transition["trend_bias"] = 0     # Any bias

        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=10,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_transition, env_config)

        obs, info = env.reset()

        # In transition, all actions should be allowed
        expected_mask = np.array([True, True, True])
        np.testing.assert_array_equal(info["action_mask"], expected_mask)

    def test_full_episode_with_masking(self, mock_ohlcv_data, features_with_regime):
        """Test a full episode with action masking enabled."""
        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=30,
            action_mask_config=ActionMaskConfig(enabled=True),
        )

        env = TradingEnv(mock_ohlcv_data, features_with_regime, env_config)

        obs, info = env.reset()

        step_count = 0
        mask_count = 0

        while step_count < 30:
            # Always choose Hold (always valid)
            action = 0

            obs, reward, terminated, truncated, info = env.step(action)

            # Track masks
            if "action_mask" in info:
                mask_count += 1
                # Verify mask is valid
                assert info["action_mask"].shape == (3,)
                assert info["action_mask"][0] == True  # Hold always allowed

            step_count += 1

            if terminated or truncated:
                break

        # Verify we got action masks throughout episode
        assert mask_count > 0
        assert mask_count == step_count
