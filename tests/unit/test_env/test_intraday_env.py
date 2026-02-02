"""Unit tests for IntradayTradingEnv."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from tradebox.env.intraday_env import IntradayTradingEnv
from tradebox.env.trading_env import IntradayEnvConfig
from tradebox.env.costs import CostConfig
from tradebox.env.rewards import RewardConfig


@pytest.fixture
def sample_intraday_data():
    """
    Create sample 5-minute intraday data for testing.

    Generates 1,000 bars (~13 sessions) of synthetic price data.
    """
    np.random.seed(42)
    n_bars = 1000

    # Generate realistic price series
    base_price = 1000.0
    returns = np.random.normal(0.0001, 0.01, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.002, 0.002, n_bars)),
        "High": prices * (1 + np.abs(np.random.uniform(0, 0.01, n_bars))),
        "Low": prices * (1 - np.abs(np.random.uniform(0, 0.01, n_bars))),
        "Close": prices,
        "Volume": np.random.randint(100000, 1000000, n_bars),
    })

    # Add datetime index (5-minute bars starting from market open)
    start_date = pd.Timestamp("2024-01-01 09:15:00")
    df["Date"] = pd.date_range(start=start_date, periods=n_bars, freq="5T")

    return df


@pytest.fixture
def sample_intraday_features(sample_intraday_data):
    """
    Create sample technical features for intraday data.

    Includes key indicators: SMA, RSI, ATR, VWAP
    """
    df = sample_intraday_data.copy()

    # Calculate simple features
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["RSI"] = 50.0 + np.random.uniform(-20, 20, len(df))  # Simplified RSI
    df["ATR"] = df["High"] - df["Low"]  # Simplified ATR
    df["VWAP"] = df["Close"] * 1.001  # Simplified VWAP
    df["VWAP_Deviation"] = (df["Close"] - df["VWAP"]) / df["VWAP"]

    # Fill NaN values from rolling calculations
    df = df.bfill()

    # Select only feature columns (exclude OHLCV and Date)
    feature_cols = ["SMA_10", "SMA_20", "RSI", "ATR", "VWAP", "VWAP_Deviation"]
    features = df[feature_cols].copy()

    return features


@pytest.fixture
def intraday_config():
    """Create IntradayEnvConfig for testing."""
    return IntradayEnvConfig(
        initial_capital=100000.0,
        lookback_window=60,
        max_episode_steps=750,  # 10 sessions × 75 bars
        bars_per_session=75,
        sessions_per_episode=10,
        force_close_eod=True,
        cost_config=CostConfig(use_dynamic_slippage=True),
        reward_config=RewardConfig(reward_type="simple"),
    )


class TestIntradayEnvConfig:
    """Test suite for IntradayEnvConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = IntradayEnvConfig()

        assert config.initial_capital == 100000.0
        assert config.lookback_window == 60
        assert config.bar_interval_minutes == 5
        assert config.bars_per_session == 75
        assert config.sessions_per_episode == 10
        assert config.force_close_eod is True
        assert config.market_open_time == "09:15"
        assert config.market_close_time == "15:30"
        assert config.overnight_gap_handling == "reset_observation"

    def test_config_validates_max_episode_steps(self):
        """Test that max_episode_steps is validated against sessions × bars."""
        config = IntradayEnvConfig(
            sessions_per_episode=10,
            bars_per_session=75,
            max_episode_steps=500,  # Wrong value
        )

        # Should be auto-corrected to 10 × 75 = 750
        assert config.max_episode_steps == 750

    def test_config_validates_gap_handling(self):
        """Test that invalid gap handling raises ValueError."""
        with pytest.raises(ValueError, match="overnight_gap_handling must be one of"):
            IntradayEnvConfig(overnight_gap_handling="invalid")


class TestIntradayTradingEnv:
    """Test suite for IntradayTradingEnv."""

    def test_env_initialization(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test environment initializes correctly."""
        env = IntradayTradingEnv(
            data=sample_intraday_data,
            features=sample_intraday_features,
            config=intraday_config,
        )

        # Check action space
        assert env.action_space.n == 3  # {Hold, Buy, Sell}

        # Check observation space
        # Expected: 60 bars × 6 features + 6 portfolio features = 366
        expected_obs_size = 60 * 6 + 6
        assert env.observation_space.shape == (expected_obs_size,)

        # Check config
        assert env.config.initial_capital == 100000.0
        assert env.config.bars_per_session == 75

    def test_env_validates_data_length(self, sample_intraday_features, intraday_config):
        """Test that short data raises ValueError."""
        # Create data that's too short
        short_data = pd.DataFrame({
            "Open": [100, 101],
            "High": [101, 102],
            "Low": [99, 100],
            "Close": [100.5, 101.5],
            "Volume": [1000, 1000],
        })

        with pytest.raises(ValueError, match="Data too short"):
            IntradayTradingEnv(
                data=short_data,
                features=sample_intraday_features.iloc[:2],
                config=intraday_config,
            )

    def test_env_validates_data_feature_alignment(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test that misaligned data and features raises ValueError."""
        # Truncate features
        short_features = sample_intraday_features.iloc[:100]

        with pytest.raises(ValueError, match="length mismatch"):
            IntradayTradingEnv(
                data=sample_intraday_data,
                features=short_features,
                config=intraday_config,
            )

    def test_reset(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test environment reset."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)

        obs, info = env.reset(seed=42)

        # Check observation shape
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

        # Check info
        assert "step" in info
        assert "session" in info
        assert "portfolio_value" in info
        assert info["portfolio_value"] == 100000.0
        assert info["cash"] == 100000.0
        assert info["position"] == 0

        # Check portfolio reset
        assert env.cash == 100000.0
        assert env.position == 0
        assert env.total_trades == 0
        assert env.current_session == 0

    def test_step_hold_action(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test step with Hold action (action=0)."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        initial_cash = env.cash
        initial_position = env.position

        obs, reward, terminated, truncated, info = env.step(0)  # Hold

        # Cash and position should not change
        assert env.cash == initial_cash
        assert env.position == initial_position
        assert env.total_trades == 0

        # Observation should be valid
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_buy_action(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test step with Buy action (action=1)."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        initial_cash = env.cash

        obs, reward, terminated, truncated, info = env.step(1)  # Buy

        # Should have bought shares
        assert env.position > 0
        assert env.cash < initial_cash
        assert env.total_trades == 1
        assert env.entry_price > 0

        # Position value should be significant
        position_value = env.position * sample_intraday_data.iloc[env.current_step]["Close"]
        assert position_value > 0

    def test_step_sell_action(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test step with Sell action (action=2)."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # First buy
        env.step(1)  # Buy
        assert env.position > 0

        shares_held = env.position

        # Then sell
        env.step(2)  # Sell

        # Position should be closed
        assert env.position == 0
        assert env.entry_price == 0.0
        assert env.total_trades == 2

        # Cash should have changed (may be profit or loss)
        assert env.cash != env.config.initial_capital

    def test_session_end_forced_closure(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test that positions are forced closed at end of session."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # Buy at start of session
        env.step(1)  # Buy
        assert env.position > 0

        # Manually advance to end of session
        env.bars_in_session = intraday_config.bars_per_session - 1

        # Take one more step (should trigger session end)
        env.step(0)  # Hold

        # If force_close_eod is True, position should be closed
        if intraday_config.force_close_eod:
            assert env.position == 0
            assert env.current_session == 1
            assert env.bars_in_session == 0

    def test_episode_termination(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test episode terminates after max sessions."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # Manually set to last session
        env.current_session = intraday_config.sessions_per_episode - 1
        env.bars_in_session = intraday_config.bars_per_session - 1

        # Take step to complete episode
        obs, reward, terminated, truncated, info = env.step(0)

        # Episode should terminate
        assert terminated or truncated

    def test_portfolio_value_calculation(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test portfolio value calculation."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        initial_value = env._get_portfolio_value()
        assert initial_value == env.config.initial_capital

        # Buy shares
        env.step(1)  # Buy

        portfolio_value = env._get_portfolio_value()
        current_price = sample_intraday_data.iloc[env.current_step]["Close"]
        expected_value = env.cash + (env.position * current_price)

        # Should match (within floating point precision)
        assert abs(portfolio_value - expected_value) < 0.01

    def test_observation_structure(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test observation vector structure."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        obs = env._get_observation()

        # Check shape
        n_features = len(sample_intraday_features.columns)
        expected_size = (intraday_config.lookback_window * n_features) + 6
        assert obs.shape == (expected_size,)

        # Check dtype
        assert obs.dtype == np.float32

        # Check no NaN or Inf
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_portfolio_state_features(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test portfolio state feature extraction."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # Get portfolio state
        portfolio_state = env._get_portfolio_state()

        # Should have 6 features
        assert portfolio_state.shape == (6,)

        # Features: [position_pct, cash_pct, unrealized_pnl_pct, entry_price_dev, bars_held, trades_today_norm]
        assert 0.0 <= portfolio_state[0] <= 1.0  # position_pct
        assert 0.0 <= portfolio_state[1] <= 1.0  # cash_pct
        # unrealized_pnl_pct can be any value
        # entry_price_dev can be any value
        assert portfolio_state[4] >= 0.0  # bars_held
        assert portfolio_state[5] >= 0.0  # trades_today_norm

    def test_dynamic_slippage_integration(self, sample_intraday_data, sample_intraday_features):
        """Test dynamic slippage is used when enabled."""
        # Config with dynamic slippage
        config_dynamic = IntradayEnvConfig(
            cost_config=CostConfig(use_dynamic_slippage=True)
        )

        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, config_dynamic)
        env.reset(seed=42)

        initial_cash = env.cash

        # Execute buy
        env.step(1)  # Buy

        # Cash should decrease (more than just trade value due to costs)
        assert env.cash < initial_cash

        # Verify cost model is configured for dynamic slippage
        assert env.cost_model.config.use_dynamic_slippage is True

    def test_multiple_episodes(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test multiple episode resets."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)

        for episode in range(3):
            obs, info = env.reset(seed=42 + episode)

            # Each episode should start fresh
            assert env.cash == env.config.initial_capital
            assert env.position == 0
            assert env.total_trades == 0
            assert env.current_session == 0

            # Take a few steps
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break

    def test_info_dict_completeness(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test info dict contains all expected fields."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        _, _, _, _, info = env.step(0)

        required_fields = [
            "step", "session", "bars_in_session", "portfolio_value",
            "cash", "position", "position_value", "entry_price",
            "current_price", "unrealized_pnl_pct", "total_trades",
            "trades_today", "bars_held"
        ]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"

    def test_trades_today_counter(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test that trades_today counter resets at session boundaries."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # Execute trades
        env.step(1)  # Buy
        env.step(2)  # Sell

        assert env.trades_today == 2

        # Advance to next session
        env.bars_in_session = intraday_config.bars_per_session
        env._check_session_end()

        # trades_today should reset
        assert env.trades_today == 0


class TestIntradayEnvEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_volume_column(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test environment handles missing Volume gracefully."""
        data_no_volume = sample_intraday_data.drop(columns=["Volume"])

        # Should not raise error (uses default volume)
        env = IntradayTradingEnv(data_no_volume, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # Should still work
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        assert env.position > 0

    def test_missing_atr_feature(self, sample_intraday_data, intraday_config):
        """Test environment handles missing ATR feature gracefully."""
        # Features without ATR
        features = pd.DataFrame({
            "SMA_10": np.random.randn(len(sample_intraday_data)),
            "RSI": np.random.randn(len(sample_intraday_data)),
        })

        env = IntradayTradingEnv(sample_intraday_data, features, intraday_config)
        env.reset(seed=42)

        # Should use default ATR (1% of price)
        obs, reward, terminated, truncated, info = env.step(1)  # Buy
        assert env.position > 0

    def test_portfolio_value_drops_below_threshold(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test episode truncates when portfolio drops too low."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # Manually set portfolio to very low value
        env.cash = 20000.0  # 80% loss
        env.position = 0

        obs, reward, terminated, truncated, info = env.step(0)

        # Should truncate (70% loss threshold)
        assert truncated

    def test_end_of_data_truncation(self, sample_intraday_data, sample_intraday_features, intraday_config):
        """Test episode truncates when reaching end of data."""
        env = IntradayTradingEnv(sample_intraday_data, sample_intraday_features, intraday_config)
        env.reset(seed=42)

        # Manually advance to near end of data
        env.current_step = len(sample_intraday_data) - 2

        obs, reward, terminated, truncated, info = env.step(0)

        # Should truncate
        assert truncated


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
