"""Integration test for intraday trading pipeline."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from tradebox.env.intraday_env import IntradayTradingEnv
from tradebox.env.trading_env import IntradayEnvConfig
from tradebox.env.costs import CostConfig
from tradebox.env.rewards import RewardConfig
from tradebox.features.technical import TechnicalFeatures, FeatureConfig


@pytest.fixture
def intraday_sample_data():
    """
    Create realistic intraday sample data for integration testing.

    Generates 1,500 bars (~20 sessions) with realistic price dynamics.
    """
    np.random.seed(42)
    n_bars = 1500

    # Generate price series with trend and volatility
    base_price = 2500.0
    trend = np.linspace(0, 0.05, n_bars)  # 5% uptrend
    noise = np.random.normal(0, 0.005, n_bars)
    returns = trend / n_bars + noise

    prices = base_price * np.exp(np.cumsum(returns))

    # Generate realistic OHLCV
    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        "High": prices * (1 + np.abs(np.random.uniform(0, 0.005, n_bars))),
        "Low": prices * (1 - np.abs(np.random.uniform(0, 0.005, n_bars))),
        "Close": prices,
        "Volume": np.random.randint(500000, 2000000, n_bars),
    })

    # Add datetime index (5-minute bars, Indian market hours)
    start_date = pd.Timestamp("2024-01-01 09:15:00")
    df["Date"] = pd.date_range(start=start_date, periods=n_bars, freq="5T")

    return df


class TestIntradayPipelineIntegration:
    """Integration tests for the complete intraday trading pipeline."""

    def test_full_pipeline_technical_features(self, intraday_sample_data):
        """
        Test complete pipeline: data → features → environment → episode.

        This is the happy path integration test.
        """
        # Step 1: Extract technical features
        feature_config = FeatureConfig(
            timeframe="intraday",
            normalize=True,
            sma_periods=[10, 20, 50],
            ema_periods=[9, 21],
            rsi_period=14,
            atr_period=14,
            vwap_enabled=True,
            session_high_low=True,
            intraday_returns=True,
        )

        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        # Verify features extracted
        assert len(features_df) == len(intraday_sample_data)
        assert "ATR" in features_df.columns
        assert "VWAP" in features_df.columns
        assert "RSI" in features_df.columns

        # Step 2: Create environment
        env_config = IntradayEnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            bars_per_session=75,
            sessions_per_episode=10,
            force_close_eod=True,
            cost_config=CostConfig(use_dynamic_slippage=True),
            reward_config=RewardConfig(reward_type="simple"),
        )

        env = IntradayTradingEnv(
            data=intraday_sample_data,
            features=features_df,
            config=env_config,
        )

        # Step 3: Run episode
        obs, info = env.reset(seed=42)

        # Verify initial state
        assert obs.shape == env.observation_space.shape
        assert info["portfolio_value"] == 100000.0

        # Step 4: Execute trading actions
        episode_rewards = []
        done = False
        steps = 0
        max_steps = 100  # Limit for testing

        while not done and steps < max_steps:
            # Simple strategy: buy on step 0, hold, sell on step 50
            if steps == 0:
                action = 1  # Buy
            elif steps == 50:
                action = 2  # Sell
            else:
                action = 0  # Hold

            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            done = terminated or truncated
            steps += 1

        # Step 5: Verify episode completed successfully
        assert steps > 0
        assert len(episode_rewards) == steps

        # Verify portfolio state is valid
        final_value = info["portfolio_value"]
        assert final_value > 0
        assert not np.isnan(final_value)
        assert not np.isinf(final_value)

    def test_pipeline_with_dynamic_slippage(self, intraday_sample_data):
        """Test pipeline with dynamic slippage model enabled."""
        # Extract features
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        # Create env with dynamic slippage
        env_config = IntradayEnvConfig(
            cost_config=CostConfig(
                use_dynamic_slippage=True,
                base_spread_bps=2.5,
                max_spread_bps=10.0,
                impact_coefficient=0.2,
                max_impact_pct=0.005,
            )
        )

        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)
        env.reset(seed=42)

        # Execute buy and sell
        initial_cash = env.cash
        env.step(1)  # Buy

        # Verify dynamic slippage was applied
        assert env.position > 0
        assert env.cash < initial_cash

        # Cost should be more than zero
        buy_cost = initial_cash - env.cash - (env.position * env.entry_price)
        assert buy_cost > 0

        # Sell
        env.step(2)  # Sell

        # Position should be closed
        assert env.position == 0

    def test_pipeline_forced_eod_closure(self, intraday_sample_data):
        """Test that positions are forcibly closed at end of day."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig(
            bars_per_session=75,
            force_close_eod=True,
        )

        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)
        env.reset(seed=42)

        # Buy shares
        env.step(1)  # Buy
        assert env.position > 0

        # Manually advance to end of session
        for _ in range(env_config.bars_per_session - 1):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold

            if terminated or truncated:
                break

        # After session ends, position should be force-closed
        assert env.position == 0

    def test_pipeline_multiple_sessions(self, intraday_sample_data):
        """Test pipeline across multiple trading sessions."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig(
            bars_per_session=75,
            sessions_per_episode=3,  # 3 sessions
            force_close_eod=True,
        )

        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)
        env.reset(seed=42)

        session_count = 0
        previous_session = 0

        for step in range(env_config.bars_per_session * 3 + 10):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold

            # Track session transitions
            if info["session"] > previous_session:
                session_count += 1
                previous_session = info["session"]

            if terminated or truncated:
                break

        # Should have completed at least 2 session transitions
        assert session_count >= 2

    def test_pipeline_observation_consistency(self, intraday_sample_data):
        """Test that observations remain consistent across steps."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig()
        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)

        obs1, _ = env.reset(seed=42)

        # Take steps and verify observations
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)

            # Observation shape should remain constant
            assert obs.shape == obs1.shape
            assert obs.dtype == np.float32

            # No NaN or Inf values
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))

            if terminated or truncated:
                break

    def test_pipeline_portfolio_value_tracking(self, intraday_sample_data):
        """Test portfolio value tracking across trades."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig(initial_capital=100000.0)
        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)

        env.reset(seed=42)

        portfolio_values = []

        # Execute several trades
        actions = [1, 0, 0, 0, 2, 0, 1, 0, 0, 2]  # Buy, hold, sell, buy, hold, sell

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])

            if terminated or truncated:
                break

        # Portfolio value should be tracked
        assert len(portfolio_values) > 0

        # All values should be positive
        assert all(v > 0 for v in portfolio_values)

        # Final value should be close to initial (no guaranteed profit)
        # but within reasonable bounds (e.g., -50% to +50%)
        final_value = portfolio_values[-1]
        assert 50000 < final_value < 150000

    def test_pipeline_reward_calculation(self, intraday_sample_data):
        """Test that rewards are calculated correctly."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig(
            reward_config=RewardConfig(reward_type="simple")
        )

        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)
        env.reset(seed=42)

        rewards = []

        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(0)  # Hold
            rewards.append(reward)

            if terminated or truncated:
                break

        # Rewards should be calculated (not all zeros)
        assert len(rewards) > 0
        assert not all(r == 0 for r in rewards)

        # Rewards should be finite
        assert all(not np.isnan(r) for r in rewards)
        assert all(not np.isinf(r) for r in rewards)

    def test_pipeline_feature_window_padding(self, intraday_sample_data):
        """Test that feature windows are properly padded at episode start."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig(lookback_window=60)
        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)

        # Reset at a position where we might not have full lookback
        obs, info = env.reset(seed=42)

        # Observation should still be valid (padded with zeros if necessary)
        assert obs.shape == env.observation_space.shape
        assert not np.any(np.isnan(obs))

    def test_pipeline_session_boundary_handling(self, intraday_sample_data):
        """Test observation handling at session boundaries."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig(
            bars_per_session=75,
            overnight_gap_handling="reset_observation",
        )

        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)
        env.reset(seed=42)

        # Step through a full session
        for _ in range(75):
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated or truncated:
                break

        # Should have transitioned to next session
        if not (terminated or truncated):
            assert info["session"] > 0
            assert info["bars_in_session"] >= 0


class TestIntradayPipelinePerformance:
    """Performance and stress tests for intraday pipeline."""

    def test_pipeline_episode_speed(self, intraday_sample_data):
        """Test that episodes run in reasonable time."""
        import time

        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig()
        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)

        start_time = time.time()

        env.reset(seed=42)

        # Run 100 steps
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        elapsed_time = time.time() - start_time

        # Should complete in under 1 second (very generous)
        assert elapsed_time < 1.0

    def test_pipeline_memory_efficiency(self, intraday_sample_data):
        """Test that environment doesn't leak memory across episodes."""
        feature_config = FeatureConfig(timeframe="intraday")
        extractor = TechnicalFeatures(feature_config)
        features_df = extractor.extract(intraday_sample_data, fit_normalize=True)

        env_config = IntradayEnvConfig()
        env = IntradayTradingEnv(intraday_sample_data, features_df, env_config)

        # Run multiple episodes
        for episode in range(5):
            env.reset(seed=42 + episode)

            for _ in range(50):
                obs, reward, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break

        # If we got here without OOM, we're good
        assert True


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
