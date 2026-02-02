"""Unit tests for custom training callbacks."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from tradebox.agents.callbacks import (
    EarlyStoppingCallback,
    TradingCheckpointCallback,
    TradingEvalCallback,
    TradingMetricsCallback,
    create_callback_list,
)
from tradebox.agents.config import TrainingConfig


class MockEnv:
    """Mock environment for testing callbacks."""

    def __init__(self):
        self.num_envs = 1


class TestTradingMetricsCallback:
    """Tests for TradingMetricsCallback."""

    def test_init(self) -> None:
        """Test callback initialization."""
        callback = TradingMetricsCallback(verbose=1)

        assert callback.verbose == 1
        assert callback.episode_rewards == []
        assert callback.episode_values == []

    def test_on_training_start_resets_lists(self) -> None:
        """Test that training start resets metric lists."""
        callback = TradingMetricsCallback()
        callback.episode_rewards = [1, 2, 3]
        callback.episode_values = [100, 200]

        callback._on_training_start()

        assert callback.episode_rewards == []
        assert callback.episode_values == []

    def test_on_step_returns_true(self) -> None:
        """Test that on_step returns True to continue training."""
        callback = TradingMetricsCallback()
        callback.locals = {"dones": [], "infos": []}
        callback.n_calls = 1
        callback.logger = MagicMock()

        result = callback._on_step()

        assert result is True

    def test_on_step_extracts_episode_info(self) -> None:
        """Test that on_step extracts info from completed episodes."""
        callback = TradingMetricsCallback()
        callback.n_calls = 1
        callback.logger = MagicMock()

        # Simulate episode completion
        callback.locals = {
            "dones": [True],
            "infos": [{
                "episode": {"r": 100.0, "l": 50},
                "portfolio_value": 110000.0,
                "total_trades": 5,
            }],
        }

        callback._on_step()

        assert 100.0 in callback.episode_rewards
        assert 50 in callback.episode_lengths
        assert 110000.0 in callback.episode_values
        assert 5 in callback.episode_trades


class TestTradingEvalCallback:
    """Tests for TradingEvalCallback."""

    @pytest.fixture
    def mock_eval_env(self):
        """Create a mock evaluation environment."""
        from tests.unit.test_agents.test_ppo_agent import MockTradingEnv
        return DummyVecEnv([lambda: MockTradingEnv()])

    def test_init(self, mock_eval_env) -> None:
        """Test callback initialization."""
        callback = TradingEvalCallback(
            eval_env=mock_eval_env,
            n_eval_episodes=5,
            eval_freq=1000,
            metric="sharpe_ratio",
            verbose=0,
        )

        assert callback.n_eval_episodes == 5
        assert callback.eval_freq == 1000
        assert callback.metric == "sharpe_ratio"
        assert callback.best_sharpe == float("-inf")

    def test_calculate_sharpe_ratio(self, mock_eval_env) -> None:
        """Test Sharpe ratio calculation."""
        callback = TradingEvalCallback(
            eval_env=mock_eval_env,
            metric="sharpe_ratio",
            verbose=0,
        )

        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = callback._calculate_sharpe_ratio(returns)

        # Sharpe should be positive for positive average returns
        assert sharpe > 0

    def test_calculate_sharpe_ratio_with_zero_std(self, mock_eval_env) -> None:
        """Test Sharpe ratio with zero standard deviation."""
        callback = TradingEvalCallback(
            eval_env=mock_eval_env,
            metric="sharpe_ratio",
            verbose=0,
        )

        returns = np.array([0.01, 0.01, 0.01])  # Same returns, zero std
        sharpe = callback._calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_with_single_return(self, mock_eval_env) -> None:
        """Test Sharpe ratio with insufficient data."""
        callback = TradingEvalCallback(
            eval_env=mock_eval_env,
            metric="sharpe_ratio",
            verbose=0,
        )

        returns = np.array([0.01])
        sharpe = callback._calculate_sharpe_ratio(returns)

        assert sharpe == 0.0


class TestTradingCheckpointCallback:
    """Tests for TradingCheckpointCallback."""

    def test_init(self) -> None:
        """Test callback initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = TradingCheckpointCallback(
                save_freq=1000,
                save_path=tmpdir,
                name_prefix="test_model",
                save_config=True,
                verbose=0,
            )

            assert callback.save_freq == 1000
            assert callback.save_path == tmpdir
            assert callback.name_prefix == "test_model"
            assert callback.save_config is True


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_init(self) -> None:
        """Test callback initialization."""
        callback = EarlyStoppingCallback(
            metric="sharpe_ratio",
            patience=10,
            min_delta=0.01,
            check_freq=1000,
            verbose=1,
        )

        assert callback.metric == "sharpe_ratio"
        assert callback.patience == 10
        assert callback.min_delta == 0.01
        assert callback.check_freq == 1000
        assert callback.best_metric == float("-inf")
        assert callback.patience_counter == 0
        assert callback.stopped_early is False

    def test_on_step_continues_training(self) -> None:
        """Test that on_step returns True normally."""
        callback = EarlyStoppingCallback(check_freq=100)
        callback.n_calls = 1  # Not at check frequency

        result = callback._on_step()

        assert result is True
        assert callback.stopped_early is False

    def test_on_step_at_check_freq(self) -> None:
        """Test on_step at check frequency without metric."""
        callback = EarlyStoppingCallback(check_freq=100)
        callback.n_calls = 100  # At check frequency
        callback.parent = None  # No parent callback list

        result = callback._on_step()

        # Should continue if no metric available
        assert result is True


class TestCreateCallbackList:
    """Tests for create_callback_list factory function."""

    @pytest.fixture
    def mock_eval_env(self):
        """Create a mock evaluation environment."""
        from tests.unit.test_agents.test_ppo_agent import MockTradingEnv
        return DummyVecEnv([lambda: MockTradingEnv()])

    def test_creates_callback_list(self, mock_eval_env) -> None:
        """Test factory creates callback list."""
        training_config = TrainingConfig(
            eval_freq=1000,
            checkpoint_freq=5000,
            n_eval_episodes=3,
            verbose=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            training_config.model_save_dir = tmpdir
            training_config.best_model_save_path = f"{tmpdir}/best"

            callbacks = create_callback_list(
                eval_env=mock_eval_env,
                training_config=training_config,
            )

            assert callbacks is not None
            assert len(callbacks.callbacks) == 3  # Eval, Checkpoint, Metrics

    def test_callback_list_contains_expected_types(self, mock_eval_env) -> None:
        """Test factory creates correct callback types."""
        training_config = TrainingConfig(verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_config.model_save_dir = tmpdir
            training_config.best_model_save_path = f"{tmpdir}/best"

            callbacks = create_callback_list(
                eval_env=mock_eval_env,
                training_config=training_config,
            )

            callback_types = [type(cb).__name__ for cb in callbacks.callbacks]

            assert "TradingEvalCallback" in callback_types
            assert "TradingCheckpointCallback" in callback_types
            assert "TradingMetricsCallback" in callback_types
