"""Unit tests for PPO agent wrapper."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from tradebox.agents import PPOAgent, PPOConfig, TrainingConfig


class MockTradingEnv(gym.Env):
    """Mock trading environment for testing."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        """Initialize mock environment."""
        super().__init__()
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32
        )
        self.current_step = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self.current_step = 0
        obs = np.random.randn(100).astype(np.float32)
        info = {
            "portfolio_value": 100000.0,
            "total_trades": 0,
        }
        return obs, info

    def step(self, action):
        """Take a step."""
        self.current_step += 1
        obs = np.random.randn(100).astype(np.float32)
        reward = np.random.randn() * 0.01  # Small random reward
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {
            "portfolio_value": 100000.0 + np.random.randn() * 1000,
            "total_trades": np.random.randint(0, 10),
        }
        return obs, reward, terminated, truncated, info


@pytest.fixture
def mock_env():
    """Create a mock trading environment."""
    return MockTradingEnv()


@pytest.fixture
def ppo_config():
    """Create PPO configuration for testing."""
    return PPOConfig(
        learning_rate=0.0003,
        n_steps=64,  # Small for testing
        batch_size=32,
        n_epochs=2,
        network_arch=[64, 64],  # Smaller network for testing
    )


@pytest.fixture
def training_config():
    """Create training configuration for testing."""
    return TrainingConfig(
        total_timesteps=128,  # Very small for testing
        n_envs=1,
        eval_freq=64,
        checkpoint_freq=64,
        verbose=0,
    )


class TestPPOAgentInit:
    """Tests for PPO agent initialization."""

    def test_init_with_env(self, mock_env, ppo_config, training_config) -> None:
        """Test agent initializes with environment."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        assert agent.model is not None
        assert agent.config == ppo_config
        assert agent.config.learning_rate == 0.0003

    def test_init_with_defaults(self, mock_env, training_config) -> None:
        """Test agent initializes with default PPO config but explicit training config."""
        agent = PPOAgent(mock_env, training_config=training_config)

        assert agent.config.learning_rate == 0.0003
        assert agent.config.network_arch == [256, 256]

    def test_init_with_training_config(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test agent initializes with training config."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        assert agent.training_config.total_timesteps == 128
        assert agent.training_config.n_envs == 1

    def test_init_requires_env_factory_for_multi_envs(
        self, mock_env, ppo_config
    ) -> None:
        """Test that env_factory is required when n_envs > 1."""
        training_config = TrainingConfig(n_envs=4, verbose=0)

        with pytest.raises(ValueError, match="env_factory must be provided"):
            PPOAgent(mock_env, ppo_config, training_config)

    def test_init_with_env_factory(self, ppo_config) -> None:
        """Test agent initializes with env_factory for multiple envs."""
        training_config = TrainingConfig(n_envs=2, verbose=0)

        def env_factory():
            return MockTradingEnv()

        agent = PPOAgent(
            MockTradingEnv(),
            ppo_config,
            training_config,
            env_factory=env_factory,
        )

        assert agent.model is not None


class TestPPOAgentTrain:
    """Tests for PPO agent training."""

    def test_train_runs(self, mock_env, ppo_config, training_config) -> None:
        """Test training runs without errors."""
        agent = PPOAgent(mock_env, ppo_config, training_config)
        agent.train(total_timesteps=64, progress_bar=False)

        assert agent.num_timesteps >= 64

    def test_train_with_callback(self, mock_env, ppo_config, training_config) -> None:
        """Test training with a callback."""
        from stable_baselines3.common.callbacks import BaseCallback

        class TestCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def _on_step(self):
                self.call_count += 1
                return True

        callback = TestCallback()
        agent = PPOAgent(mock_env, ppo_config, training_config)
        agent.train(total_timesteps=64, callback=callback, progress_bar=False)

        assert callback.call_count > 0

    def test_train_returns_self(self, mock_env, ppo_config, training_config) -> None:
        """Test train returns self for chaining."""
        agent = PPOAgent(mock_env, ppo_config, training_config)
        result = agent.train(total_timesteps=64, progress_bar=False)

        assert result is agent


class TestPPOAgentPredict:
    """Tests for PPO agent prediction."""

    def test_predict_returns_valid_action(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test predict returns valid action."""
        agent = PPOAgent(mock_env, ppo_config, training_config)
        obs, _ = mock_env.reset()

        action, state = agent.predict(obs)

        assert action in [0, 1, 2]
        assert state is None  # MLP policy has no state

    def test_predict_deterministic(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test deterministic prediction gives consistent results."""
        agent = PPOAgent(mock_env, ppo_config, training_config)
        obs, _ = mock_env.reset()

        # Multiple deterministic predictions should give same result
        action1, _ = agent.predict(obs, deterministic=True)
        action2, _ = agent.predict(obs, deterministic=True)

        assert action1 == action2

    def test_predict_batch(self, mock_env, ppo_config, training_config) -> None:
        """Test prediction with batch of observations."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        # Create batch of observations
        batch_obs = np.random.randn(5, 100).astype(np.float32)

        # This should work with VecEnv internally
        for obs in batch_obs:
            action, _ = agent.predict(obs)
            assert action in [0, 1, 2]


class TestPPOAgentSaveLoad:
    """Tests for PPO agent serialization."""

    def test_save_creates_files(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test save creates model and config files."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            agent.save(save_path)

            assert (Path(tmpdir) / "test_model.zip").exists()
            assert (Path(tmpdir) / "test_model_config.json").exists()
            assert (Path(tmpdir) / "test_model_metadata.json").exists()

    def test_save_load_roundtrip(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test model can be saved and loaded."""
        agent = PPOAgent(mock_env, ppo_config, training_config)
        agent.train(total_timesteps=64, progress_bar=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            agent.save(save_path)

            loaded_agent = PPOAgent.load(save_path, env=mock_env)

            assert loaded_agent.config.learning_rate == agent.config.learning_rate
            assert loaded_agent.config.network_arch == agent.config.network_arch

    def test_load_without_env(self, mock_env, ppo_config, training_config) -> None:
        """Test model can be loaded without environment for inference."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            agent.save(save_path)

            loaded_agent = PPOAgent.load(save_path, env=None)

            # Should be able to predict
            obs = np.random.randn(100).astype(np.float32)
            action, _ = loaded_agent.predict(obs)
            assert action in [0, 1, 2]

    def test_load_nonexistent_raises(self) -> None:
        """Test loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            PPOAgent.load("/nonexistent/path/model")


class TestPPOAgentEvaluate:
    """Tests for PPO agent evaluation."""

    def test_evaluate_returns_metrics(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test evaluate returns expected metrics."""
        agent = PPOAgent(mock_env, ppo_config, training_config)
        agent.train(total_timesteps=64, progress_bar=False)

        metrics = agent.evaluate(mock_env, n_eval_episodes=2)

        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "mean_ep_length" in metrics
        assert "sharpe_ratio" in metrics
        assert "n_episodes" in metrics
        assert metrics["n_episodes"] == 2

    def test_evaluate_metrics_types(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test evaluate returns correct types."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        metrics = agent.evaluate(mock_env, n_eval_episodes=2)

        assert isinstance(metrics["mean_reward"], float)
        assert isinstance(metrics["std_reward"], float)
        assert isinstance(metrics["mean_ep_length"], float)
        assert isinstance(metrics["sharpe_ratio"], float)


class TestPPOAgentProperties:
    """Tests for PPO agent properties."""

    def test_policy_property(self, mock_env, ppo_config, training_config) -> None:
        """Test policy property returns policy network."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        policy = agent.policy

        assert policy is not None
        assert hasattr(policy, "forward")

    def test_num_timesteps_property(
        self, mock_env, ppo_config, training_config
    ) -> None:
        """Test num_timesteps property tracks training."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        assert agent.num_timesteps == 0

        agent.train(total_timesteps=64, progress_bar=False)

        assert agent.num_timesteps >= 64

    def test_get_parameters(self, mock_env, ppo_config, training_config) -> None:
        """Test get_parameters returns configuration."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        params = agent.get_parameters()

        assert params["algorithm"] == "PPO"
        assert params["learning_rate"] == ppo_config.learning_rate
        assert params["n_steps"] == ppo_config.n_steps
        assert params["batch_size"] == ppo_config.batch_size
        assert params["network_arch"] == ppo_config.network_arch

    def test_get_env(self, mock_env, ppo_config, training_config) -> None:
        """Test get_env returns environment."""
        agent = PPOAgent(mock_env, ppo_config, training_config)

        env = agent.get_env()

        assert env is not None
