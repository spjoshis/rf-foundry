"""Unit tests for agent configuration dataclasses."""

import pytest

from tradebox.agents.config import AgentConfig, PPOConfig, TrainingConfig


class TestPPOConfig:
    """Tests for PPOConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PPOConfig()

        assert config.learning_rate == 0.0003
        assert config.n_steps == 2048
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.clip_range_vf is None
        assert config.ent_coef == 0.0
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5
        assert config.network_arch == [256, 256]
        assert config.activation_fn == "tanh"
        assert config.normalize_advantage is True

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = PPOConfig(
            learning_rate=0.0001,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            network_arch=[512, 512, 256],
            activation_fn="relu",
        )

        assert config.learning_rate == 0.0001
        assert config.n_steps == 1024
        assert config.batch_size == 32
        assert config.n_epochs == 5
        assert config.gamma == 0.95
        assert config.network_arch == [512, 512, 256]
        assert config.activation_fn == "relu"

    def test_invalid_learning_rate(self) -> None:
        """Test validation rejects invalid learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            PPOConfig(learning_rate=0)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            PPOConfig(learning_rate=-0.001)

    def test_invalid_n_steps(self) -> None:
        """Test validation rejects invalid n_steps."""
        with pytest.raises(ValueError, match="n_steps must be positive"):
            PPOConfig(n_steps=0)

    def test_invalid_batch_size(self) -> None:
        """Test validation rejects invalid batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            PPOConfig(batch_size=-1)

    def test_invalid_gamma(self) -> None:
        """Test validation rejects invalid gamma."""
        with pytest.raises(ValueError, match="gamma must be in"):
            PPOConfig(gamma=0)

        with pytest.raises(ValueError, match="gamma must be in"):
            PPOConfig(gamma=1.5)

    def test_invalid_gae_lambda(self) -> None:
        """Test validation rejects invalid gae_lambda."""
        with pytest.raises(ValueError, match="gae_lambda must be in"):
            PPOConfig(gae_lambda=-0.1)

        with pytest.raises(ValueError, match="gae_lambda must be in"):
            PPOConfig(gae_lambda=1.5)

    def test_invalid_clip_range(self) -> None:
        """Test validation rejects invalid clip_range."""
        with pytest.raises(ValueError, match="clip_range must be positive"):
            PPOConfig(clip_range=0)

    def test_invalid_network_arch(self) -> None:
        """Test validation rejects empty network_arch."""
        with pytest.raises(ValueError, match="network_arch cannot be empty"):
            PPOConfig(network_arch=[])

    def test_invalid_activation_fn(self) -> None:
        """Test validation rejects invalid activation function."""
        with pytest.raises(ValueError, match="activation_fn must be"):
            PPOConfig(activation_fn="sigmoid")


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.total_timesteps == 2000000
        assert config.n_envs == 8
        assert config.eval_freq == 10000
        assert config.n_eval_episodes == 5
        assert config.checkpoint_freq == 50000
        assert config.log_interval == 10
        assert config.seed is None
        assert config.device == "auto"
        assert config.tensorboard_log == "logs/tensorboard"
        assert config.model_save_dir == "models"
        assert config.best_model_save_path == "models/best"
        assert config.verbose == 1

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = TrainingConfig(
            total_timesteps=500000,
            n_envs=4,
            eval_freq=5000,
            seed=42,
            device="cpu",
            verbose=0,
        )

        assert config.total_timesteps == 500000
        assert config.n_envs == 4
        assert config.eval_freq == 5000
        assert config.seed == 42
        assert config.device == "cpu"
        assert config.verbose == 0

    def test_invalid_total_timesteps(self) -> None:
        """Test validation rejects invalid total_timesteps."""
        with pytest.raises(ValueError, match="total_timesteps must be positive"):
            TrainingConfig(total_timesteps=0)

    def test_invalid_n_envs(self) -> None:
        """Test validation rejects invalid n_envs."""
        with pytest.raises(ValueError, match="n_envs must be positive"):
            TrainingConfig(n_envs=0)

    def test_invalid_eval_freq(self) -> None:
        """Test validation rejects invalid eval_freq."""
        with pytest.raises(ValueError, match="eval_freq must be positive"):
            TrainingConfig(eval_freq=-100)

    def test_invalid_device(self) -> None:
        """Test validation rejects invalid device."""
        with pytest.raises(ValueError, match="device must be"):
            TrainingConfig(device="invalid")

    def test_invalid_verbose(self) -> None:
        """Test validation rejects invalid verbose."""
        with pytest.raises(ValueError, match="verbose must be"):
            TrainingConfig(verbose=3)


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AgentConfig()

        assert config.algorithm == "PPO"
        assert isinstance(config.ppo, PPOConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_custom_nested_configs(self) -> None:
        """Test configuration with custom nested configs."""
        ppo = PPOConfig(learning_rate=0.0001)
        training = TrainingConfig(total_timesteps=100000)

        config = AgentConfig(
            algorithm="PPO",
            ppo=ppo,
            training=training,
        )

        assert config.ppo.learning_rate == 0.0001
        assert config.training.total_timesteps == 100000

    def test_invalid_algorithm(self) -> None:
        """Test validation rejects invalid algorithm."""
        with pytest.raises(ValueError, match="algorithm must be"):
            AgentConfig(algorithm="DQN")

    def test_dict_conversion(self) -> None:
        """Test that dict inputs are converted to proper config objects."""
        config = AgentConfig(
            ppo={"learning_rate": 0.0005},
            training={"total_timesteps": 500000},
        )

        assert isinstance(config.ppo, PPOConfig)
        assert config.ppo.learning_rate == 0.0005
        assert isinstance(config.training, TrainingConfig)
        assert config.training.total_timesteps == 500000
