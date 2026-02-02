"""Configuration dataclasses for RL agents."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from loguru import logger


@dataclass
class PPOConfig:
    """
    Configuration for PPO agent hyperparameters.

    Matches Stable-Baselines3 PPO parameters with sensible defaults
    for trading environments.

    Attributes:
        learning_rate: Learning rate for optimizer (default: 3e-4)
        n_steps: Number of steps to run per environment per update (default: 2048)
        batch_size: Minibatch size for optimization (default: 64)
        n_epochs: Number of epochs for optimization per update (default: 10)
        gamma: Discount factor for future rewards (default: 0.99)
        gae_lambda: GAE lambda for advantage estimation (default: 0.95)
        clip_range: PPO clip range for policy updates (default: 0.2)
        clip_range_vf: Value function clip range, None for no clipping (default: None)
        ent_coef: Entropy coefficient for exploration bonus (default: 0.0)
        vf_coef: Value function coefficient in loss (default: 0.5)
        max_grad_norm: Max gradient norm for clipping (default: 0.5)
        network_arch: MLP hidden layer sizes (default: [256, 256])
        activation_fn: Activation function name ('tanh', 'relu') (default: 'tanh')
        normalize_advantage: Whether to normalize advantages (default: True)
        use_cnn_extractor: Whether to use CNN-based feature extraction (default: False)
        cnn_type: Type of CNN architecture ('simple', 'multiscale', 'residual') (default: 'multiscale')
        price_embed_dim: CNN embedding dimension for price data (default: 128)
        ind_embed_dim: MLP embedding dimension for indicators (default: 64)
        port_embed_dim: MLP embedding dimension for portfolio state (default: 32)
        cnn_dropout: Dropout probability for CNN layers (default: 0.1)
        use_fusion: Whether to use fusion layer after concatenating embeddings (default: True)

    Example:
        >>> config = PPOConfig(learning_rate=0.0001, network_arch=[512, 512])
        >>> print(f"LR: {config.learning_rate}, Arch: {config.network_arch}")
        LR: 0.0001, Arch: [512, 512]

        >>> # CNN-based configuration
        >>> cnn_config = PPOConfig(
        ...     use_cnn_extractor=True,
        ...     cnn_type='multiscale',
        ...     price_embed_dim=128
        ... )
    """

    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    network_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"
    normalize_advantage: bool = True

    # CNN feature extractor options
    use_cnn_extractor: bool = True
    extractor_type: Literal["hybrid", "trading"] = "hybrid"  # "hybrid" for HybridCNNExtractor, "trading" for TradingCNNExtractor
    cnn_type: Literal["simple", "multiscale", "residual"] = "multiscale"
    price_embed_dim: int = 128
    ind_embed_dim: int = 64
    port_embed_dim: int = 32
    fund_embed_dim: int = 16  # For fundamental features (EOD only)
    cnn_dropout: float = 0.1
    use_fusion: bool = True
    use_attention: bool = True  # For TradingCNNExtractor attention mechanism

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.clip_range <= 0:
            raise ValueError(f"clip_range must be positive, got {self.clip_range}")
        if len(self.network_arch) == 0:
            raise ValueError("network_arch cannot be empty")
        if self.activation_fn not in ("tanh", "relu"):
            raise ValueError(
                f"activation_fn must be 'tanh' or 'relu', got {self.activation_fn}"
            )

        # Validate CNN-specific options
        if self.use_cnn_extractor:
            if self.cnn_type not in ("simple", "multiscale", "residual"):
                raise ValueError(
                    f"cnn_type must be 'simple', 'multiscale', or 'residual', "
                    f"got {self.cnn_type}"
                )
            if self.price_embed_dim <= 0:
                raise ValueError(f"price_embed_dim must be positive, got {self.price_embed_dim}")
            if self.ind_embed_dim <= 0:
                raise ValueError(f"ind_embed_dim must be positive, got {self.ind_embed_dim}")
            if self.port_embed_dim <= 0:
                raise ValueError(f"port_embed_dim must be positive, got {self.port_embed_dim}")
            if not 0 <= self.cnn_dropout < 1:
                raise ValueError(f"cnn_dropout must be in [0, 1), got {self.cnn_dropout}")

        logger.debug(f"PPOConfig initialized: {self}")


@dataclass
class TrainingConfig:
    """
    Configuration for training process.

    Attributes:
        total_timesteps: Total environment steps to train (default: 2,000,000)
        n_envs: Number of parallel environments for training (default: 8)
        eval_freq: Evaluation frequency in timesteps (default: 10,000)
        n_eval_episodes: Number of episodes for each evaluation (default: 5)
        checkpoint_freq: Checkpoint save frequency in timesteps (default: 50,000)
        log_interval: Logging interval in PPO updates (default: 10)
        seed: Random seed for reproducibility, None for random (default: None)
        device: Device for training ('auto', 'cpu', 'cuda') (default: 'auto')
        tensorboard_log: TensorBoard log directory (default: 'logs/tensorboard')
        model_save_dir: Directory for saved model checkpoints (default: 'models')
        best_model_save_path: Path for best model based on eval (default: 'models/best')
        verbose: Verbosity level (0=none, 1=info, 2=debug) (default: 1)

    Example:
        >>> config = TrainingConfig(total_timesteps=1000000, n_envs=4)
        >>> print(f"Steps: {config.total_timesteps}, Envs: {config.n_envs}")
        Steps: 1000000, Envs: 4
    """

    total_timesteps: int = 2000000
    n_envs: int = 8
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 50000
    log_interval: int = 10
    seed: Optional[int] = None
    device: str = "auto"
    tensorboard_log: str = "logs/tensorboard"
    model_save_dir: str = "models"
    best_model_save_path: str = "models/best"
    verbose: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.total_timesteps <= 0:
            raise ValueError(
                f"total_timesteps must be positive, got {self.total_timesteps}"
            )
        if self.n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {self.n_envs}")
        if self.eval_freq <= 0:
            raise ValueError(f"eval_freq must be positive, got {self.eval_freq}")
        if self.n_eval_episodes <= 0:
            raise ValueError(
                f"n_eval_episodes must be positive, got {self.n_eval_episodes}"
            )
        if self.checkpoint_freq <= 0:
            raise ValueError(
                f"checkpoint_freq must be positive, got {self.checkpoint_freq}"
            )
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError(
                f"device must be 'auto', 'cpu', 'cuda', or 'mps', got {self.device}"
            )
        if self.verbose not in (0, 1, 2):
            raise ValueError(f"verbose must be 0, 1, or 2, got {self.verbose}")

        logger.debug(f"TrainingConfig initialized: {self}")


@dataclass
class AgentConfig:
    """
    Combined configuration for agent and training.

    This is the top-level configuration that combines algorithm-specific
    hyperparameters with training process configuration.

    Attributes:
        algorithm: RL algorithm to use ('PPO' or 'SAC') (default: 'PPO')
        ppo: PPO hyperparameters configuration
        training: Training process configuration

    Example:
        >>> config = AgentConfig(
        ...     algorithm="PPO",
        ...     ppo=PPOConfig(learning_rate=0.0001),
        ...     training=TrainingConfig(total_timesteps=500000)
        ... )
        >>> print(f"Algorithm: {config.algorithm}")
        Algorithm: PPO
    """

    algorithm: Literal["PPO", "SAC"] = "PPO"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.algorithm not in ("PPO", "SAC"):
            raise ValueError(
                f"algorithm must be 'PPO' or 'SAC', got {self.algorithm}"
            )

        # Ensure nested configs are proper instances
        if not isinstance(self.ppo, PPOConfig):
            self.ppo = PPOConfig(**self.ppo) if isinstance(self.ppo, dict) else PPOConfig()
        if not isinstance(self.training, TrainingConfig):
            self.training = (
                TrainingConfig(**self.training)
                if isinstance(self.training, dict)
                else TrainingConfig()
            )

        logger.info(f"AgentConfig initialized: algorithm={self.algorithm}")
