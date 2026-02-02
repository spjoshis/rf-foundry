"""PPO Agent wrapper for trading environments."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from tradebox.agents.base_agent import BaseAgent
from tradebox.agents.config import AgentConfig, PPOConfig, TrainingConfig
from tradebox.agents.serialization import (
    load_model_with_config,
    save_model_with_config,
)


class PPOAgent(BaseAgent):
    """
    PPO Agent wrapper for trading environments.

    Wraps Stable-Baselines3 PPO with trading-specific defaults
    and convenience methods. Provides config-driven training,
    model serialization, and integration with custom callbacks.

    Attributes:
        model: Underlying SB3 PPO model
        config: PPO hyperparameters
        training_config: Training process configuration
        agent_config: Full agent configuration

    Example:
        >>> from tradebox.env import TradingEnv, EnvConfig
        >>> from tradebox.agents import PPOAgent, PPOConfig, TrainingConfig
        >>>
        >>> # Create environment and configs
        >>> env = TradingEnv(data, features, EnvConfig())
        >>> ppo_config = PPOConfig(learning_rate=0.0003)
        >>> training_config = TrainingConfig(total_timesteps=2000000)
        >>>
        >>> # Create and train agent
        >>> agent = PPOAgent(env, ppo_config, training_config)
        >>> agent.train()
        >>>
        >>> # Save and load
        >>> agent.save("models/ppo_trading")
        >>> loaded_agent = PPOAgent.load("models/ppo_trading", env=env)
        >>>
        >>> # Predict
        >>> obs, info = env.reset()
        >>> action, _ = loaded_agent.predict(obs)
    """

    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        config: Optional[PPOConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        env_factory: Optional[Callable[[], gym.Env]] = None,
    ) -> None:
        """
        Initialize PPO agent.

        Args:
            env: Gymnasium environment or vectorized environment.
                 If a single environment is passed and n_envs > 1 in
                 training_config, env_factory must be provided.
            config: PPO hyperparameters (uses defaults if None).
            training_config: Training configuration (uses defaults if None).
            env_factory: Optional factory function to create environments
                        for vectorization. Required if n_envs > 1 and
                        env is not already vectorized.

        Raises:
            ValueError: If env is invalid or env_factory is needed but not provided.
        """
        self.config = config or PPOConfig()
        self.training_config = training_config or TrainingConfig()
        self.agent_config = AgentConfig(
            algorithm="PPO",
            ppo=self.config,
            training=self.training_config,
        )

        # Store original env and factory
        self._single_env = env if not isinstance(env, VecEnv) else None
        self._env_factory = env_factory

        # Create vectorized environment if needed
        if isinstance(env, VecEnv):
            vec_env = env
            logger.debug(f"Using provided VecEnv with {env.num_envs} environments")
        elif self.training_config.n_envs > 1:
            if env_factory is None:
                raise ValueError(
                    f"env_factory must be provided when n_envs > 1 "
                    f"(got n_envs={self.training_config.n_envs})"
                )
            vec_env = self._create_vectorized_env(
                env_factory, self.training_config.n_envs
            )
        else:
            vec_env = DummyVecEnv([lambda: env])
            logger.debug("Wrapped single environment in DummyVecEnv")

        # Build policy kwargs
        policy_kwargs = self._build_policy_kwargs()

        # ALWAYS use MultiInputPolicy since environment now uses Dict observation space
        # The use_cnn_extractor flag controls which features_extractor is used,
        # not the policy type (MultiInputPolicy works with both CNN and MLP extractors)
        policy_type = "MultiInputPolicy"

        # Create PPO model
        self.model = PPO(
            policy=policy_type,
            env=vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            normalize_advantage=self.config.normalize_advantage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.training_config.tensorboard_log,
            seed=self.training_config.seed,
            device=self.training_config.device,
            verbose=self.training_config.verbose,
        )

        logger.info(
            f"PPOAgent initialized: lr={self.config.learning_rate}, "
            f"arch={self.config.network_arch}, n_envs={vec_env.num_envs}"
        )

    def _create_vectorized_env(
        self,
        env_factory: Callable[[], gym.Env],
        n_envs: int,
        use_subprocess: bool = True,
    ) -> VecEnv:
        """
        Create vectorized environment for parallel training.

        Args:
            env_factory: Factory function that creates environments.
            n_envs: Number of parallel environments.
            use_subprocess: If True, use SubprocVecEnv (faster but more memory).
                           If False, use DummyVecEnv (sequential but less memory).

        Returns:
            Vectorized environment.
        """
        env_fns = [env_factory for _ in range(n_envs)]

        if use_subprocess:
            try:
                vec_env = SubprocVecEnv(env_fns)
                logger.info(f"Created SubprocVecEnv with {n_envs} environments")
            except Exception as e:
                logger.warning(
                    f"Failed to create SubprocVecEnv: {e}. "
                    "Falling back to DummyVecEnv."
                )
                vec_env = DummyVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)
            logger.info(f"Created DummyVecEnv with {n_envs} environments")

        return vec_env

    def _build_policy_kwargs(self) -> Dict[str, Any]:
        """
        Build policy keyword arguments from config.

        Supports both MLP-only (default) and CNN-based feature extraction.
        When use_cnn_extractor=True, the observation space must be a Dict
        with keys: "price", "indicators", "portfolio".

        Returns:
            Dictionary with net_arch, activation_fn, and optional feature extractor.

        Raises:
            ImportError: If CNN extractor is requested but import fails.
        """
        # Get activation function
        if self.config.activation_fn == "tanh":
            activation_fn = th.nn.Tanh
        elif self.config.activation_fn == "relu":
            activation_fn = th.nn.ReLU
        else:
            activation_fn = th.nn.Tanh

        policy_kwargs = {
            "net_arch": dict(
                pi=self.config.network_arch.copy(),
                vf=self.config.network_arch.copy(),
            ),
            "activation_fn": activation_fn,
        }

        # Add CNN feature extractor if requested
        if self.config.use_cnn_extractor:
            if self.config.extractor_type == "trading":
                # Use new TradingCNNExtractor (production-ready, attention-based)
                try:
                    from tradebox.models import TradingCNNExtractor
                except ImportError as e:
                    raise ImportError(
                        "Failed to import TradingCNNExtractor. "
                        "Ensure tradebox.models.trading_cnn_extractor is available."
                    ) from e

                # Calculate total features dimension (includes optional fundamentals)
                features_dim = (
                    self.config.price_embed_dim +
                    self.config.ind_embed_dim +
                    self.config.port_embed_dim +
                    self.config.fund_embed_dim  # Optional fundamentals for EOD
                )

                policy_kwargs["features_extractor_class"] = TradingCNNExtractor
                policy_kwargs["features_extractor_kwargs"] = {
                    "features_dim": features_dim,
                    "cnn_embedding_dim": self.config.price_embed_dim,
                    "indicator_embedding_dim": self.config.ind_embed_dim,
                    "portfolio_embedding_dim": self.config.port_embed_dim,
                    "fundamental_embedding_dim": self.config.fund_embed_dim,
                    "use_attention": self.config.use_attention,
                }

                logger.info(
                    f"Using TradingCNNExtractor with attention={self.config.use_attention}, "
                    f"features_dim={features_dim}"
                )

            else:
                # Use existing HybridCNNExtractor
                try:
                    from tradebox.models.hybrid_extractor import HybridCNNExtractor
                except ImportError as e:
                    raise ImportError(
                        "Failed to import HybridCNNExtractor. "
                        "Ensure tradebox.models.hybrid_extractor is available."
                    ) from e

                # Calculate features_dim from embeddings
                if self.config.use_fusion:
                    # With fusion layer, use sum of embeddings (typical choice)
                    features_dim = (
                        self.config.price_embed_dim +
                        self.config.ind_embed_dim +
                        self.config.port_embed_dim
                    )
                else:
                    # Without fusion, concatenation dimension
                    features_dim = (
                        self.config.price_embed_dim +
                        self.config.ind_embed_dim +
                        self.config.port_embed_dim
                    )

                policy_kwargs["features_extractor_class"] = HybridCNNExtractor
                policy_kwargs["features_extractor_kwargs"] = {
                    "features_dim": features_dim,
                    "cnn_type": self.config.cnn_type,
                    "price_embed_dim": self.config.price_embed_dim,
                    "ind_embed_dim": self.config.ind_embed_dim,
                    "port_embed_dim": self.config.port_embed_dim,
                    "use_fusion": self.config.use_fusion,
                    "dropout": self.config.cnn_dropout,
                }

                logger.info(
                    f"Using HybridCNNExtractor with cnn_type={self.config.cnn_type}, "
                    f"features_dim={features_dim}"
                )
        else:
            logger.debug("Using default MLP feature extractor")

        logger.debug(f"Policy kwargs: {policy_kwargs}")
        return policy_kwargs

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
        progress_bar: bool = True,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "PPO",
    ) -> "PPOAgent":
        """
        Train the PPO agent.

        Args:
            total_timesteps: Override total timesteps from config.
            callback: Callback(s) to use during training.
            progress_bar: Show training progress bar.
            reset_num_timesteps: Reset timestep counter.
            tb_log_name: Name for TensorBoard logs.

        Returns:
            Self for method chaining.

        Example:
            >>> agent.train(total_timesteps=1000000, callback=eval_callback)
            >>> # Or with chaining
            >>> agent.train().save("models/trained")
        """
        timesteps = total_timesteps or self.training_config.total_timesteps

        logger.info(
            f"Starting training for {timesteps:,} timesteps "
            f"(log_interval={self.training_config.log_interval})"
        )

        self.model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=self.training_config.log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        logger.info(
            f"Training complete. Total timesteps: {self.model.num_timesteps:,}"
        )

        return self

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: bool = True,
        episode_start: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for observation with optional action masking.

        Args:
            observation: Current observation from environment.
            state: Hidden state (for recurrent policies, None for MLP).
            deterministic: If True, use mean action (no exploration).
            episode_start: Whether episode started (for recurrent).
            action_mask: Optional mask for invalid actions (regime-based constraints).
                        Boolean array where True = action allowed, False = action masked.
                        If provided and predicted action is invalid, samples from valid actions.

        Returns:
            Tuple of (action, next_state).

        Example:
            >>> obs, info = env.reset()
            >>> action, _ = agent.predict(obs, deterministic=True)
            >>> obs, reward, done, truncated, info = env.step(action)

            >>> # With action masking
            >>> action_mask = info.get("action_mask", None)
            >>> action, _ = agent.predict(obs, deterministic=True, action_mask=action_mask)
        """
        # Get action from SB3 model (standard predict doesn't support action_masks)
        action, state = self.model.predict(
            observation,
            state=state,
            deterministic=deterministic,
            episode_start=episode_start,
        )

        # Apply action masking if provided
        # SB3's standard PPO doesn't support action_masks in predict(),
        # so we manually enforce the mask by resampling if needed
        if action_mask is not None:
            # Handle both scalar and array actions
            if np.isscalar(action) or action.shape == ():
                action_int = int(action)
                # If predicted action is invalid, sample from valid actions
                if not action_mask[action_int]:
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        # Randomly sample from valid actions
                        action = np.random.choice(valid_actions)
                        logger.debug(
                            f"Action {action_int} masked, resampled to {action} "
                            f"from valid actions {valid_actions}"
                        )
                    else:
                        logger.warning(
                            "No valid actions in mask! Using original action."
                        )
            else:
                # Vectorized environment case
                for i in range(len(action)):
                    action_int = int(action[i])
                    mask = action_mask[i] if action_mask.ndim > 1 else action_mask
                    if not mask[action_int]:
                        valid_actions = np.where(mask)[0]
                        if len(valid_actions) > 0:
                            action[i] = np.random.choice(valid_actions)

        return action, state

    def save(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save agent to disk.

        Saves:
        - Model weights and optimizer state (.zip)
        - Configuration (_config.json)
        - Training metadata (_metadata.json)

        Args:
            path: Path to save model (without extension).
            metadata: Additional metadata to save.

        Example:
            >>> agent.save("models/ppo_best")
            # Creates: models/ppo_best.zip, models/ppo_best_config.json, etc.
        """
        save_model_with_config(
            model=self.model,
            path=path,
            config=self.agent_config,
            metadata=metadata,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[Union[gym.Env, VecEnv]] = None,
        device: str = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "PPOAgent":
        """
        Load agent from disk.

        Args:
            path: Path to saved model (without extension).
            env: Environment to use (optional, can be None for inference).
            device: Device to load model on.
            custom_objects: Custom objects for unpickling.
            **kwargs: Additional arguments.

        Returns:
            Loaded PPOAgent instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.

        Example:
            >>> agent = PPOAgent.load("models/ppo_best", env=eval_env)
            >>> action, _ = agent.predict(obs)
        """
        model, config, metadata = load_model_with_config(
            path=path,
            algorithm_class=PPO,
            env=env,
            device=device,
        )

        # Create agent instance without reinitializing model
        agent = cls.__new__(cls)
        agent.model = model
        agent.config = config.ppo
        agent.training_config = config.training
        agent.agent_config = config
        agent._single_env = env if env and not isinstance(env, VecEnv) else None
        agent._env_factory = None

        logger.info(
            f"Loaded PPOAgent from {path} "
            f"(trained for {metadata.get('num_timesteps', 'unknown')} timesteps)"
        )

        return agent

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get agent hyperparameters.

        Returns:
            Dictionary with all configuration parameters.
        """
        return {
            "algorithm": "PPO",
            "learning_rate": self.config.learning_rate,
            "n_steps": self.config.n_steps,
            "batch_size": self.config.batch_size,
            "n_epochs": self.config.n_epochs,
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
            "clip_range": self.config.clip_range,
            "clip_range_vf": self.config.clip_range_vf,
            "ent_coef": self.config.ent_coef,
            "vf_coef": self.config.vf_coef,
            "max_grad_norm": self.config.max_grad_norm,
            "network_arch": self.config.network_arch,
            "activation_fn": self.config.activation_fn,
            "normalize_advantage": self.config.normalize_advantage,
            "total_timesteps": self.training_config.total_timesteps,
            "n_envs": self.training_config.n_envs,
        }

    @property
    def policy(self) -> Any:
        """Get the underlying policy network."""
        return self.model.policy

    @property
    def num_timesteps(self) -> int:
        """Get total timesteps trained."""
        return self.model.num_timesteps

    def evaluate(
        self,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent on environment.

        Args:
            env: Environment to evaluate on.
            n_eval_episodes: Number of evaluation episodes.
            deterministic: Use deterministic policy.

        Returns:
            Dictionary with mean_reward, std_reward, mean_ep_length.

        Example:
            >>> metrics = agent.evaluate(test_env, n_eval_episodes=20)
            >>> print(f"Mean reward: {metrics['mean_reward']:.2f}")
        """
        # Wrap in VecEnv if needed
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
        )

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_length = float(np.mean(episode_lengths))

        # Calculate additional metrics
        returns = np.array(episode_rewards)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float((np.mean(returns) / np.std(returns)) * np.sqrt(252))

        metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_ep_length": mean_length,
            "sharpe_ratio": sharpe,
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "n_episodes": n_eval_episodes,
        }

        logger.info(
            f"Evaluation: mean_reward={mean_reward:.2f} (+/- {std_reward:.2f}), "
            f"sharpe={sharpe:.3f}"
        )

        return metrics

    def get_env(self) -> Optional[VecEnv]:
        """
        Get the training environment.

        Returns:
            The vectorized environment attached to this agent.
        """
        return self.model.get_env()

    def set_env(self, env: Union[gym.Env, VecEnv]) -> None:
        """
        Set the training environment.

        Args:
            env: Environment to attach.
        """
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        self.model.set_env(env)
