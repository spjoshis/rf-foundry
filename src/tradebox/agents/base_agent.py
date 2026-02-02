"""Abstract base class for RL trading agents."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for RL trading agents.

    Defines the interface that all agents must implement,
    enabling easy swapping between algorithms (PPO, SAC, etc.).

    This class provides a consistent API for:
    - Training agents on trading environments
    - Making predictions (selecting actions)
    - Saving and loading trained models
    - Evaluating agent performance

    Example:
        >>> # Using a concrete implementation (PPOAgent)
        >>> from tradebox.agents import PPOAgent, PPOConfig
        >>> agent = PPOAgent(env, PPOConfig())
        >>> agent.train(total_timesteps=1000000)
        >>> action, _ = agent.predict(observation)
        >>> agent.save("models/my_agent")
        >>> loaded_agent = PPOAgent.load("models/my_agent", env=env)
    """

    @abstractmethod
    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[Any] = None,
        progress_bar: bool = True,
    ) -> "BaseAgent":
        """
        Train the agent on the environment.

        Args:
            total_timesteps: Total timesteps to train. If None, uses config value.
            callback: Callback(s) to use during training. Can be a single callback
                     or list of callbacks.
            progress_bar: Whether to display a progress bar during training.

        Returns:
            Self for method chaining.

        Example:
            >>> agent.train(total_timesteps=500000)
            >>> agent.train(callback=eval_callback, progress_bar=True)
        """
        pass

    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for given observation.

        Args:
            observation: Current observation from environment. Shape depends on
                        environment's observation space.
            state: Hidden state for recurrent policies. None for MLP policies.
            deterministic: If True, use mean action (no exploration noise).
                          If False, sample from action distribution.

        Returns:
            Tuple of:
            - action: Selected action(s). Shape depends on environment's action space.
            - state: Next hidden state for recurrent policies. None for MLP policies.

        Example:
            >>> obs, info = env.reset()
            >>> action, _ = agent.predict(obs, deterministic=True)
            >>> obs, reward, done, truncated, info = env.step(action)
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save agent to disk.

        Saves:
        - Model weights and optimizer state
        - Configuration (hyperparameters)
        - Training metadata (timesteps, etc.)

        The exact file format depends on the implementation.
        Typically creates a .zip file for the model and JSON for config.

        Args:
            path: Path to save model (without extension). The implementation
                 may add appropriate extensions.

        Example:
            >>> agent.save("models/ppo_best")
            # Creates: models/ppo_best.zip, models/ppo_best_config.json
        """
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        **kwargs: Any,
    ) -> "BaseAgent":
        """
        Load agent from disk.

        Args:
            path: Path to saved model (without extension).
            env: Environment to attach to loaded agent. Can be None for
                inference-only usage.
            **kwargs: Additional arguments passed to the underlying load function.

        Returns:
            Loaded agent instance.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If saved model is incompatible.

        Example:
            >>> agent = PPOAgent.load("models/ppo_best", env=eval_env)
            >>> action, _ = agent.predict(obs)
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get agent hyperparameters.

        Returns:
            Dictionary containing all configuration parameters used to
            create and train this agent.

        Example:
            >>> params = agent.get_parameters()
            >>> print(f"Learning rate: {params['learning_rate']}")
        """
        pass

    @property
    @abstractmethod
    def policy(self) -> Any:
        """
        Get the underlying policy network.

        Returns:
            The policy object from the underlying RL library.
            For Stable-Baselines3, this is the policy attribute.
        """
        pass

    @property
    @abstractmethod
    def num_timesteps(self) -> int:
        """
        Get total timesteps trained.

        Returns:
            Total number of environment steps the agent has been trained for.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        env: gym.Env,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent on environment.

        Runs the agent for n_eval_episodes and collects performance metrics.

        Args:
            env: Environment to evaluate on. Should be compatible with
                the agent's observation and action spaces.
            n_eval_episodes: Number of complete episodes to run.
            deterministic: Use deterministic policy (no exploration).

        Returns:
            Dictionary containing:
            - mean_reward: Average episode reward
            - std_reward: Standard deviation of episode rewards
            - mean_ep_length: Average episode length
            - Additional algorithm-specific metrics

        Example:
            >>> metrics = agent.evaluate(test_env, n_eval_episodes=20)
            >>> print(f"Mean reward: {metrics['mean_reward']:.2f}")
            >>> print(f"Std reward: {metrics['std_reward']:.2f}")
        """
        pass

    def get_env(self) -> Optional[gym.Env]:
        """
        Get the training environment.

        Returns:
            The environment attached to this agent, or None if not set.
        """
        return getattr(self, "_env", None)

    def set_env(self, env: gym.Env) -> None:
        """
        Set the training environment.

        Args:
            env: Environment to attach to this agent.
        """
        self._env = env
