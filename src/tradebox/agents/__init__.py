"""
RL Agents for TradeBox trading system.

This module provides reinforcement learning agents for automated trading,
currently supporting PPO (Proximal Policy Optimization) from Stable-Baselines3
with plans for SAC (Soft Actor-Critic) in future phases.

Key Components:
    - PPOAgent: Main PPO agent wrapper with trading-specific features
    - AgentTrainer: High-level training orchestrator
    - Callbacks: Trading-specific callbacks for evaluation and checkpointing
    - Serialization: Model save/load utilities with configuration

Configuration:
    - PPOConfig: PPO hyperparameters (learning rate, network arch, etc.)
    - TrainingConfig: Training process settings (timesteps, envs, logging)
    - AgentConfig: Combined configuration for agents

Example:
    >>> from tradebox.agents import PPOAgent, PPOConfig, TrainingConfig
    >>> from tradebox.env import TradingEnv, EnvConfig
    >>>
    >>> # Create environment
    >>> env = TradingEnv(data, features, EnvConfig())
    >>>
    >>> # Configure and create agent
    >>> ppo_config = PPOConfig(learning_rate=0.0003, network_arch=[256, 256])
    >>> training_config = TrainingConfig(total_timesteps=2000000, n_envs=8)
    >>> agent = PPOAgent(env, ppo_config, training_config)
    >>>
    >>> # Train with callbacks
    >>> from tradebox.agents import create_callback_list
    >>> callbacks = create_callback_list(eval_env, training_config)
    >>> agent.train(callback=callbacks)
    >>>
    >>> # Save and load
    >>> agent.save("models/ppo_trading")
    >>> loaded = PPOAgent.load("models/ppo_trading", env=env)
    >>>
    >>> # Predict
    >>> obs, info = env.reset()
    >>> action, _ = loaded.predict(obs, deterministic=True)

Using AgentTrainer (config-driven):
    >>> from tradebox.agents import AgentTrainer
    >>>
    >>> trainer = AgentTrainer.from_config(
    ...     "configs/experiments/exp001_baseline.yaml",
    ...     train_data=train_df,
    ...     train_features=train_features,
    ...     eval_data=val_df,
    ...     eval_features=val_features,
    ... )
    >>> results = trainer.train()
    >>> trainer.save_final_model("models/exp001_final")
"""

from tradebox.agents.config import (
    AgentConfig,
    PPOConfig,
    TrainingConfig,
)
from tradebox.agents.base_agent import BaseAgent
from tradebox.agents.ppo_agent import PPOAgent
from tradebox.agents.callbacks import (
    EarlyStoppingCallback,
    TradingCheckpointCallback,
    TradingEvalCallback,
    TradingMetricsCallback,
    create_callback_list,
)
from tradebox.agents.trainer import AgentTrainer
from tradebox.agents.serialization import (
    delete_model,
    get_model_info,
    list_saved_models,
    load_model_with_config,
    save_model_with_config,
)

__all__ = [
    # Configuration
    "AgentConfig",
    "PPOConfig",
    "TrainingConfig",
    # Agents
    "BaseAgent",
    "PPOAgent",
    # Callbacks
    "EarlyStoppingCallback",
    "TradingCheckpointCallback",
    "TradingEvalCallback",
    "TradingMetricsCallback",
    "create_callback_list",
    # Training
    "AgentTrainer",
    # Serialization
    "delete_model",
    "get_model_info",
    "list_saved_models",
    "load_model_with_config",
    "save_model_with_config",
]
