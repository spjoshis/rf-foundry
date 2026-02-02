"""High-level training orchestrator for RL trading agents."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import pandas as pd
import yaml
from loguru import logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from tradebox.agents.callbacks import create_callback_list
from tradebox.agents.config import AgentConfig, PPOConfig, TrainingConfig
from tradebox.agents.ppo_agent import PPOAgent


class AgentTrainer:
    """
    High-level training orchestrator for RL trading agents.

    Handles:
    - Config loading from YAML files
    - Environment creation and vectorization
    - Agent initialization with proper configurations
    - Training with callbacks (evaluation, checkpointing, metrics)
    - Model saving and experiment logging

    The trainer provides a clean interface between configuration files
    and the training loop, abstracting away the complexity of setting
    up environments, callbacks, and logging.

    Attributes:
        agent_config: Combined agent and training configuration
        agent: The RL agent (created during training)
        train_env: Vectorized training environment
        eval_env: Evaluation environment
        experiment_name: Name for logging and saving

    Example:
        >>> # From YAML config
        >>> trainer = AgentTrainer.from_config(
        ...     "configs/experiments/exp001_baseline.yaml",
        ...     train_data=train_df,
        ...     train_features=train_features,
        ...     eval_data=val_df,
        ...     eval_features=val_features,
        ... )
        >>> results = trainer.train()
        >>> trainer.save_final_model("models/exp001_final")
        >>>
        >>> # Programmatic setup
        >>> trainer = AgentTrainer(
        ...     agent_config=AgentConfig(),
        ...     train_env_factory=lambda: TradingEnv(data, features, config),
        ...     eval_env_factory=lambda: TradingEnv(eval_data, eval_features, config),
        ...     experiment_name="my_experiment",
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        train_env_factory: Callable[[], gym.Env],
        eval_env_factory: Optional[Callable[[], gym.Env]] = None,
        experiment_name: str = "experiment",
        env_config: Optional[Any] = None,
    ) -> None:
        """
        Initialize trainer.

        Args:
            agent_config: Agent and training configuration.
            train_env_factory: Factory function that creates training environments.
                              Called multiple times for vectorized training.
            eval_env_factory: Factory function to create evaluation environment.
                             If None, uses train_env_factory.
            experiment_name: Name for logging, TensorBoard, and model saving.
            env_config: Optional environment configuration object.
        """
        self.agent_config = agent_config
        self.train_env_factory = train_env_factory
        self.eval_env_factory = eval_env_factory or train_env_factory
        self.experiment_name = experiment_name
        self.env_config = env_config

        # Will be created during training
        self.agent: Optional[PPOAgent] = None
        self.train_env: Optional[VecEnv] = None
        self.eval_env: Optional[VecEnv] = None
        self._training_results: Dict[str, Any] = {}

        logger.info(
            f"AgentTrainer initialized: experiment={experiment_name}, "
            f"algorithm={agent_config.algorithm}, "
            f"n_envs={agent_config.training.n_envs}"
        )

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        train_data: pd.DataFrame,
        train_features: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
        eval_features: Optional[pd.DataFrame] = None,
        env_config: Optional[Any] = None,
    ) -> "AgentTrainer":
        """
        Create trainer from YAML configuration file.

        Loads configuration from YAML and creates environment factories
        using the provided data and features.

        Args:
            config_path: Path to experiment YAML config.
            train_data: Training OHLCV DataFrame.
            train_features: Training features DataFrame.
            eval_data: Evaluation OHLCV data. If None, uses train_data.
            eval_features: Evaluation features. If None, uses train_features.
            env_config: Optional environment config override.

        Returns:
            Configured AgentTrainer instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.

        Example:
            >>> trainer = AgentTrainer.from_config(
            ...     "configs/experiments/exp001_baseline.yaml",
            ...     train_data=splits["train"],
            ...     train_features=train_features,
            ...     eval_data=splits["validation"],
            ...     eval_features=val_features,
            ... )
        """
        # Import here to avoid circular imports
        from tradebox.env import EnvConfig, TradingEnv

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Parse agent config
        agent_cfg = cfg.get("agent", {})
        training_cfg = cfg.get("training", {})
        env_cfg = cfg.get("env", {})
        experiment_cfg = cfg.get("experiment", {})

        # Build PPOConfig
        ppo_config = PPOConfig(
            learning_rate=agent_cfg.get("learning_rate", 0.0003),
            n_steps=agent_cfg.get("n_steps", 2048),
            batch_size=agent_cfg.get("batch_size", 64),
            n_epochs=agent_cfg.get("n_epochs", 10),
            gamma=agent_cfg.get("gamma", 0.99),
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_range=agent_cfg.get("clip_range", 0.2),
            network_arch=agent_cfg.get("network_arch", [256, 256]),
            activation_fn=agent_cfg.get("activation_fn", "tanh"),
        )

        # Handle nested ppo config
        if "ppo" in agent_cfg:
            ppo_nested = agent_cfg["ppo"]
            ppo_config = PPOConfig(
                learning_rate=ppo_nested.get("learning_rate", ppo_config.learning_rate),
                n_steps=ppo_nested.get("n_steps", ppo_config.n_steps),
                batch_size=ppo_nested.get("batch_size", ppo_config.batch_size),
                n_epochs=ppo_nested.get("n_epochs", ppo_config.n_epochs),
                gamma=ppo_nested.get("gamma", ppo_config.gamma),
                gae_lambda=ppo_nested.get("gae_lambda", ppo_config.gae_lambda),
                clip_range=ppo_nested.get("clip_range", ppo_config.clip_range),
                ent_coef=ppo_nested.get("ent_coef", ppo_config.ent_coef),
                vf_coef=ppo_nested.get("vf_coef", ppo_config.vf_coef),
                max_grad_norm=ppo_nested.get("max_grad_norm", ppo_config.max_grad_norm),
                network_arch=ppo_nested.get("network_arch", ppo_config.network_arch),
                activation_fn=ppo_nested.get("activation_fn", ppo_config.activation_fn),
                # CNN-related parameters
                use_cnn_extractor=ppo_nested.get("use_cnn_extractor", ppo_config.use_cnn_extractor),
                extractor_type=ppo_nested.get("extractor_type", ppo_config.extractor_type),
                cnn_type=ppo_nested.get("cnn_type", ppo_config.cnn_type),
                price_embed_dim=ppo_nested.get("price_embed_dim", ppo_config.price_embed_dim),
                ind_embed_dim=ppo_nested.get("ind_embed_dim", ppo_config.ind_embed_dim),
                port_embed_dim=ppo_nested.get("port_embed_dim", ppo_config.port_embed_dim),
                fund_embed_dim=ppo_nested.get("fund_embed_dim", ppo_config.fund_embed_dim),
                cnn_dropout=ppo_nested.get("cnn_dropout", ppo_config.cnn_dropout),
                use_fusion=ppo_nested.get("use_fusion", ppo_config.use_fusion),
                use_attention=ppo_nested.get("use_attention", ppo_config.use_attention),
            )

        # Build TrainingConfig
        training_config = TrainingConfig(
            total_timesteps=training_cfg.get("total_timesteps", 2000000),
            n_envs=training_cfg.get("n_envs", 8),
            eval_freq=training_cfg.get("eval_freq", 10000),
            n_eval_episodes=training_cfg.get("n_eval_episodes", 5),
            checkpoint_freq=training_cfg.get("checkpoint_freq", 50000),
            seed=training_cfg.get("seed"),
            device=training_cfg.get("device", "auto"),
            tensorboard_log=training_cfg.get("tensorboard_log", "logs/tensorboard"),
            model_save_dir=training_cfg.get("model_save_dir", "models"),
            best_model_save_path=training_cfg.get("best_model_save_path", "models/best"),
            verbose=training_cfg.get("verbose", 1),
        )

        # Build AgentConfig
        agent_config = AgentConfig(
            algorithm=agent_cfg.get("algorithm", "PPO"),
            ppo=ppo_config,
            training=training_config,
        )

        # Build EnvConfig
        if env_config is None:
            from tradebox.env.costs import CostConfig
            from tradebox.env.rewards import RewardConfig

            # Extract reward config from YAML (nested under env.reward_config)
            reward_cfg = env_cfg.get("reward_config", {})

            # Determine environment type (intraday vs EOD)
            env_type = env_cfg.get("type", "eod")
            is_intraday = env_type == "intraday"

            # Build cost config with all parameters
            cost_cfg_dict = env_cfg.get("cost_config", {})
            cost_config = CostConfig(
                slippage_pct=cost_cfg_dict.get("slippage_pct", 0.001),
                use_dynamic_slippage=cost_cfg_dict.get("use_dynamic_slippage", False),
                base_spread_bps=cost_cfg_dict.get("base_spread_bps", 2.5),
                spread_volatility_multiplier=cost_cfg_dict.get("spread_volatility_multiplier", False),
                max_spread_bps=cost_cfg_dict.get("max_spread_bps", 10.0),
                impact_coefficient=cost_cfg_dict.get("impact_coefficient", 0.2),
                max_impact_pct=cost_cfg_dict.get("max_impact_pct", 0.005),
                brokerage_pct=cost_cfg_dict.get("brokerage_pct", 0.0003),
                brokerage_cap=cost_cfg_dict.get("brokerage_cap", 20.0),
                stt_pct=cost_cfg_dict.get("stt_pct", 0.001),
                transaction_charges_pct=cost_cfg_dict.get("transaction_charges_pct", 0.0000325),
                gst_rate=cost_cfg_dict.get("gst_rate", 0.18),
                stamp_duty_pct=cost_cfg_dict.get("stamp_duty_pct", 0.00015),
            )

            # Build reward config
            reward_config = RewardConfig(
                reward_type=reward_cfg.get("reward_type", "risk_adjusted"),
                drawdown_penalty=reward_cfg.get("drawdown_penalty", 0.5),
                trade_penalty=reward_cfg.get("trade_penalty", 0.001),
                sharpe_window=reward_cfg.get("sharpe_window", 20),
                sortino_window=reward_cfg.get("sortino_window", 20),
                risk_free_rate=reward_cfg.get("risk_free_rate", 0.06),
                volatility_penalty=reward_cfg.get("volatility_penalty", 0.1),
                enhanced_drawdown_penalty=reward_cfg.get("enhanced_drawdown_penalty", 2.0),
                calmar_window=reward_cfg.get("calmar_window", 60),
                # Active trading parameters for frequent trading
                max_holding_days=reward_cfg.get("max_holding_days", 3),
                holding_penalty=reward_cfg.get("holding_penalty", 0.001),
                trade_completion_bonus=reward_cfg.get("trade_completion_bonus", 0.002),
                inactivity_penalty=reward_cfg.get("inactivity_penalty", 0.0005),
                inactivity_threshold=reward_cfg.get("inactivity_threshold", 5),
            )

            # Create appropriate config type
            if is_intraday:
                from tradebox.env.trading_env import IntradayEnvConfig
                env_config = IntradayEnvConfig(
                    initial_capital=env_cfg.get("initial_capital", 100000.0),
                    lookback_window=env_cfg.get("lookback_window", 60),
                    max_episode_steps=env_cfg.get("max_episode_steps", 500),
                    cost_config=cost_config,
                    reward_config=reward_config,
                    bar_interval_minutes=env_cfg.get("bar_interval_minutes", 5),
                    bars_per_session=env_cfg.get("bars_per_session", 75),
                    sessions_per_episode=env_cfg.get("sessions_per_episode", 10),
                    force_close_eod=env_cfg.get("force_close_eod", True),
                    market_open_time=env_cfg.get("market_open_time", "09:15"),
                    market_close_time=env_cfg.get("market_close_time", "15:30"),
                    overnight_gap_handling=env_cfg.get("overnight_gap_handling", "reset_observation"),
                )
            else:
                env_config = EnvConfig(
                    initial_capital=env_cfg.get("initial_capital", 100000.0),
                    lookback_window=env_cfg.get("lookback_window", 60),
                    max_episode_steps=env_cfg.get("max_episode_steps", 500),
                    cost_config=cost_config,
                    reward_config=reward_config,
                )

        # Use provided eval data or fallback to train data
        _eval_data = eval_data if eval_data is not None else train_data
        _eval_features = eval_features if eval_features is not None else train_features

        # Determine which environment class to use
        env_type = env_cfg.get("type", "eod")
        if env_type == "intraday":
            from tradebox.env.intraday_env import IntradayTradingEnv
            EnvClass = IntradayTradingEnv
        else:
            EnvClass = TradingEnv

        # Create eval config with adjusted max_episode_steps if needed
        eval_env_config = env_config
        if env_type == "intraday" and eval_data is not None:
            # For eval environment, adjust max_episode_steps to fit available data
            # Need: lookback_window + max_episode_steps + bars_per_session (for random start)
            available_bars = len(_eval_data)
            bars_per_session = env_config.bars_per_session
            min_required_bars = env_config.lookback_window + env_config.max_episode_steps + bars_per_session

            if available_bars < min_required_bars:
                # Calculate adjusted max_episode_steps
                # Reserve: lookback_window + 1 session for random start
                adjusted_max_steps = available_bars - env_config.lookback_window - bars_per_session
                # Round down to nearest session boundary
                adjusted_sessions = max(1, adjusted_max_steps // bars_per_session)
                adjusted_max_steps = adjusted_sessions * bars_per_session

                logger.info(
                    f"Adjusting eval env max_episode_steps: {env_config.max_episode_steps} -> {adjusted_max_steps} "
                    f"(available bars: {available_bars}, sessions: {adjusted_sessions})"
                )

                # Create adjusted config for eval
                from tradebox.env.trading_env import IntradayEnvConfig
                eval_env_config = IntradayEnvConfig(
                    initial_capital=env_config.initial_capital,
                    lookback_window=env_config.lookback_window,
                    max_episode_steps=adjusted_max_steps,
                    cost_config=env_config.cost_config,
                    reward_config=env_config.reward_config,
                    bar_interval_minutes=env_config.bar_interval_minutes,
                    bars_per_session=env_config.bars_per_session,
                    sessions_per_episode=adjusted_sessions,
                    force_close_eod=env_config.force_close_eod,
                    market_open_time=env_config.market_open_time,
                    market_close_time=env_config.market_close_time,
                    overnight_gap_handling=env_config.overnight_gap_handling,
                )

        # Create environment factories
        # Note: TradingEnv already provides Dict observation space for CNN compatibility
        def train_env_factory() -> gym.Env:
            return EnvClass(train_data, train_features, env_config)

        def eval_env_factory() -> gym.Env:
            return EnvClass(_eval_data, _eval_features, eval_env_config)

        experiment_name = experiment_cfg.get("name", config_path.stem)

        logger.info(f"Loaded config from {config_path}: {experiment_name}")

        return cls(
            agent_config=agent_config,
            train_env_factory=train_env_factory,
            eval_env_factory=eval_env_factory,
            experiment_name=experiment_name,
            env_config=env_config,
        )

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        train_env_factory: Callable[[], gym.Env],
        eval_env_factory: Optional[Callable[[], gym.Env]] = None,
        env_config: Optional[Any] = None,
    ) -> "AgentTrainer":
        """
        Create trainer from configuration dictionary.

        Args:
            config: Configuration dictionary with 'agent' and 'training' keys.
            train_env_factory: Factory function for training environments.
            eval_env_factory: Factory function for evaluation environment.
            env_config: Optional environment configuration object.

        Returns:
            Configured AgentTrainer instance.
        """
        agent_cfg = config.get("agent", {})
        training_cfg = config.get("training", {})

        ppo_config = PPOConfig(**agent_cfg.get("ppo", {}))
        training_config = TrainingConfig(**training_cfg)
        agent_config = AgentConfig(
            algorithm=agent_cfg.get("algorithm", "PPO"),
            ppo=ppo_config,
            training=training_config,
        )

        return cls(
            agent_config=agent_config,
            train_env_factory=train_env_factory,
            eval_env_factory=eval_env_factory,
            experiment_name=config.get("experiment", {}).get("name", "experiment"),
            env_config=env_config,
        )

    def _create_environments(self) -> None:
        """Create training and evaluation environments."""
        training_config = self.agent_config.training

        # Create vectorized training environment
        if training_config.n_envs > 1:
            try:
                self.train_env = SubprocVecEnv(
                    [self.train_env_factory for _ in range(training_config.n_envs)]
                )
                logger.info(
                    f"Created SubprocVecEnv with {training_config.n_envs} environments"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create SubprocVecEnv: {e}. "
                    "Falling back to DummyVecEnv."
                )
                self.train_env = DummyVecEnv(
                    [self.train_env_factory for _ in range(training_config.n_envs)]
                )
        else:
            self.train_env = DummyVecEnv([self.train_env_factory])

        # Create evaluation environment (single environment)
        self.eval_env = DummyVecEnv([self.eval_env_factory])
        logger.info("Created evaluation environment")

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run training process.

        Creates environments, sets up agent and callbacks, runs training,
        and returns results.

        Args:
            total_timesteps: Override timesteps from config.
            callbacks: Additional callbacks to add to the default list.

        Returns:
            Training results dictionary with:
            - final_timesteps: Total timesteps trained
            - best_sharpe: Best Sharpe ratio achieved
            - training_time: Total training time in seconds

        Example:
            >>> results = trainer.train()
            >>> print(f"Final Sharpe: {results['best_sharpe']:.2f}")
        """
        import time

        start_time = time.time()

        # Create environments
        self._create_environments()

        # Create agent
        self.agent = PPOAgent(
            env=self.train_env,
            config=self.agent_config.ppo,
            training_config=self.agent_config.training,
        )

        # Create callbacks
        default_callbacks = create_callback_list(
            eval_env=self.eval_env,
            training_config=self.agent_config.training,
            agent_config=self.agent_config,
        )

        all_callbacks = default_callbacks
        if callbacks:
            from stable_baselines3.common.callbacks import CallbackList
            all_callbacks = CallbackList([default_callbacks] + callbacks)

        # Train
        timesteps = total_timesteps or self.agent_config.training.total_timesteps
        self.agent.train(
            total_timesteps=timesteps,
            callback=all_callbacks,
            tb_log_name=self.experiment_name,
            progress_bar=False,  # Disable progress bar (requires tqdm/rich)
        )

        training_time = time.time() - start_time

        # Collect results
        self._training_results = {
            "experiment_name": self.experiment_name,
            "final_timesteps": self.agent.num_timesteps,
            "training_time_seconds": training_time,
            "config": self.agent.get_parameters(),
        }

        # Get best Sharpe from eval callback if available
        for cb in default_callbacks.callbacks:
            from tradebox.agents.callbacks import TradingEvalCallback
            if isinstance(cb, TradingEvalCallback):
                self._training_results["best_sharpe"] = cb.best_sharpe
                self._training_results["sharpe_history"] = cb.sharpe_history
                break

        logger.info(
            f"Training complete in {training_time:.1f}s. "
            f"Timesteps: {self.agent.num_timesteps:,}"
        )

        return self._training_results

    def save_final_model(self, path: Union[str, Path]) -> None:
        """
        Save final trained model.

        Args:
            path: Path for saved model.

        Raises:
            ValueError: If training hasn't been run yet.
        """
        if self.agent is None:
            raise ValueError("No agent to save. Run train() first.")

        self.agent.save(path, metadata=self._training_results)
        logger.info(f"Saved final model to {path}")

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics and results.

        Returns:
            Dictionary with training results (empty if not trained yet).
        """
        return self._training_results.copy()

    def evaluate(
        self,
        env: Optional[gym.Env] = None,
        n_eval_episodes: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate the trained agent.

        Args:
            env: Environment to evaluate on. If None, uses eval_env.
            n_eval_episodes: Number of episodes for evaluation.

        Returns:
            Evaluation metrics dictionary.

        Raises:
            ValueError: If training hasn't been run yet.
        """
        if self.agent is None:
            raise ValueError("No agent to evaluate. Run train() first.")

        eval_env = env if env is not None else self.eval_env
        if eval_env is None:
            eval_env = DummyVecEnv([self.eval_env_factory])

        return self.agent.evaluate(
            env=eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )

    def cleanup(self) -> None:
        """Clean up environments and resources."""
        if self.train_env is not None:
            self.train_env.close()
            self.train_env = None

        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

        logger.debug("Cleaned up trainer resources")
