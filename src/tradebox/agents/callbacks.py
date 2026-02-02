"""Custom callbacks for RL trading agent training."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from tradebox.agents.config import TrainingConfig


class TradingMetricsCallback(BaseCallback):
    """
    Custom callback to log trading-specific metrics to TensorBoard.

    Logs during training:
    - Episode portfolio values
    - Number of trades per episode
    - Win rate (trades with positive P&L)
    - Max drawdown

    The callback extracts metrics from the info dict returned by the
    trading environment at the end of each episode.

    Attributes:
        episode_rewards: List of episode total rewards
        episode_lengths: List of episode lengths
        episode_values: List of final portfolio values
        episode_trades: List of number of trades per episode

    Example:
        >>> callback = TradingMetricsCallback(verbose=1)
        >>> agent.train(callback=callback)
    """

    def __init__(self, verbose: int = 0) -> None:
        """
        Initialize trading metrics callback.

        Args:
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_values: List[float] = []
        self.episode_trades: List[int] = []

    def _on_training_start(self) -> None:
        """Called before training starts."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_values = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Extracts episode information from info dict when episodes end.

        Returns:
            True to continue training, False to stop.
        """
        # Check for episode end in vectorized environments
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                # Extract info from the environment
                infos = self.locals.get("infos", [])
                if i < len(infos):
                    info = infos[i]

                    # Get final episode info if available
                    episode_info = info.get("episode", {})
                    if episode_info:
                        reward = episode_info.get("r", 0)
                        length = episode_info.get("l", 0)
                        self.episode_rewards.append(reward)
                        self.episode_lengths.append(length)

                    # Get trading-specific metrics
                    portfolio_value = info.get("portfolio_value", 0)
                    total_trades = info.get("total_trades", 0)

                    if portfolio_value > 0:
                        self.episode_values.append(portfolio_value)
                    if total_trades > 0:
                        self.episode_trades.append(total_trades)

        # Log metrics periodically
        if self.n_calls % 1000 == 0 and len(self.episode_values) > 0:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Log trading metrics to TensorBoard."""
        if len(self.episode_values) == 0:
            return

        # Calculate metrics
        mean_value = np.mean(self.episode_values[-100:])
        mean_trades = np.mean(self.episode_trades[-100:]) if self.episode_trades else 0

        # Log to TensorBoard
        self.logger.record("trading/mean_portfolio_value", mean_value)
        self.logger.record("trading/mean_trades_per_episode", mean_trades)
        self.logger.record("trading/total_episodes", len(self.episode_values))

        if self.verbose > 0:
            logger.info(
                f"Trading metrics - Value: {mean_value:.2f}, "
                f"Trades/ep: {mean_trades:.1f}, Episodes: {len(self.episode_values)}"
            )

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose > 0:
            logger.info(
                f"Training complete - Total episodes: {len(self.episode_values)}"
            )


class TradingEvalCallback(EvalCallback):
    """
    Extended evaluation callback with trading-specific best model selection.

    This callback extends Stable-Baselines3's EvalCallback to select the
    best model based on Sharpe ratio instead of mean reward.

    The callback:
    1. Evaluates the agent every eval_freq steps
    2. Calculates Sharpe ratio from episode returns
    3. Saves the model when a new best Sharpe is achieved

    Attributes:
        best_sharpe: Best Sharpe ratio achieved during training
        sharpe_history: List of Sharpe ratios over evaluations
        metric: Metric to use for best model selection

    Example:
        >>> eval_callback = TradingEvalCallback(
        ...     eval_env,
        ...     best_model_save_path="models/best",
        ...     eval_freq=10000,
        ...     metric="sharpe_ratio",
        ... )
        >>> agent.train(callback=eval_callback)
    """

    def __init__(
        self,
        eval_env: Union[VecEnv, Any],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        metric: str = "sharpe_ratio",
        verbose: int = 1,
        warn: bool = True,
    ) -> None:
        """
        Initialize trading evaluation callback.

        Args:
            eval_env: Environment for evaluation.
            callback_on_new_best: Callback when new best model found.
            callback_after_eval: Callback after each evaluation.
            n_eval_episodes: Episodes per evaluation.
            eval_freq: Evaluation frequency in timesteps.
            log_path: Path for evaluation logs.
            best_model_save_path: Path to save best model.
            deterministic: Use deterministic policy for evaluation.
            metric: Metric to use for best model selection.
                   Options: 'sharpe_ratio', 'mean_reward', 'sortino_ratio'
            verbose: Verbosity level.
            warn: Whether to warn about potential issues.
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
            warn=warn,
        )
        # Store callback_after_eval as instance variable if parent doesn't
        if not hasattr(self, 'callback_after_eval'):
            self.callback_after_eval = callback_after_eval

        self.metric = metric
        self.best_sharpe = float("-inf")
        self.sharpe_history: List[float] = []
        self.episode_returns: List[float] = []

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Evaluates the agent and potentially saves a new best model
        based on the configured metric.

        Returns:
            True to continue training, False to stop.
        """
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync normalization if needed
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    pass

            # Evaluate and collect episode returns
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            # Calculate metrics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            sharpe = self._calculate_sharpe_ratio(np.array(episode_rewards))

            self.sharpe_history.append(sharpe)

            # Log metrics
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/sharpe_ratio", sharpe)
            self.logger.record("eval/mean_ep_length", np.mean(episode_lengths))

            if self.verbose > 0:
                logger.info(
                    f"Eval at {self.n_calls} steps - "
                    f"Mean reward: {mean_reward:.2f} (+/- {std_reward:.2f}), "
                    f"Sharpe: {sharpe:.3f}"
                )

            # Check if new best based on metric
            if self.metric == "sharpe_ratio":
                current_metric = sharpe
                best_metric = self.best_sharpe
            else:
                current_metric = mean_reward
                best_metric = self.best_mean_reward

            if current_metric > best_metric:
                if self.verbose > 0:
                    logger.info(
                        f"New best {self.metric}: {current_metric:.3f} "
                        f"(previous: {best_metric:.3f})"
                    )

                if self.metric == "sharpe_ratio":
                    self.best_sharpe = current_metric
                else:
                    self.best_mean_reward = current_metric

                # Save best model
                if self.best_model_save_path is not None:
                    self.model.save(
                        Path(self.best_model_save_path) / "best_model"
                    )

                    # Also save evaluation metrics
                    metrics_path = (
                        Path(self.best_model_save_path) / "best_model_metrics.json"
                    )
                    with open(metrics_path, "w") as f:
                        json.dump({
                            "mean_reward": float(mean_reward),
                            "std_reward": float(std_reward),
                            "sharpe_ratio": float(sharpe),
                            "timesteps": int(self.n_calls),
                            "n_eval_episodes": int(self.n_eval_episodes),
                        }, f, indent=2)

                # Trigger callback on new best
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after eval
            if self.callback_after_eval is not None:
                continue_training = continue_training and self.callback_after_eval.on_step()

            # Check early stopping conditions could be added here

        return continue_training

    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Calculate Sharpe ratio from episode returns.

        Args:
            returns: Array of episode returns.
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            Sharpe ratio (annualized assuming ~252 episodes/year).
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualization factor (assuming ~252 trading days/episodes per year)
        annualization = np.sqrt(252)
        sharpe = ((mean_return - risk_free_rate) / std_return) * annualization

        return float(sharpe)


class TradingCheckpointCallback(CheckpointCallback):
    """
    Extended checkpoint callback that saves config alongside model.

    Extends Stable-Baselines3's CheckpointCallback to also save
    configuration information with each checkpoint.

    Example:
        >>> checkpoint_callback = TradingCheckpointCallback(
        ...     save_freq=50000,
        ...     save_path="models/checkpoints",
        ...     name_prefix="ppo_trading",
        ... )
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_config: bool = True,
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ) -> None:
        """
        Initialize checkpoint callback.

        Args:
            save_freq: Save frequency in timesteps.
            save_path: Directory for checkpoints.
            name_prefix: Prefix for checkpoint names.
            save_config: Whether to save agent config with checkpoint.
            save_replay_buffer: Whether to save replay buffer (for off-policy).
            save_vecnormalize: Whether to save VecNormalize stats.
            verbose: Verbosity level.
        """
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.save_config = save_config

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Saves checkpoint and config when appropriate.

        Returns:
            True to continue training.
        """
        if self.n_calls % self.save_freq == 0:
            # Get the path that will be used for saving
            checkpoint_path = Path(self.save_path) / f"{self.name_prefix}_{self.num_timesteps}_steps"

            # Call parent to save model
            result = super()._on_step()

            # Save additional metadata
            if self.save_config:
                metadata_path = checkpoint_path.parent / f"{checkpoint_path.name}_metadata.json"
                metadata = {
                    "timesteps": int(self.num_timesteps),
                    "n_calls": int(self.n_calls),
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            return result

        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on validation performance.

    Stops training if no improvement is seen after a specified
    number of evaluations (patience).

    Attributes:
        best_metric: Best metric value seen so far
        patience_counter: Number of evaluations without improvement
        stopped_early: Whether training was stopped early

    Example:
        >>> early_stop = EarlyStoppingCallback(
        ...     metric="sharpe_ratio",
        ...     patience=10,
        ...     min_delta=0.01,
        ... )
        >>> agent.train(callback=early_stop)
    """

    def __init__(
        self,
        metric: str = "sharpe_ratio",
        patience: int = 10,
        min_delta: float = 0.0,
        check_freq: int = 10000,
        verbose: int = 1,
    ) -> None:
        """
        Initialize early stopping callback.

        Args:
            metric: Metric to monitor (from TensorBoard logs).
            patience: Number of checks without improvement before stopping.
            min_delta: Minimum improvement to count as progress.
            check_freq: How often to check for improvement (in timesteps).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.best_metric = float("-inf")
        self.patience_counter = 0
        self.stopped_early = False

    def _on_step(self) -> bool:
        """
        Called at each training step.

        Checks for improvement and stops if patience exceeded.

        Returns:
            True to continue training, False to stop.
        """
        if self.n_calls % self.check_freq != 0:
            return True

        # Try to get the metric from the logger
        # Note: This assumes the metric is being logged by another callback
        current_metric = None

        # Check locals for eval results
        if hasattr(self, "parent") and self.parent is not None:
            # Look for TradingEvalCallback in the callback list
            for callback in self.parent.callbacks:
                if isinstance(callback, TradingEvalCallback):
                    if callback.sharpe_history:
                        current_metric = callback.sharpe_history[-1]
                    break

        if current_metric is None:
            return True

        # Check for improvement
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.patience_counter = 0
            if self.verbose > 0:
                logger.info(f"New best {self.metric}: {current_metric:.3f}")
        else:
            self.patience_counter += 1
            if self.verbose > 0:
                logger.info(
                    f"No improvement in {self.metric}. "
                    f"Patience: {self.patience_counter}/{self.patience}"
                )

        # Check if should stop
        if self.patience_counter >= self.patience:
            if self.verbose > 0:
                logger.warning(
                    f"Early stopping triggered. No improvement for {self.patience} checks."
                )
            self.stopped_early = True
            return False

        return True


def create_callback_list(
    eval_env: VecEnv,
    training_config: TrainingConfig,
    agent_config: Optional[Any] = None,
    log_path: str = "logs",
) -> CallbackList:
    """
    Factory function to create standard callback list.

    Creates:
    - TradingEvalCallback (for best model selection by Sharpe)
    - TradingCheckpointCallback (for periodic saves)
    - TradingMetricsCallback (for TensorBoard logging)

    Args:
        eval_env: Evaluation environment.
        training_config: Training configuration.
        agent_config: Optional agent configuration for saving with checkpoints.
        log_path: Base path for logs.

    Returns:
        CallbackList with all callbacks configured.

    Example:
        >>> callbacks = create_callback_list(eval_env, training_config)
        >>> agent.train(callback=callbacks)
    """
    callbacks = []

    # Evaluation callback
    eval_callback = TradingEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=training_config.n_eval_episodes,
        eval_freq=training_config.eval_freq,
        log_path=str(Path(log_path) / "evaluations"),
        best_model_save_path=training_config.best_model_save_path,
        deterministic=True,
        metric="sharpe_ratio",
        verbose=training_config.verbose,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = TradingCheckpointCallback(
        save_freq=training_config.checkpoint_freq,
        save_path=training_config.model_save_dir,
        name_prefix="checkpoint",
        save_config=True,
        verbose=training_config.verbose,
    )
    callbacks.append(checkpoint_callback)

    # Trading metrics callback
    metrics_callback = TradingMetricsCallback(verbose=training_config.verbose)
    callbacks.append(metrics_callback)

    logger.info(
        f"Created callback list with {len(callbacks)} callbacks: "
        f"EvalCallback (freq={training_config.eval_freq}), "
        f"CheckpointCallback (freq={training_config.checkpoint_freq}), "
        f"MetricsCallback"
    )

    return CallbackList(callbacks)
