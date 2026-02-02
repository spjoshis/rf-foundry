#!/usr/bin/env python3
"""
Train intraday PPO agent on 5-minute bar data.

This script implements STORY-037: Train and Validate Intraday Agent.
It trains a PPO agent on 5-minute intraday data, validates on out-of-sample
data, backtests the strategy, and compares performance against EOD strategy.

Key Features:
    - 5-minute bar data from Yahoo Finance
    - Intraday-specific technical features (VWAP, session high/low)
    - Dynamic slippage model for realistic costs
    - Session-aware environment (forced EOD closure)
    - Comprehensive validation and backtesting

Usage:
    # Basic training with config file
    python scripts/train_intraday.py --config configs/experiments/exp003_intraday_baseline.yaml

    # Quick test run (reduced timesteps)
    python scripts/train_intraday.py --config configs/experiments/exp003_intraday_baseline.yaml --quick

    # Override specific parameters
    python scripts/train_intraday.py --config configs/experiments/exp003_intraday_baseline.yaml \\
        --timesteps 1000000 --n-envs 16

Example:
    $ cd TradeBox-RL
    $ python scripts/train_intraday.py --config configs/experiments/exp003_intraday_baseline.yaml

    Training will:
    1. Download 5-minute intraday data (60 days)
    2. Extract intraday-specific features (VWAP, session extremes)
    3. Create train/validation/test splits (40/10/10 days)
    4. Train PPO agent with session handling
    5. Validate on out-of-sample data
    6. Generate backtest report with comparisons

Author: TradeBox-RL
Date: 2024-12-11
Story: STORY-037
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.data.splitter import DataSplitter, SplitConfig
from tradebox.features.technical import TechnicalFeatures, FeatureConfig
from tradebox.env.intraday_env import IntradayTradingEnv, IntradayEnvConfig
from tradebox.env.costs import CostConfig
from tradebox.env.rewards import RewardConfig
from tradebox.agents import AgentTrainer


# ============================================================================
# Configuration and Setup
# ============================================================================


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Configure logging for training.

    Args:
        verbose: If True, set debug level logging
        log_file: Optional path to write logs to file

    Returns:
        None
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB",
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed argument namespace

    Raises:
        SystemExit: If arguments are invalid
    """
    parser = argparse.ArgumentParser(
        description="Train PPO agent on intraday 5-minute data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file",
    )

    # Training overrides
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps (default: from config)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override number of parallel environments (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Device for training (default: auto)",
    )

    # Data overrides
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Override symbols to train on",
    )
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="Override data period (e.g., '60d', '90d')",
    )

    # Output paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for saved models (default: models)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs (default: logs)",
    )

    # Execution modes
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with reduced timesteps (100K)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip final backtest on test set",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# Data Loading and Preparation
# ============================================================================


def load_intraday_data(
    symbols: List[str],
    period: str,
    interval: str,
    cache_dir: str = "cache",
) -> Dict[str, pd.DataFrame]:
    """
    Load intraday data for given symbols.

    Args:
        symbols: List of stock symbols (e.g., ["RELIANCE.NS"])
        period: Data period (e.g., "60d")
        interval: Bar interval (e.g., "5m")
        cache_dir: Directory for caching downloaded data

    Returns:
        Dictionary mapping symbol to DataFrame with OHLCV data

    Raises:
        ValueError: If no data could be loaded for any symbol
    """
    logger.info(f"Loading intraday data for {len(symbols)} symbols...")
    logger.info(f"Period: {period}, Interval: {interval}")

    loader = YahooDataLoader(cache_dir=cache_dir, use_cache=True)
    all_data = {}

    for symbol in symbols:
        logger.info(f"  Downloading {symbol}...")
        try:
            # Use intraday download method
            data = loader.download_intraday(
                symbol=symbol,
                period=period,
                interval=interval,
            )
            all_data[symbol] = data
            logger.info(f"    Got {len(data)} bars")
        except Exception as e:
            logger.warning(f"    Failed to download {symbol}: {e}")

    if not all_data:
        raise ValueError("No data loaded for any symbol")

    return all_data


def split_intraday_data(
    data: pd.DataFrame,
    train_days: int,
    val_days: int,
    test_days: int,
    bars_per_day: int = 75,
) -> Dict[str, pd.DataFrame]:
    """
    Split intraday data into train/validation/test sets.

    Uses temporal splits (no random shuffling) to prevent lookahead bias.

    Args:
        data: Full dataset with OHLCV bars
        train_days: Number of trading days for training
        val_days: Number of trading days for validation
        test_days: Number of trading days for testing
        bars_per_day: Number of bars per trading day (default: 75 for 5-min bars)

    Returns:
        Dictionary with 'train', 'validation', 'test' DataFrames

    Raises:
        ValueError: If data is too short for requested splits
    """
    total_bars_needed = (train_days + val_days + test_days) * bars_per_day

    if len(data) < total_bars_needed:
        raise ValueError(
            f"Data too short: {len(data)} bars available, "
            f"{total_bars_needed} bars needed "
            f"({train_days}+{val_days}+{test_days} days × {bars_per_day} bars)"
        )

    # Calculate split indices (strict temporal order)
    train_end = train_days * bars_per_day
    val_end = train_end + (val_days * bars_per_day)
    test_end = val_end + (test_days * bars_per_day)

    splits = {
        "train": data.iloc[:train_end].copy(),
        "validation": data.iloc[train_end:val_end].copy(),
        "test": data.iloc[val_end:test_end].copy(),
    }

    logger.info("Data splits:")
    logger.info(f"  Train: {len(splits['train'])} bars ({train_days} days)")
    logger.info(f"  Validation: {len(splits['validation'])} bars ({val_days} days)")
    logger.info(f"  Test: {len(splits['test'])} bars ({test_days} days)")

    return splits


def extract_intraday_features(
    data: pd.DataFrame,
    feature_config: FeatureConfig,
    fit_normalize: bool = False,
) -> pd.DataFrame:
    """
    Extract technical features for intraday data.

    Includes standard technical indicators plus intraday-specific features
    like VWAP, session high/low, and multi-bar returns.

    Args:
        data: OHLCV DataFrame
        feature_config: Configuration for feature extraction
        fit_normalize: If True, fit normalizer on this data

    Returns:
        DataFrame with extracted features (same length as input)

    Raises:
        ValueError: If feature extraction fails
    """
    logger.info("Extracting intraday technical features...")

    extractor = TechnicalFeatures(feature_config)
    features = extractor.extract(data, fit_normalize=fit_normalize)

    logger.info(f"  Extracted {len(features.columns)} features")
    logger.info(f"  Feature names: {list(features.columns[:10])}...")

    return features


# ============================================================================
# Training and Evaluation
# ============================================================================


def create_intraday_trainer(
    config_path: Path,
    train_data: pd.DataFrame,
    train_features: pd.DataFrame,
    eval_data: pd.DataFrame,
    eval_features: pd.DataFrame,
) -> AgentTrainer:
    """
    Create AgentTrainer for intraday trading.

    Args:
        config_path: Path to experiment config YAML
        train_data: Training OHLCV data
        train_features: Training features
        eval_data: Validation OHLCV data
        eval_features: Validation features

    Returns:
        Configured AgentTrainer instance

    Raises:
        ValueError: If trainer creation fails
    """
    logger.info("Creating intraday agent trainer...")

    trainer = AgentTrainer.from_config(
        config_path=config_path,
        train_data=train_data,
        train_features=train_features,
        eval_data=eval_data,
        eval_features=eval_features,
    )

    return trainer


def run_backtest(
    trainer: AgentTrainer,
    test_data: pd.DataFrame,
    test_features: pd.DataFrame,
    n_eval_episodes: int = 10,
) -> Dict[str, float]:
    """
    Run backtest on test set.

    Args:
        trainer: Trained AgentTrainer
        test_data: Test OHLCV data
        test_features: Test features
        n_eval_episodes: Number of evaluation episodes

    Returns:
        Dictionary of backtest metrics

    Raises:
        ValueError: If backtest fails
    """
    logger.info("Running backtest on test set...")

    # Create test environment
    from tradebox.env import IntradayTradingEnv
    from tradebox.env.trading_env import IntradayEnvConfig

    # Load config from trainer
    env_config = trainer.env_config

    # Adjust config if test data is too short
    test_env_config = env_config
    available_bars = len(test_data)
    bars_per_session = env_config.bars_per_session
    min_required_bars = env_config.lookback_window + env_config.max_episode_steps + bars_per_session

    if available_bars < min_required_bars:
        # Calculate adjusted max_episode_steps for test data
        adjusted_max_steps = available_bars - env_config.lookback_window - bars_per_session
        adjusted_sessions = max(1, adjusted_max_steps // bars_per_session)
        adjusted_max_steps = adjusted_sessions * bars_per_session

        logger.info(
            f"Adjusting test env max_episode_steps: {env_config.max_episode_steps} -> {adjusted_max_steps} "
            f"(available bars: {available_bars}, sessions: {adjusted_sessions})"
        )

        # Create adjusted config for test
        test_env_config = IntradayEnvConfig(
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

    test_env = IntradayTradingEnv(
        data=test_data,
        features=test_features,
        config=test_env_config,
    )

    # Evaluate
    test_metrics = trainer.evaluate(
        env=test_env,
        n_eval_episodes=n_eval_episodes,
    )

    logger.info("")
    logger.info("Backtest Results:")
    logger.info(f"  Mean reward: {test_metrics.get('mean_reward', 0):.2f}")
    logger.info(f"  Sharpe ratio: {test_metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"  Max drawdown: {test_metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"  Win rate: {test_metrics.get('win_rate', 0):.2%}")

    return test_metrics


# ============================================================================
# Main Training Pipeline
# ============================================================================


def main() -> int:
    """
    Main training entry point.

    Returns:
        Exit code (0 for success, 1 for error)

    Raises:
        Exception: Various exceptions during training pipeline
    """
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.log_dir) / f"train_intraday_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=args.verbose, log_file=str(log_file))

    logger.info("=" * 70)
    logger.info("TradeBox-RL Intraday Training Script")
    logger.info("STORY-037: Train and Validate Intraday Agent")
    logger.info("=" * 70)

    try:
        # Load configuration
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        experiment_name = config.get("experiment", {}).get("name", "intraday_experiment")
        logger.info(f"Experiment: {experiment_name}")

        # Apply command-line overrides
        if args.quick:
            config.setdefault("training", {})["total_timesteps"] = 100000
            config["training"]["eval_freq"] = 10000
            config["training"]["checkpoint_freq"] = 50000
            logger.info("Quick mode: reduced to 100K timesteps")

        if args.timesteps:
            config.setdefault("training", {})["total_timesteps"] = args.timesteps
            logger.info(f"Override timesteps: {args.timesteps}")

        if args.n_envs:
            config.setdefault("training", {})["n_envs"] = args.n_envs
            logger.info(f"Override n_envs: {args.n_envs}")

        if args.seed:
            config.setdefault("training", {})["seed"] = args.seed
            logger.info(f"Override seed: {args.seed}")

        if args.device:
            config.setdefault("training", {})["device"] = args.device
            logger.info(f"Override device: {args.device}")

        if args.symbols:
            config.setdefault("data", {})["symbols"] = args.symbols
            logger.info(f"Override symbols: {args.symbols}")

        if args.period:
            config.setdefault("data", {})["period"] = args.period
            logger.info(f"Override period: {args.period}")

        # Extract data configuration
        data_config = config.get("data", {})
        symbols = data_config.get("symbols", ["RELIANCE.NS"])
        period = data_config.get("period", "60d")
        interval = data_config.get("interval", "5m")
        train_days = data_config.get("train_days", 40)
        val_days = data_config.get("val_days", 10)
        test_days = data_config.get("test_days", 10)

        # Load intraday data
        all_data = load_intraday_data(
            symbols=symbols,
            period=period,
            interval=interval,
            cache_dir="cache",
        )

        # Use first symbol for training (multi-stock support in future)
        first_symbol = list(all_data.keys())[0]
        data = all_data[first_symbol]
        logger.info(f"Using {first_symbol} for training ({len(data)} bars)")

        # Split data
        bars_per_day = config.get("env", {}).get("bars_per_session", 75)
        splits = split_intraday_data(
            data=data,
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            bars_per_day=bars_per_day,
        )

        # Extract features
        feature_config_dict = config.get("features", {})
        feature_config = FeatureConfig(
            timeframe="intraday",
            normalize=True,
            **feature_config_dict.get("technical", {}),
        )

        train_features = extract_intraday_features(
            splits["train"],
            feature_config,
            fit_normalize=True,
        )
        val_features = extract_intraday_features(
            splits["validation"],
            feature_config,
            fit_normalize=False,
        )
        test_features = extract_intraday_features(
            splits["test"],
            feature_config,
            fit_normalize=False,
        )

        # Create trainer
        config_path = Path(args.config)
        trainer = create_intraday_trainer(
            config_path=config_path,
            train_data=splits["train"],
            train_features=train_features,
            eval_data=splits["validation"],
            eval_features=val_features,
        )

        # Log training configuration
        logger.info("")
        logger.info("Training configuration:")
        training_config = config.get("training", {})
        logger.info(f"  Total timesteps: {training_config.get('total_timesteps', 3000000):,}")
        logger.info(f"  Parallel envs: {training_config.get('n_envs', 16)}")
        logger.info(f"  Eval frequency: {training_config.get('eval_freq', 25000):,}")

        agent_config = config.get("agent", {}).get("ppo", {})
        logger.info(f"  Learning rate: {agent_config.get('learning_rate', 0.0001)}")
        logger.info(f"  Network arch: {agent_config.get('network_arch', [256, 256])}")

        # Train
        logger.info("")
        logger.info("Starting training...")
        logger.info("  TensorBoard: tensorboard --logdir logs/tensorboard")
        logger.info("")

        results = trainer.train(
            total_timesteps=training_config.get("total_timesteps"),
        )

        # Save final model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        final_model_path = output_dir / f"{experiment_name}_{timestamp}"
        trainer.save_final_model(final_model_path)

        # Print training results
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"  Final timesteps: {results.get('final_timesteps', 'N/A'):,}")
        logger.info(f"  Best Sharpe: {results.get('best_sharpe', 'N/A'):.3f}")
        logger.info(f"  Training time: {results.get('training_time_seconds', 0):.1f}s")
        logger.info(f"  Model saved to: {final_model_path}")
        logger.info("")

        # Run backtest on test set
        if not args.skip_backtest:
            test_metrics = run_backtest(
                trainer=trainer,
                test_data=splits["test"],
                test_features=test_features,
                n_eval_episodes=10,
            )

            # Save metrics
            metrics_path = output_dir / f"{experiment_name}_{timestamp}_metrics.yaml"
            with open(metrics_path, "w") as f:
                yaml.dump(test_metrics, f)
            logger.info(f"  Metrics saved to: {metrics_path}")

        # Cleanup
        trainer.cleanup()

        logger.info("")
        logger.info("Done!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review TensorBoard logs for training progress")
        logger.info("  2. Analyze backtest metrics vs EOD baseline")
        logger.info("  3. Run scripts/compare_eod_vs_intraday.py for comparison")
        logger.info("  4. Iterate on hyperparameters if needed")

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
