#!/usr/bin/env python3
"""
Train RL agent on trading environment.

This script provides a CLI interface for training PPO agents on
the TradeBox trading environment with config-driven hyperparameters.

Usage:
    # Basic training with config file
    python scripts/train.py --config configs/experiments/exp001_baseline.yaml

    # Override specific parameters
    python scripts/train.py --config configs/experiments/exp001_baseline.yaml \\
        --timesteps 500000 --n-envs 4

    # Quick test run
    python scripts/train.py --config configs/experiments/exp001_baseline.yaml \\
        --quick

Example:
    $ cd TradeBox-RL
    $ python scripts/train.py --config configs/experiments/exp001_baseline.yaml

    Training will:
    1. Load data for configured symbols
    2. Create train/validation/test splits
    3. Extract technical features
    4. Initialize PPO agent with configured hyperparameters
    5. Train with TensorBoard logging and checkpointing
    6. Save best model based on validation Sharpe ratio
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from loguru import logger


def setup_logging(verbose: bool = False, log_file: str = None) -> None:
    """
    Configure logging for training.

    Args:
        verbose: If True, set debug level logging.
        log_file: Optional path to write logs to file.
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    level = "INFO" # "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation="10 MB",
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on trading environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file",
    )
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with reduced timesteps (10000)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Override symbols to train on",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def main() -> int:
    """
    Main training entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.log_dir) / f"train_{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=args.verbose, log_file=str(log_file))

    logger.info("=" * 60)
    logger.info("TradeBox-RL Training Script")
    logger.info("=" * 60)

    try:
        # Load configuration
        logger.info(f"Loading config from: {args.config}")
        config = load_config(args.config)
        experiment_name = config.get("experiment", {}).get("name", "experiment")
        logger.info(f"Experiment: {experiment_name}")

        # Apply command-line overrides
        if args.quick:
            config.setdefault("training", {})["total_timesteps"] = 10000
            config["training"]["eval_freq"] = 2000
            config["training"]["checkpoint_freq"] = 5000
            logger.info("Quick mode: reduced to 10000 timesteps")

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

        # Import modules (after config loading for better error messages)
        logger.info("Importing modules...")
        from tradebox.data.loaders.yahoo_loader import YahooDataLoader
        from tradebox.data.splitter import DataSplitter, SplitConfig
        from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
        from tradebox.features.technical import FeatureConfig
        from tradebox.features.regime import RegimeConfig
        from tradebox.data.loaders.fundamental_loader import FundamentalConfig
        from tradebox.agents import AgentTrainer

        # Load data
        data_config = config.get("data", {})
        symbols = data_config.get("symbols", ["RELIANCE.NS"])
        start_date = data_config.get("start_date", "2015-01-01")
        end_date = data_config.get("end_date", "2024-12-31")

        logger.info(f"Loading data for {len(symbols)} symbols...")
        logger.info(f"Date range: {start_date} to {end_date}")

        loader = YahooDataLoader(cache_dir='cache', use_cache=True)
        all_data = {}

        for symbol in symbols:
            logger.info(f"  Downloading {symbol}...")
            try:
                data = loader.download(symbol, start_date, end_date)
                all_data[symbol] = data
                logger.info(f"    Got {len(data)} rows")
            except Exception as e:
                logger.warning(f"    Failed to download {symbol}: {e}")

        if not all_data:
            logger.error("No data loaded. Exiting.")
            return 1

        # Use first symbol for training (multi-stock training in future)
        first_symbol = list(all_data.keys())[0]
        data = all_data[first_symbol]
        logger.info(f"Using {first_symbol} for training ({len(data)} rows)")

        # Split data
        logger.info("Splitting data into train/validation/test...")
        # Use default split configuration:
        # Train: 2010-2018 (9 years, ~70%)
        # Validation: 2019-2021 (3 years, ~15%)
        # Test: 2022-2024 (3 years, ~15%)
        split_config = SplitConfig.default()
        splitter = DataSplitter(split_config)
        splits = splitter.split(data)

        # Set Date as index for FeatureExtractor compatibility
        for split_name in ['train', 'validation', 'test']:
            if 'Date' in splits[split_name].columns:
                splits[split_name] = splits[split_name].set_index('Date')
                # Remove timezone info to avoid comparison issues
                if splits[split_name].index.tz is not None:
                    splits[split_name].index = splits[split_name].index.tz_localize(None)

        logger.info(f"  Train: {len(splits['train'])} rows")
        logger.info(f"  Validation: {len(splits['validation'])} rows")
        logger.info(f"  Test: {len(splits['test'])} rows")

        # Extract features
        logger.info("Extracting technical + regime features...")

        # Build FeatureExtractorConfig from config file
        feature_config_dict = config.get("features", {})

        # Create technical config
        technical_config = FeatureConfig()
        if "technical" in feature_config_dict:
            tech_params = feature_config_dict["technical"]
            for key, value in tech_params.items():
                if hasattr(technical_config, key):
                    setattr(technical_config, key, value)

        # Create regime config
        regime_config = RegimeConfig()
        if "regime" in feature_config_dict:
            regime_params = feature_config_dict["regime"]
            for key, value in regime_params.items():
                if hasattr(regime_config, key):
                    setattr(regime_config, key, value)

        # Create fundamental config
        fundamental_config = FundamentalConfig()
        if "fundamental" in feature_config_dict:
            fund_params = feature_config_dict["fundamental"]
            for key, value in fund_params.items():
                if hasattr(fundamental_config, key):
                    setattr(fundamental_config, key, value)

        # Create FeatureExtractor
        extractor_config = FeatureExtractorConfig(
            technical=technical_config,
            regime=regime_config,
            fundamental=fundamental_config
        )
        feature_extractor = FeatureExtractor(extractor_config)

        train_features = feature_extractor.extract(
            symbol=first_symbol,
            price_data=splits["train"],
            fit_normalize=True
        )
        val_features = feature_extractor.extract(
            symbol=first_symbol,
            price_data=splits["validation"],
            fit_normalize=False
        )

        logger.info(f"  {len(train_features.columns)} features extracted")

        # Create trainer
        logger.info("Creating agent trainer...")

        # Write config to temp file for trainer
        config_path = Path(args.config)

        trainer = AgentTrainer.from_config(
            config_path=config_path,
            train_data=splits["train"],
            train_features=train_features,
            eval_data=splits["validation"],
            eval_features=val_features,
        )

        # Log configuration
        logger.info("Training configuration:")
        training_config = config.get("training", {})
        logger.info(f"  Total timesteps: {training_config.get('total_timesteps', 2000000):,}")
        logger.info(f"  Parallel envs: {training_config.get('n_envs', 8)}")
        logger.info(f"  Eval frequency: {training_config.get('eval_freq', 10000):,}")

        agent_config = config.get("agent", {})
        logger.info(f"  Learning rate: {agent_config.get('learning_rate', 0.0003)}")
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

        # Print results
        logger.info("")
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)
        logger.info(f"  Final timesteps: {results.get('final_timesteps', 'N/A'):,}")
        logger.info(f"  Best Sharpe: {results.get('best_sharpe', 'N/A'):.3f}")
        logger.info(f"  Training time: {results.get('training_time_seconds', 0):.1f}s")
        logger.info(f"  Model saved to: {final_model_path}")
        logger.info("")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_features = feature_extractor.extract(
            symbol=first_symbol,
            price_data=splits["test"],
            fit_normalize=False
        )

        from tradebox.env import TradingEnv, EnvConfig

        test_env = TradingEnv(
            splits["test"],
            test_features,
            EnvConfig(
                initial_capital=100000.0,
                lookback_window=60,
                max_episode_steps=min(500, len(splits["test"]) - 61),
            ),
        )

        test_metrics = trainer.evaluate(env=test_env, n_eval_episodes=5)

        logger.info("Test set evaluation:")
        logger.info(f"  Mean reward: {test_metrics['mean_reward']:.2f}")
        logger.info(f"  Sharpe ratio: {test_metrics['sharpe_ratio']:.3f}")

        # Cleanup
        trainer.cleanup()

        logger.info("")
        logger.info("Done!")

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
