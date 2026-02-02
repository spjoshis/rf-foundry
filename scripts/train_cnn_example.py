"""
Example training script demonstrating CNN-based feature extraction.

This script shows how to train a PPO agent with CNN-based pattern recognition
on raw OHLCV price data while leveraging technical indicators and portfolio state.

Usage:
    python scripts/train_cnn_example.py --symbol RELIANCE.NS --cnn-type multiscale

Features:
    - CNN-based price pattern extraction
    - Multi-stock rotation for better generalization
    - Comprehensive evaluation and comparison with MLP baseline
    - Checkpoint saving and TensorBoard logging
"""

import argparse
import random
from pathlib import Path
from typing import Callable

import gymnasium as gym
import pandas as pd
from loguru import logger

from tradebox.agents import PPOAgent, PPOConfig, TrainingConfig
from tradebox.agents.callbacks import EvalCallback, MetricCallback
from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.env import EnvConfig, TradingEnv
from tradebox.env.wrappers import make_cnn_compatible_env
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent with CNN feature extraction"
    )

    # Data arguments
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        help="Stock symbols to train on (multi-stock if multiple provided)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date for training data",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date for training data",
    )

    # CNN architecture arguments
    parser.add_argument(
        "--cnn-type",
        type=str,
        default="multiscale",
        choices=["simple", "multiscale", "residual"],
        help="Type of CNN architecture to use",
    )
    parser.add_argument(
        "--use-cnn",
        action="store_true",
        default=True,
        help="Use CNN feature extractor (default: True)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train MLP baseline instead of CNN (for comparison)",
    )

    # Training arguments
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate (auto-adjusted for CNN if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for training",
    )

    # CNN hyperparameters
    parser.add_argument(
        "--price-embed-dim",
        type=int,
        default=128,
        help="CNN embedding dimension for price data",
    )
    parser.add_argument(
        "--ind-embed-dim",
        type=int,
        default=64,
        help="MLP embedding dimension for indicators",
    )
    parser.add_argument(
        "--cnn-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for CNN layers",
    )

    # Environment arguments
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=60,
        help="Lookback window size (days for EOD)",
    )
    parser.add_argument(
        "--normalize-price",
        action="store_true",
        help="Apply normalization to price data",
    )

    # Output arguments
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (auto-generated if not provided)",
    )

    return parser.parse_args()


def load_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    loader: YahooDataLoader,
    extractor: FeatureExtractor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OHLCV data and extract features for a symbol.

    Args:
        symbol: Stock symbol
        start_date: Start date string
        end_date: End date string
        loader: Data loader instance
        extractor: Feature extractor instance

    Returns:
        Tuple of (data, features) DataFrames
    """
    logger.info(f"Loading data for {symbol}")
    data = loader.download(symbol, start_date, end_date)

    logger.info(f"Extracting features for {symbol}")
    features = extractor.extract(symbol, data, fit_normalize=True)

    logger.info(f"Loaded {len(data)} bars for {symbol}")
    return data, features


def create_env_factory(
    symbols: list[str],
    start_date: str,
    end_date: str,
    env_config: EnvConfig,
    lookback_window: int,
    normalize_price: bool,
    use_cnn: bool,
    loader: YahooDataLoader,
    extractor: FeatureExtractor,
) -> Callable[[], gym.Env]:
    """
    Create environment factory function for vectorized training.

    Args:
        symbols: List of stock symbols to sample from
        start_date: Start date for data
        end_date: End date for data
        env_config: Environment configuration
        lookback_window: Lookback window size
        normalize_price: Whether to normalize price data
        use_cnn: Whether to wrap for CNN compatibility
        loader: Data loader instance
        extractor: Feature extractor instance

    Returns:
        Factory function that creates environments
    """
    def make_env() -> gym.Env:
        # Randomly sample symbol for multi-stock training
        symbol = random.choice(symbols) if len(symbols) > 1 else symbols[0]

        # Load data and features
        data, features = load_stock_data(
            symbol, start_date, end_date, loader, extractor
        )

        # Create base environment
        base_env = TradingEnv(data, features, env_config)

        # Wrap for CNN if needed
        if use_cnn:
            env = make_cnn_compatible_env(
                base_env,
                data=data,
                lookback_window=lookback_window,
                normalize=normalize_price,
            )
        else:
            env = base_env

        return env

    return make_env


def main():
    """Main training loop."""
    args = parse_args()

    # Override use_cnn if baseline mode
    use_cnn = args.use_cnn and not args.baseline

    # Auto-adjust learning rate for CNN if not explicitly set
    if use_cnn and args.learning_rate == 0.0003:
        # Lower learning rate for CNN (especially for deeper architectures)
        lr_map = {
            "simple": 0.0002,
            "multiscale": 0.00015,
            "residual": 0.0001,
        }
        learning_rate = lr_map.get(args.cnn_type, 0.00015)
        logger.info(f"Auto-adjusted learning rate to {learning_rate} for CNN")
    else:
        learning_rate = args.learning_rate

    # Generate experiment name
    if args.experiment_name is None:
        if args.baseline:
            exp_name = f"mlp_baseline_{'_'.join(args.symbols[:2])}"
        else:
            exp_name = f"cnn_{args.cnn_type}_{'_'.join(args.symbols[:2])}"
        args.experiment_name = exp_name

    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Architecture: {'MLP baseline' if args.baseline else f'CNN ({args.cnn_type})'}")

    # Initialize loaders
    loader = YahooDataLoader()
    feature_config = FeatureExtractorConfig()
    extractor = FeatureExtractor(feature_config)

    # Environment configuration
    env_config = EnvConfig(
        lookback_window=args.lookback_window,
        max_episode_steps=500,
    )

    # Create environment factory
    env_factory = create_env_factory(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        env_config=env_config,
        lookback_window=args.lookback_window,
        normalize_price=args.normalize_price,
        use_cnn=use_cnn,
        loader=loader,
        extractor=extractor,
    )

    # Create initial environment
    logger.info("Creating initial environment")
    env = env_factory()

    # Configure PPO agent
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=128 if use_cnn else 64,  # Larger batch for CNN
        n_epochs=10,
        network_arch=[256, 256],
        # CNN-specific settings
        use_cnn_extractor=use_cnn,
        cnn_type=args.cnn_type,
        price_embed_dim=args.price_embed_dim,
        ind_embed_dim=args.ind_embed_dim,
        port_embed_dim=32,
        cnn_dropout=args.cnn_dropout,
        use_fusion=True,
    )

    training_config = TrainingConfig(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        eval_freq=10000,
        n_eval_episodes=5,
        checkpoint_freq=50000,
        device=args.device,
        tensorboard_log=f"logs/tensorboard/{args.experiment_name}",
        model_save_dir=args.model_save_dir,
        best_model_save_path=f"{args.model_save_dir}/best_{args.experiment_name}",
    )

    logger.info("Creating PPO agent")
    logger.info(f"PPO Config: {ppo_config}")

    # Create agent
    agent = PPOAgent(
        env=env,
        config=ppo_config,
        training_config=training_config,
        env_factory=env_factory,
    )

    # Create evaluation environment (single stock for consistency)
    eval_symbol = args.symbols[0]
    eval_data, eval_features = load_stock_data(
        eval_symbol, args.start_date, args.end_date, loader, extractor
    )
    eval_base_env = TradingEnv(eval_data, eval_features, env_config)
    eval_env = make_cnn_compatible_env(
        eval_base_env,
        data=eval_data,
        lookback_window=args.lookback_window,
        normalize=args.normalize_price,
    ) if use_cnn else eval_base_env

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=training_config.best_model_save_path,
        log_path=f"logs/eval/{args.experiment_name}",
        eval_freq=training_config.eval_freq,
        n_eval_episodes=training_config.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    metric_callback = MetricCallback()

    callbacks = [eval_callback, metric_callback]

    # Train agent
    logger.info("Starting training")
    agent.train(callback=callbacks)

    # Save final model
    final_path = Path(args.model_save_dir) / args.experiment_name
    agent.save(final_path)
    logger.info(f"Final model saved to {final_path}")

    # Final evaluation
    logger.info("Running final evaluation")
    final_metrics = agent.evaluate(eval_env, n_eval_episodes=20, deterministic=True)

    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)
    for key, value in final_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    logger.info("=" * 60)

    # Save metrics
    metrics_path = Path(args.model_save_dir) / f"{args.experiment_name}_metrics.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
