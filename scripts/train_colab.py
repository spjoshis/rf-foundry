"""
TradeBox-RL Training Script for Google Colab

This is a standalone training script designed to run on Google Colab.
Simply copy this entire file into a Colab notebook cell and run it.

Usage in Colab:
    1. Create new Colab notebook
    2. Copy this entire script into a cell
    3. Modify CONFIG section below as needed
    4. Run the cell
    5. Monitor training progress
"""

# ============================================================================
# SECTION 1: INSTALL DEPENDENCIES (Run this cell first)
# ============================================================================
"""
!pip install -q gymnasium==0.29.1
!pip install -q stable-baselines3==2.2.1
!pip install -q torch==2.1.2
!pip install -q pandas==2.1.4
!pip install -q numpy==1.26.2
!pip install -q pandas-ta==0.3.14b0
!pip install -q yfinance==0.2.33
!pip install -q pyarrow==14.0.1
!pip install -q pydantic==2.5.3
!pip install -q omegaconf==2.3.0
!pip install -q loguru==0.7.2
!pip install -q matplotlib==3.8.2
!pip install -q plotly==5.18.0
"""

# ============================================================================
# SECTION 2: MOUNT GOOGLE DRIVE (Optional)
# ============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/TradeBox-RL')
"""

# ============================================================================
# SECTION 3: MAIN TRAINING SCRIPT
# ============================================================================

import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from loguru import logger

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Experiment settings
    "experiment_name": "colab_ppo_baseline",
    "description": "PPO training on Google Colab",

    # Data settings
    "symbols": ["RELIANCE.NS"],  # Indian stocks (add .NS suffix)
    "start_date": "2010-01-01",
    "end_date": "2024-12-31",

    # Training settings (adjust for quick testing)
    "total_timesteps": 500000,  # Reduce to 50000 for quick test
    "n_envs": 8,  # Parallel environments
    "eval_freq": 10000,  # Evaluation frequency
    "checkpoint_freq": 50000,
    "seed": 42,

    # Agent hyperparameters
    "learning_rate": 0.0003,
    "network_arch": [256, 256],
    "gamma": 0.99,
    "clip_range": 0.2,

    # Environment settings
    "initial_capital": 100000.0,
    "reward_type": "risk_adjusted",  # simple, risk_adjusted, sharpe, sortino
    "lookback_window": 60,

    # Paths (will be created automatically)
    "cache_dir": "cache",
    "model_dir": "models",
    "log_dir": "logs",
}

# ============================================================================
# SETUP
# ============================================================================

def setup_environment():
    """Setup directories and Python path."""
    # Add src to path
    if os.path.exists('src'):
        sys.path.insert(0, 'src')

    # Create directories
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def create_config_yaml():
    """Create YAML config file for trainer."""
    config_dict = {
        "experiment": {
            "name": CONFIG["experiment_name"],
            "description": CONFIG["description"],
        },
        "data": {
            "symbols": CONFIG["symbols"],
            "start_date": CONFIG["start_date"],
            "end_date": CONFIG["end_date"],
        },
        "env": {
            "initial_capital": CONFIG["initial_capital"],
            "lookback_window": CONFIG["lookback_window"],
            "max_episode_steps": 500,
            "action_space": "Discrete(3)",
            "reward_config": {
                "reward_type": CONFIG["reward_type"],
            },
        },
        "agent": {
            "algorithm": "PPO",
            "ppo": {
                "learning_rate": CONFIG["learning_rate"],
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": CONFIG["gamma"],
                "gae_lambda": 0.95,
                "clip_range": CONFIG["clip_range"],
                "network_arch": CONFIG["network_arch"],
                "activation_fn": "tanh",
            },
        },
        "training": {
            "total_timesteps": CONFIG["total_timesteps"],
            "n_envs": CONFIG["n_envs"],
            "eval_freq": CONFIG["eval_freq"],
            "n_eval_episodes": 5,
            "checkpoint_freq": CONFIG["checkpoint_freq"],
            "seed": CONFIG["seed"],
            "device": "auto",
            "tensorboard_log": f"{CONFIG['log_dir']}/tensorboard",
            "model_save_dir": CONFIG["model_dir"],
            "best_model_save_path": f"{CONFIG['model_dir']}/best",
        },
    }

    config_path = Path("colab_training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    return config_path


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("TradeBox-RL Training on Google Colab")
    logger.info("=" * 60)

    # Setup
    logger.info("Setting up environment...")
    setup_environment()

    # Import modules (after path setup)
    logger.info("Importing modules...")
    from tradebox.data.loaders.yahoo_loader import YahooDataLoader
    from tradebox.data.splitter import DataSplitter, SplitConfig
    from tradebox.features.technical import TechnicalFeatures
    from tradebox.env import TradingEnv, EnvConfig
    from tradebox.agents import AgentTrainer

    # Load data
    logger.info(f"Loading data for {len(CONFIG['symbols'])} symbols...")
    logger.info(f"Date range: {CONFIG['start_date']} to {CONFIG['end_date']}")

    loader = YahooDataLoader(cache_dir=CONFIG["cache_dir"], use_cache=True)
    all_data = {}

    for symbol in CONFIG["symbols"]:
        logger.info(f"  Downloading {symbol}...")
        try:
            data = loader.download(symbol, CONFIG["start_date"], CONFIG["end_date"])
            all_data[symbol] = data
            logger.info(f"    Got {len(data)} rows")
        except Exception as e:
            logger.warning(f"    Failed to download {symbol}: {e}")

    if not all_data:
        logger.error("No data loaded. Exiting.")
        return 1

    # Use first symbol
    first_symbol = list(all_data.keys())[0]
    data = all_data[first_symbol]
    logger.info(f"Using {first_symbol} for training ({len(data)} rows)")

    # Split data
    logger.info("Splitting data into train/validation/test...")
    split_config = SplitConfig.default()
    splitter = DataSplitter(split_config)
    splits = splitter.split(data)

    logger.info(f"  Train: {len(splits['train'])} rows")
    logger.info(f"  Validation: {len(splits['validation'])} rows")
    logger.info(f"  Test: {len(splits['test'])} rows")

    # Extract features
    logger.info("Extracting technical features...")
    feature_extractor = TechnicalFeatures()

    train_features = feature_extractor.extract(splits["train"])
    val_features = feature_extractor.extract(splits["validation"])
    test_features = feature_extractor.extract(splits["test"])

    logger.info(f"  {len(train_features.columns)} features extracted")

    # Create config file
    logger.info("Creating config file...")
    config_path = create_config_yaml()

    # Create trainer
    logger.info("Creating agent trainer...")
    trainer = AgentTrainer.from_config(
        config_path=config_path,
        train_data=splits["train"],
        train_features=train_features,
        eval_data=splits["validation"],
        eval_features=val_features,
    )

    # Log configuration
    logger.info("")
    logger.info("Training configuration:")
    logger.info(f"  Total timesteps: {CONFIG['total_timesteps']:,}")
    logger.info(f"  Parallel envs: {CONFIG['n_envs']}")
    logger.info(f"  Eval frequency: {CONFIG['eval_freq']:,}")
    logger.info(f"  Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"  Network arch: {CONFIG['network_arch']}")
    logger.info("")

    # Train
    logger.info("Starting training...")
    logger.info("  To monitor: %tensorboard --logdir logs/tensorboard")
    logger.info("")

    results = trainer.train(
        total_timesteps=CONFIG["total_timesteps"],
    )

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = Path(CONFIG["model_dir"]) / f"{CONFIG['experiment_name']}_{timestamp}"
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
    test_env = TradingEnv(
        splits["test"],
        test_features,
        EnvConfig(
            initial_capital=CONFIG["initial_capital"],
            lookback_window=CONFIG["lookback_window"],
            max_episode_steps=min(500, len(splits["test"]) - 61),
        ),
    )

    test_metrics = trainer.evaluate(env=test_env, n_eval_episodes=5)

    logger.info("")
    logger.info("Test Set Evaluation:")
    logger.info(f"  Mean reward: {test_metrics['mean_reward']:.2f}")
    logger.info(f"  Sharpe ratio: {test_metrics['sharpe_ratio']:.3f}")
    logger.info("")

    # Cleanup
    trainer.cleanup()

    logger.info("Done!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review TensorBoard metrics")
    logger.info("  2. Download model from Colab")
    logger.info("  3. Run backtesting locally")
    logger.info("  4. Adjust hyperparameters if needed")

    return 0


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)
