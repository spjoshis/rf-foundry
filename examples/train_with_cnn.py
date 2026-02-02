"""
Example: Training PPO agent with CNN-based feature extraction.

This script demonstrates how to use TradingCNNExtractor for learning
directly from raw OHLCV candlestick patterns.

Usage:
    # Intraday trading
    poetry run python examples/train_with_cnn.py --env intraday

    # EOD swing trading
    poetry run python examples/train_with_cnn.py --env eod
"""

import argparse
from pathlib import Path

import numpy as np
from loguru import logger

from tradebox.agents.config import AgentConfig, PPOConfig, TrainingConfig
from tradebox.agents.ppo_agent import PPOAgent
from tradebox.agents.trainer import AgentTrainer
from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.env.intraday_env import IntradayTradingEnv
from tradebox.env.trading_env import EnvConfig, IntradayEnvConfig, TradingEnv
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig


def create_intraday_env_with_cnn():
    """Create intraday environment with CNN-compatible Dict observation space."""
    logger.info("Loading intraday data...")
    loader = YahooDataLoader(Path("data/intraday"))
    data = loader.download_intraday("RELIANCE.NS", period="60d", interval="5m")

    logger.info("Extracting features...")
    extractor = FeatureExtractor(FeatureExtractorConfig())
    features = extractor.extract("RELIANCE", data, fit_normalize=True)

    logger.info("Creating IntradayTradingEnv with Dict observation space...")
    config = IntradayEnvConfig(
        lookback_window=60,
        max_episode_steps=750,
        bars_per_session=75,
        sessions_per_episode=10,
    )
    env = IntradayTradingEnv(data, features, config)

    logger.info(f"Environment created: {env.observation_space}")
    return env


def create_eod_env_with_cnn():
    """Create EOD environment with CNN-compatible Dict observation space."""
    logger.info("Loading EOD data...")
    loader = YahooDataLoader(Path("data/eod"))
    data = loader.download(
        symbol="RELIANCE.NS",
        start_date="2020-01-01",
        end_date="2024-12-31",
    )

    logger.info("Extracting features (technicals + fundamentals)...")
    extractor = FeatureExtractor(
        FeatureExtractorConfig(
            use_technical=True,
            use_fundamental=True,  # EOD includes fundamentals
        )
    )
    features = extractor.extract("RELIANCE", data, fit_normalize=True)

    logger.info("Creating TradingEnv with Dict observation space...")
    config = EnvConfig(
        lookback_window=60,
        max_episode_steps=500,
    )
    env = TradingEnv(data, features, config)

    logger.info(f"Environment created: {env.observation_space}")
    return env


def create_cnn_agent_config(env_type: str) -> AgentConfig:
    """
    Create agent configuration with CNN feature extraction enabled.

    Args:
        env_type: "intraday" or "eod"

    Returns:
        AgentConfig with TradingCNNExtractor enabled
    """
    logger.info(f"Creating {env_type} agent config with CNN extractor...")

    ppo_config = PPOConfig(
        # Standard PPO hyperparameters
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # Enable CNN feature extractor
        use_cnn_extractor=True,
        extractor_type="trading",  # Use TradingCNNExtractor (recommended)
        # CNN architecture
        price_embed_dim=128,  # CNN output dimension for price patterns
        ind_embed_dim=64,  # MLP output dimension for indicators
        port_embed_dim=32,  # MLP output dimension for portfolio state
        fund_embed_dim=16,  # MLP output dimension for fundamentals (EOD only)
        # CNN options
        use_attention=True,  # Enable attention mechanism (recommended)
        use_fusion=True,  # Use fusion layer after concatenation
        cnn_dropout=0.1,  # Dropout for regularization
        # Policy network (after feature extraction)
        network_arch=[256, 256],
        activation_fn="tanh",
    )

    training_config = TrainingConfig(
        total_timesteps=100000,  # Short for demo (increase to 2M for real training)
        n_envs=4,  # Reduce from 8 to save memory with CNN
        eval_freq=10000,
        checkpoint_freq=50000,
        tensorboard_log=f"logs/tensorboard/cnn_{env_type}_example",
        model_save_dir=f"models/cnn_{env_type}_example",
        best_model_save_path=f"models/cnn_{env_type}_example/best",
        verbose=1,
        device="auto",  # Use GPU if available
    )

    return AgentConfig(
        algorithm="PPO",
        ppo=ppo_config,
        training=training_config,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent with CNN extractor")
    parser.add_argument(
        "--env",
        type=str,
        choices=["intraday", "eod"],
        default="intraday",
        help="Environment type: intraday or eod",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"Training PPO agent with CNN-based feature extraction ({args.env})")
    logger.info("=" * 80)

    # Create environment
    if args.env == "intraday":
        env = create_intraday_env_with_cnn()
    else:
        env = create_eod_env_with_cnn()

    # Verify Dict observation space
    assert isinstance(env.observation_space, dict) or hasattr(
        env.observation_space, "spaces"
    ), "Environment must return Dict observations for CNN extractor"

    logger.info(f"Observation space keys: {list(env.observation_space.spaces.keys())}")
    logger.info(f"  - price: {env.observation_space['price'].shape}")
    logger.info(f"  - indicators: {env.observation_space['indicators'].shape}")
    if "fundamentals" in env.observation_space.spaces:
        logger.info(f"  - fundamentals: {env.observation_space['fundamentals'].shape}")
    logger.info(f"  - portfolio: {env.observation_space['portfolio'].shape}")

    # Create agent config with CNN
    agent_config = create_cnn_agent_config(args.env)

    logger.info("Agent configuration:")
    logger.info(f"  - Algorithm: {agent_config.algorithm}")
    logger.info(f"  - Use CNN: {agent_config.ppo.use_cnn_extractor}")
    logger.info(f"  - Extractor Type: {agent_config.ppo.extractor_type}")
    logger.info(f"  - Price embed dim: {agent_config.ppo.price_embed_dim}")
    logger.info(f"  - Indicator embed dim: {agent_config.ppo.ind_embed_dim}")
    logger.info(f"  - Portfolio embed dim: {agent_config.ppo.port_embed_dim}")
    logger.info(f"  - Fundamental embed dim: {agent_config.ppo.fund_embed_dim}")
    logger.info(f"  - Use attention: {agent_config.ppo.use_attention}")

    # Calculate total features dimension
    total_features = (
        agent_config.ppo.price_embed_dim
        + agent_config.ppo.ind_embed_dim
        + agent_config.ppo.port_embed_dim
        + agent_config.ppo.fund_embed_dim
    )
    logger.info(f"  - Total features dim: {total_features}")

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = AgentTrainer(
        agent_config=agent_config,
        train_env_factory=lambda: env,
        eval_env_factory=lambda: env,  # In real training, use separate eval env
        experiment_name=f"cnn_{args.env}_example",
    )

    # Train agent
    logger.info("\nStarting training...")
    logger.info("Monitor with: tensorboard --logdir logs/tensorboard/")
    logger.info("=" * 80)

    trainer.train()

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {agent_config.training.best_model_save_path}")
    logger.info("=" * 80)

    # Example: Load and use trained model
    logger.info("\nExample: Loading trained model for inference...")
    from stable_baselines3 import PPO

    model_path = f"{agent_config.training.best_model_save_path}/best_model.zip"
    if Path(model_path).exists():
        model = PPO.load(model_path)
        logger.info("Model loaded successfully!")

        # Test inference
        obs, info = env.reset()
        logger.info(f"\nObservation keys: {obs.keys()}")
        logger.info(f"Price shape: {obs['price'].shape}")
        logger.info(f"Indicators shape: {obs['indicators'].shape}")
        logger.info(f"Portfolio shape: {obs['portfolio'].shape}")

        action, _states = model.predict(obs, deterministic=True)
        logger.info(f"\nPredicted action: {action} ({['Hold', 'Buy', 'Sell'][action]})")
    else:
        logger.warning(f"Model not found at {model_path}")


if __name__ == "__main__":
    main()
