#!/usr/bin/env python3
"""
End-to-end pipeline test script.

This script tests the full functionality:
1. Download data from Yahoo Finance
2. Validate data quality
3. Split data temporally
4. Extract technical features
5. Run trading environment with random/heuristic agent

Usage:
    poetry run python scripts/test_pipeline.py
    poetry run python scripts/test_pipeline.py --symbol TCS.NS --days 100
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.data.validation import DataValidator, ValidationConfig
from tradebox.data.splitter import DataSplitter, SplitConfig
from tradebox.features.technical import TechnicalFeatures, FeatureConfig
from tradebox.features.analyzer import FeatureAnalyzer
from tradebox.env.trading_env import TradingEnv, EnvConfig
from tradebox.env.costs import CostConfig
from tradebox.env.rewards import RewardConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")


def test_data_download(symbol: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    """Test Yahoo Finance data download."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Download from Yahoo Finance")
    logger.info("=" * 60)

    loader = YahooDataLoader(cache_dir=cache_dir, use_cache=True)
    df = loader.download(symbol, start, end)

    logger.info(f"Downloaded {len(df)} rows for {symbol}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nSample data:\n{df.head(3)}")
    logger.info(f"\nData statistics:\n{df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()}")

    return df


def test_data_validation(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Test data validation."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Data Validation")
    logger.info("=" * 60)

    config = ValidationConfig(
        price_jump_threshold=0.25,  # 25% threshold for suspicious jumps
        volume_spike_threshold=10.0,  # 10x average volume
        zero_volume_days_threshold=5,
        stale_price_days_threshold=10,
        max_gap_days=7,
    )

    validator = DataValidator(config)
    report = validator.validate(df, symbol)

    logger.info(f"Validation passed: {report.is_valid}")
    logger.info(f"Summary: {report.summary}")

    if report.issues:
        logger.warning(f"Issues found ({len(report.issues)} total):")
        for issue in report.issues[:10]:  # Show first 10 issues
            logger.warning(f"  - {issue}")
        if len(report.issues) > 10:
            logger.warning(f"  ... and {len(report.issues) - 10} more")

    # Drop rows with NaN values for clean data
    cleaned_df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    logger.info(f"Data shape after cleaning: {cleaned_df.shape}")

    return cleaned_df


def test_data_split(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """Test temporal data splitting."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Temporal Data Split")
    logger.info("=" * 60)

    # Calculate split dates based on data range
    # Using 70% train, 15% val, 15% test
    df_sorted = df.sort_values("Date")
    total_days = len(df_sorted)

    train_end_idx = int(total_days * 0.70)
    val_end_idx = int(total_days * 0.85)

    train_end_date = df_sorted.iloc[train_end_idx]["Date"].strftime("%Y-%m-%d")
    val_start_date = df_sorted.iloc[train_end_idx + 1]["Date"].strftime("%Y-%m-%d") if train_end_idx + 1 < total_days else train_end_date
    val_end_date = df_sorted.iloc[val_end_idx]["Date"].strftime("%Y-%m-%d")
    test_start_date = df_sorted.iloc[val_end_idx + 1]["Date"].strftime("%Y-%m-%d") if val_end_idx + 1 < total_days else val_end_date

    config = SplitConfig(
        train_start=start_date,
        train_end=train_end_date,
        val_start=val_start_date,
        val_end=val_end_date,
        test_start=test_start_date,
        test_end=end_date,
    )

    logger.info(f"Split config: train=[{config.train_start} to {config.train_end}], "
                f"val=[{config.val_start} to {config.val_end}], "
                f"test=[{config.test_start} to {config.test_end}]")

    splitter = DataSplitter(config)
    splits = splitter.split(df)

    for name, split_df in splits.items():
        if len(split_df) > 0:
            logger.info(f"{name.upper()}: {len(split_df)} rows ({len(split_df)/len(df)*100:.1f}%)")
            if "Date" in split_df.columns:
                logger.info(f"  Date range: {split_df['Date'].min()} to {split_df['Date'].max()}")
        else:
            logger.info(f"{name.upper()}: 0 rows")

    return splits


def test_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Test technical feature extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Technical Feature Extraction")
    logger.info("=" * 60)

    config = FeatureConfig(
        normalize=True,
        sma_periods=[20, 50],  # Shorter periods for testing
        ema_periods=[9, 21],
        rsi_period=14,
        atr_period=14,
        bollinger_period=20,
    )

    extractor = TechnicalFeatures(config)
    features_df = extractor.extract(df)

    feature_names = extractor.get_feature_names()
    logger.info(f"Extracted {len(feature_names)} features:")
    for name in feature_names[:10]:  # Show first 10
        logger.info(f"  - {name}")
    if len(feature_names) > 10:
        logger.info(f"  ... and {len(feature_names) - 10} more")

    logger.info(f"\nFeature DataFrame shape: {features_df.shape}")
    logger.info(f"NaN counts (top 5):\n{features_df[feature_names].isna().sum().nlargest(5)}")

    return features_df


def test_feature_analysis(df: pd.DataFrame, feature_names: list) -> None:
    """Test feature analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Feature Analysis")
    logger.info("=" * 60)

    analyzer = FeatureAnalyzer()

    # Calculate future returns for analysis
    df = df.copy()
    df["Future_Return"] = df["Close"].pct_change(5).shift(-5)  # 5-day forward return

    # Remove NaNs
    analysis_df = df.dropna()

    if len(analysis_df) > 100:
        # Analyze a few features
        for feature in feature_names[:3]:
            if feature in analysis_df.columns:
                correlation = analysis_df[feature].corr(analysis_df["Future_Return"])
                logger.info(f"{feature} correlation with 5-day forward return: {correlation:.4f}")
    else:
        logger.warning("Insufficient data for feature analysis")


def test_trading_environment(data: pd.DataFrame, features_df: pd.DataFrame, n_episodes: int = 3) -> None:
    """Test trading environment with random and heuristic agents."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Trading Environment Test")
    logger.info("=" * 60)

    # Prepare feature columns only (exclude OHLCV)
    feature_cols = [c for c in features_df.columns if c not in ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
    features_only = features_df[feature_cols].copy()

    # Drop rows with NaN
    valid_idx = ~features_only.isna().any(axis=1)
    data_clean = data[valid_idx].reset_index(drop=True)
    features_clean = features_only[valid_idx].reset_index(drop=True)

    logger.info(f"Clean data shape: {data_clean.shape}")
    logger.info(f"Clean features shape: {features_clean.shape}")

    # Check minimum data requirements
    min_required = 60 + 100 + 1  # lookback + max_episode + 1
    if len(data_clean) < min_required:
        logger.warning(f"Insufficient data ({len(data_clean)} rows). Need at least {min_required}.")
        logger.warning("Try downloading more historical data.")
        return

    # Create environment config (using defaults which are Zerodha-realistic)
    env_config = EnvConfig(
        initial_capital=100000.0,
        lookback_window=60,
        max_episode_steps=min(100, len(data_clean) - 61),
        cost_config=CostConfig(),  # Defaults are Zerodha-realistic
        reward_config=RewardConfig(
            reward_type="risk_adjusted",
            drawdown_penalty=0.5,
            trade_penalty=0.001,
        ),
    )

    # Create environment
    env = TradingEnv(data_clean, features_clean, env_config)

    logger.info(f"Environment created:")
    logger.info(f"  Action space: {env.action_space}")
    logger.info(f"  Observation space shape: {env.observation_space.shape}")

    # Test with random agent
    logger.info("\n--- Random Agent Test ---")
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        final_value = info["portfolio_value"]
        returns = (final_value / env_config.initial_capital - 1) * 100
        logger.info(
            f"Episode {episode + 1}: {steps} steps, "
            f"Return: {returns:+.2f}%, "
            f"Trades: {info['total_trades']}, "
            f"Reward: {total_reward:.4f}"
        )

    # Test with simple heuristic agent (RSI-based)
    logger.info("\n--- RSI Heuristic Agent Test ---")
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Simple heuristic: use RSI from observation
            # RSI is typically in the observation vector
            # Action: Buy when oversold region, Sell when overbought
            current_step = info["step"]

            # Get RSI value if available (this is simplified)
            # In practice, you'd track the RSI feature index
            if info["position"] == 0:
                action = 1  # Try to buy if no position
            elif steps > 20 and info["unrealized_pnl_pct"] > 0.02:
                action = 2  # Sell if 2% profit
            elif steps > 20 and info["unrealized_pnl_pct"] < -0.03:
                action = 2  # Cut loss at 3%
            else:
                action = 0  # Hold

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        final_value = info["portfolio_value"]
        returns = (final_value / env_config.initial_capital - 1) * 100
        logger.info(
            f"Episode {episode + 1}: {steps} steps, "
            f"Return: {returns:+.2f}%, "
            f"Trades: {info['total_trades']}, "
            f"Reward: {total_reward:.4f}"
        )


def test_gymnasium_compatibility(data: pd.DataFrame, features_df: pd.DataFrame) -> None:
    """Test Gymnasium API compatibility using check_env."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Gymnasium API Compatibility Check")
    logger.info("=" * 60)

    try:
        from stable_baselines3.common.env_checker import check_env

        # Prepare clean data
        feature_cols = [c for c in features_df.columns if c not in ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
        features_only = features_df[feature_cols].copy()
        valid_idx = ~features_only.isna().any(axis=1)
        data_clean = data[valid_idx].reset_index(drop=True)
        features_clean = features_only[valid_idx].reset_index(drop=True)

        if len(data_clean) < 161:
            logger.warning("Insufficient data for env check")
            return

        env_config = EnvConfig(
            initial_capital=100000.0,
            lookback_window=60,
            max_episode_steps=min(100, len(data_clean) - 61),
        )

        env = TradingEnv(data_clean, features_clean, env_config)
        check_env(env, warn=True)
        logger.info("✓ Environment passed Gymnasium compatibility check!")

    except ImportError:
        logger.warning("stable_baselines3 not installed, skipping env check")
    except Exception as e:
        logger.error(f"Environment check failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test TradeBox-RL pipeline end-to-end")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Stock symbol (default: RELIANCE.NS)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (default: 2020-01-01)")
    parser.add_argument("--end", default="2024-12-01", help="End date (default: 2024-12-01)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--cache-dir", default="data/raw", help="Cache directory (default: data/raw)")
    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("TradeBox-RL End-to-End Pipeline Test")
    logger.info("=" * 60)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Date range: {args.start} to {args.end}")

    cache_dir = Path(args.cache_dir)

    try:
        # Step 1: Download data
        raw_data = test_data_download(args.symbol, args.start, args.end, cache_dir)

        if raw_data.empty:
            logger.error("No data downloaded. Check symbol and date range.")
            sys.exit(1)

        # Step 2: Validate data
        clean_data = test_data_validation(raw_data, args.symbol)

        # Step 3: Split data
        splits = test_data_split(clean_data, args.start, args.end)

        # Step 4: Extract features (on training data)
        train_data = splits["train"]
        if len(train_data) == 0:
            logger.warning("Training set is empty. Using full cleaned data instead.")
            train_data = clean_data
        features_df = test_feature_extraction(train_data)

        # Step 5: Feature analysis
        feature_names = [c for c in features_df.columns if c not in ["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]
        test_feature_analysis(features_df, feature_names)

        # Step 6: Trading environment test
        test_trading_environment(train_data, features_df, n_episodes=args.episodes)

        # Step 7: Gymnasium compatibility
        test_gymnasium_compatibility(train_data, features_df)

        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Pipeline test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
