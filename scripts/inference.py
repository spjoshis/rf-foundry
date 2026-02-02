#!/usr/bin/env python3
"""
Run inference with trained RL trading agent.

This script provides a lightweight interface for running predictions with
a trained PPO agent on real market data. Unlike the backtesting script,
this focuses on step-by-step inference with detailed logging of each
prediction and the agent's decision-making process.

Use Cases:
    - Test agent behavior on specific dates
    - Debug agent decisions with verbose logging
    - Validate model loading and prediction pipeline
    - Quick sanity checks before deployment

Usage:
    # Run for N steps with detailed logging
    python scripts/inference.py \\
        --model models/exp007_active_trading_20251219_235451 \\
        --symbol RELIANCE.NS \\
        --steps 10 \\
        --verbose

    # Run full episode on specific date
    python scripts/inference.py \\
        --model models/exp007_active_trading_20251219_235451 \\
        --symbol RELIANCE.NS \\
        --start 2024-12-01 \\
        --end 2024-12-31 \\
        --episode

    # Test intraday model
    python scripts/inference.py \\
        --model models/intraday_model \\
        --symbol ^NSEI \\
        --env-type intraday \\
        --steps 20
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from tradebox.agents import PPOAgent
from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.env import EnvConfig, IntradayEnvConfig, IntradayTradingEnv, TradingEnv
from tradebox.features.technical import TechnicalFeatures


def setup_logging(verbose: bool) -> None:
    """
    Setup loguru logging with appropriate level.

    Args:
        verbose: If True, enable DEBUG level logging, otherwise INFO.
    """
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def format_action_name(action) -> str:
    """
    Convert action integer to human-readable name.

    Args:
        action: Action integer or numpy array (0=Hold, 1=Buy, 2=Sell)

    Returns:
        Action name as string.
    """
    # Handle numpy arrays
    if isinstance(action, np.ndarray):
        action = int(action.item())
    else:
        action = int(action)

    action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
    return action_map.get(action, "Unknown")


def format_currency(value: float) -> str:
    """
    Format value as Indian currency.

    Args:
        value: Numeric value to format.

    Returns:
        Formatted string with rupee symbol and commas.
    """
    return f"₹{value:,.2f}"


def log_simple(
    step: int,
    action: int,
    reward: float,
    info: Dict[str, Any],
) -> None:
    """
    Log prediction in simple format.

    Args:
        step: Step number
        action: Action taken (0=Hold, 1=Buy, 2=Sell)
        reward: Reward received
        info: Info dictionary from environment
    """
    action_name = format_action_name(action)
    portfolio_value = info.get("portfolio_value", 0)
    sign = "+" if reward >= 0 else ""

    logger.info(
        f"Step {step:3d}: Action={action} ({action_name:4s}) | "
        f"Reward: {sign}{reward:8.2f} | "
        f"Portfolio: {format_currency(portfolio_value)}"
    )


def log_detailed(
    step: int,
    action: int,
    obs: Dict[str, np.ndarray],
    info: Dict[str, Any],
    data: pd.DataFrame,
    current_step: int,
    reward: float,
) -> None:
    """
    Log prediction in detailed format with market state.

    Args:
        step: Step number
        action: Action taken
        obs: Observation dictionary
        info: Info dictionary from environment
        data: Market data DataFrame
        current_step: Current step in data
        reward: Reward received
    """
    # Get current date and OHLCV
    current_date = pd.to_datetime(data.iloc[current_step]["Date"]).strftime("%Y-%m-%d")
    ohlcv = data.iloc[current_step]
    open_price = float(ohlcv["Open"])
    high_price = float(ohlcv["High"])
    low_price = float(ohlcv["Low"])
    close_price = float(ohlcv["Close"])
    volume = int(ohlcv["Volume"])

    # Get portfolio state
    portfolio = obs.get("portfolio", np.array([0, 0, 0, 0]))
    cash = info.get("cash", 0)
    position = info.get("position", 0)
    portfolio_value = info.get("portfolio_value", 0)

    # Print formatted output
    logger.info("━" * 60)
    logger.info(f"Step {step} | {current_date}")
    logger.info("━" * 60)
    logger.info("Market State:")
    logger.info(
        f"  Price: {format_currency(close_price)} "
        f"(O: {open_price:.2f} | H: {high_price:.2f} | "
        f"L: {low_price:.2f})"
    )
    logger.info(f"  Volume: {volume:,}")

    logger.info("")
    logger.info("Portfolio State:")
    logger.info(f"  Cash: {format_currency(cash)} | Position: {position} shares")
    logger.info(f"  Portfolio Value: {format_currency(portfolio_value)}")

    logger.info("")
    logger.info(f"Agent Decision: {format_action_name(action).upper()} (action={action})")

    logger.info("")
    logger.info("Result:")
    sign = "+" if reward >= 0 else ""
    logger.info(f"  Reward: {sign}{reward:.2f}")
    logger.info("")


def save_json_log(
    predictions: list,
    filepath: str,
    args: argparse.Namespace,
    summary: Dict[str, Any],
) -> None:
    """
    Save predictions to JSON file.

    Args:
        predictions: List of prediction dictionaries
        filepath: Path to save JSON file
        args: Parsed command-line arguments
        summary: Summary statistics dictionary
    """
    output = {
        "metadata": {
            "model": args.model,
            "symbol": args.symbol,
            "start_date": args.start,
            "end_date": args.end,
            "env_type": args.env_type,
            "total_steps": len(predictions),
            "timestamp": datetime.now().isoformat(),
        },
        "predictions": predictions,
        "summary": summary,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Predictions saved to: {filepath}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with trained RL trading agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (with or without .zip extension)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol (e.g., RELIANCE.NS, ^NSEI)",
    )

    # Data selection arguments
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD, default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD, default: 2024-12-31)",
    )

    # Execution mode
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps to run (default: 10 if --episode not set)",
    )
    parser.add_argument(
        "--episode",
        action="store_true",
        help="Run complete episode until done/truncated",
    )

    # Environment configuration
    parser.add_argument(
        "--env-type",
        type=str,
        choices=["eod", "intraday"],
        default="eod",
        help="Environment type (default: eod)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=60,
        help="Lookback window size (default: 60)",
    )

    # Prediction settings
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (exploration enabled)",
    )

    # Output options
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["simple", "detailed", "json"],
        default="detailed",
        help="Logging format (default: detailed)",
    )
    parser.add_argument(
        "--save-log",
        type=str,
        default=None,
        help="Save predictions to JSON file (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    # Utility options
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for inference (default: auto)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main inference entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("TradeBox-RL Inference")
    logger.info("=" * 60)
    logger.info("")

    try:
        # 1. Load model
        logger.info(f"Loading model: {args.model}")
        model_path = args.model
        # Remove .zip extension if present for consistent loading
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]

        try:
            agent = PPOAgent.load(model_path, env=None, device=args.device)
            logger.info(f"Model loaded successfully: {agent.__class__.__name__}")
        except FileNotFoundError:
            logger.error(f"Model not found: {model_path}")
            logger.info("Available models in models/:")
            models_dir = Path("models")
            if models_dir.exists():
                for model_file in models_dir.glob("**/*.zip"):
                    logger.info(f"  - {model_file}")
            else:
                logger.info("  models/ directory not found")
            return 1
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Model may be corrupted or incompatible")
            return 1

        # 2. Load data
        logger.info(f"Loading data for {args.symbol}...")
        loader = YahooDataLoader(cache_dir="cache", use_cache=True)

        try:
            data = loader.download(args.symbol, args.start, args.end)
            if len(data) == 0:
                logger.error(
                    f"No data available for {args.symbol} "
                    f"in range {args.start} to {args.end}"
                )
                logger.info("Try adjusting date range or check symbol validity")
                return 1
            logger.info(f"Loaded {len(data)} days of data")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            logger.info("Possible reasons:")
            logger.info("  - Invalid symbol (use Yahoo Finance format, e.g., RELIANCE.NS)")
            logger.info("  - Network connectivity issues")
            logger.info("  - Market closed for date range")
            return 1

        # 3. Extract features
        logger.info("Extracting technical features...")
        extractor = TechnicalFeatures()
        features = extractor.extract(data)
        logger.info(f"Extracted {len(features.columns)} features")

        # Check minimum data requirement
        min_required = args.lookback_window + 10
        if len(data) < min_required:
            logger.error(
                f"Insufficient data: {len(data)} rows, need at least {min_required}"
            )
            logger.info(f"Available data range: {data.index[0]} to {data.index[-1]}")
            logger.info("Solutions:")
            logger.info("  1. Expand date range (--start, --end)")
            logger.info("  2. Reduce lookback window (--lookback-window)")
            return 1

        # 4. Create environment
        logger.info(f"Creating {args.env_type.upper()} environment...")

        # Calculate reasonable max_episode_steps based on available data and requested steps
        available_steps = len(data) - args.lookback_window
        if args.steps:
            max_episode_steps = min(available_steps, args.steps + 10)  # Add buffer
        elif args.episode:
            max_episode_steps = available_steps
        else:
            max_episode_steps = min(available_steps, 20)  # Default 20 for short inference

        try:
            if args.env_type == "eod":
                config = EnvConfig(
                    initial_capital=args.initial_capital,
                    lookback_window=args.lookback_window,
                    max_episode_steps=max_episode_steps,
                )
                env = TradingEnv(data, features, config)
            else:  # intraday
                config = IntradayEnvConfig(
                    initial_capital=args.initial_capital,
                    lookback_window=args.lookback_window,
                    bar_interval_minutes=5,
                    bars_per_session=75,
                    sessions_per_episode=10,
                    force_close_eod=True,
                )
                env = IntradayTradingEnv(data, features, config)

            logger.info(f"Environment created successfully")
            logger.info(f"  Observation space: {env.observation_space}")
            logger.info(f"  Action space: {env.action_space}")
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            logger.info("Check that data and features are aligned")
            return 1

        # 5. Reset environment
        obs, info = env.reset()
        logger.info("")
        logger.info("Environment ready. Starting inference...")
        logger.info("")

        # 6. Determine execution mode
        if args.steps:
            max_steps = args.steps
        elif args.episode:
            max_steps = float("inf")
        else:
            max_steps = 10  # Default

        # Use deterministic policy by default (unless --stochastic flag)
        deterministic = not args.stochastic
        if args.stochastic:
            logger.info("Using stochastic policy (exploration enabled)")
        else:
            logger.info("Using deterministic policy")
        logger.info("")

        # 7. Run inference loop
        predictions_log = []
        step_count = 0
        total_reward = 0.0
        episode_done = False

        while step_count < max_steps and not episode_done:
            # Predict action
            try:
                action, _ = agent.predict(obs, deterministic=deterministic)
                # Convert numpy array to int
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                else:
                    action = int(action)
            except Exception as e:
                logger.error(f"Prediction failed at step {step_count}: {e}")
                logger.debug(f"Observation: {obs}")
                logger.info("Model may be incompatible with environment observation space")
                return 1

            # Log prediction BEFORE stepping (to show decision-making)
            if args.output_format == "simple":
                # Will log after step to include reward
                pass
            elif args.output_format == "detailed":
                log_detailed(
                    step_count + 1,
                    action,
                    obs,
                    info,
                    data,
                    env.current_step,
                    0.0,  # Placeholder, will show actual reward after step
                )

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)

            # Track metrics
            total_reward += reward
            step_count += 1

            # Log simple format AFTER step (to include reward)
            if args.output_format == "simple":
                log_simple(step_count, action, reward, info)

            # Save for JSON output
            if args.save_log or args.output_format == "json":
                # Get current market data for logging
                current_date = pd.to_datetime(
                    data.iloc[env.current_step - 1]["Date"]
                ).strftime("%Y-%m-%d")
                ohlcv = data.iloc[env.current_step - 1]

                predictions_log.append(
                    {
                        "step": step_count,
                        "date": current_date,
                        "action": int(action),
                        "action_name": format_action_name(action),
                        "reward": float(reward),
                        "portfolio_value": float(info.get("portfolio_value", 0)),
                        "cash": float(info.get("cash", 0)),
                        "position": int(info.get("position", 0)),
                        "price": {
                            "open": float(ohlcv["Open"]),
                            "high": float(ohlcv["High"]),
                            "low": float(ohlcv["Low"]),
                            "close": float(ohlcv["Close"]),
                            "volume": int(ohlcv["Volume"]),
                        },
                    }
                )

            # Check termination
            if terminated or truncated:
                logger.info("")
                logger.info(f"Episode ended at step {step_count}")
                if terminated:
                    logger.info("Reason: Episode terminated (done=True)")
                if truncated:
                    logger.info("Reason: Episode truncated (max steps reached)")
                episode_done = True

        # 8. Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Inference Complete")
        logger.info("=" * 60)
        logger.info(f"Total steps: {step_count}")
        logger.info(f"Total reward: {total_reward:.2f}")
        logger.info(f"Final portfolio value: {format_currency(info.get('portfolio_value', 0))}")
        logger.info(f"Total trades: {info.get('total_trades', 0)}")
        logger.info("")

        # 9. Save log if requested
        if args.save_log or args.output_format == "json":
            summary = {
                "total_steps": step_count,
                "total_reward": float(total_reward),
                "final_portfolio_value": float(info.get("portfolio_value", 0)),
                "total_trades": int(info.get("total_trades", 0)),
            }

            if args.save_log:
                save_json_log(predictions_log, args.save_log, args, summary)
            elif args.output_format == "json":
                # Default JSON filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_file = f"predictions_{args.symbol}_{timestamp}.json"
                save_json_log(predictions_log, json_file, args, summary)

        return 0

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Inference interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
