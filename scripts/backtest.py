#!/usr/bin/env python3
"""
Run backtest on trained RL agent.

This script loads a trained agent and runs it on historical test data,
calculating comprehensive performance metrics and generating reports.

Usage:
    # Basic backtest with default settings
    python scripts/backtest.py --model models/ppo_best.zip \\
        --symbol RELIANCE.NS --start 2022-01-01 --end 2024-12-31

    # Backtest with custom output directory
    python scripts/backtest.py --model models/ppo_best.zip \\
        --symbol RELIANCE.NS --output-dir reports/exp001

    # Generate full dashboard
    python scripts/backtest.py --model models/ppo_best.zip \\
        --symbol RELIANCE.NS --dashboard

Example:
    $ python scripts/backtest.py --model models/ppo_best.zip --symbol RELIANCE.NS

    Output:
    - reports/backtest_summary.txt
    - reports/backtest.json
    - reports/equity_curve.png
    - reports/drawdown.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from loguru import logger

from tradebox.agents import PPOAgent
from tradebox.backtest import (
    BacktestAnalyzer,
    BacktestConfig,
    BacktestEngine,
    BacktestReport,
    MetricsCalculator,
)
from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.features.technical import TechnicalFeatures, FeatureConfig
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
from tradebox.features.regime import RegimeConfig
from tradebox.data.loaders.fundamental_loader import FundamentalConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backtest trained RL trading agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Stock symbol to backtest (e.g., RELIANCE.NS)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate comprehensive dashboard",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to experiment config YAML (to match feature extraction settings)",
    )

    return parser.parse_args()


def main() -> int:
    """
    Main backtest entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Setup logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    logger.info("=" * 60)
    logger.info("TradeBox-RL Backtest")
    logger.info("=" * 60)

    try:
        # Load model
        logger.info(f"Loading model from: {args.model}")
        agent = PPOAgent.load(args.model, env=None)
        logger.info(f"Model loaded: {agent.__class__.__name__}")

        # Load data
        cache_dir = 'cache'
        logger.info(f"Loading data for {args.symbol}...")
        loader = YahooDataLoader(cache_dir=cache_dir, use_cache=True)
        data = loader.download(args.symbol, args.start, args.end)
        logger.info(f"Loaded {len(data)} days of data")

        # Set Date as index for FeatureExtractor compatibility
        if 'Date' in data.columns:
            data = data.set_index('Date')
            # Remove timezone info to avoid comparison issues
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

        # Extract features using same config as training
        logger.info("Extracting technical + regime + fundamental features...")

        # Load feature config from file if provided, otherwise use defaults
        if args.config:
            logger.info(f"Loading feature config from: {args.config}")
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
            feature_config_dict = config_dict.get("features", {})
        else:
            logger.warning("No config file provided, using default feature config")
            feature_config_dict = {}

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

        # Create FeatureExtractor with all configs
        extractor_config = FeatureExtractorConfig(
            technical=technical_config,
            regime=regime_config,
            fundamental=fundamental_config
        )
        feature_extractor = FeatureExtractor(extractor_config)

        # Extract features (fit_normalize=False since we're using trained model)
        features = feature_extractor.extract(
            symbol=args.symbol,
            price_data=data,
            fit_normalize=False
        )
        logger.info(f"Extracted {len(features.columns)} features")

        # Reset index to make Date a column again (backtest engine expects Date column)
        if data.index.name == 'Date':
            data = data.reset_index()
        if features.index.name == 'Date':
            features = features.reset_index(drop=True)  # Features don't need Date column

        # Create backtest config
        config = BacktestConfig(initial_capital=args.initial_capital)

        # Create environment config from YAML if provided
        env_config = None
        if args.config:
            logger.info("Loading environment config from YAML...")
            from tradebox.env import EnvConfig
            from tradebox.env.costs import CostConfig
            from tradebox.env.rewards import RewardConfig
            from tradebox.env.action_mask import ActionMaskConfig

            env_dict = config_dict.get("env", {})

            # Create cost config
            cost_config = None
            if "cost_config" in env_dict:
                cost_config = CostConfig(**env_dict["cost_config"])

            # Create reward config
            reward_config = None
            if "reward_config" in env_dict:
                reward_config = RewardConfig(**env_dict["reward_config"])

            # Create action mask config
            action_mask_config = None
            if "action_mask_config" in env_dict:
                action_mask_config = ActionMaskConfig(**env_dict["action_mask_config"])
                logger.info(f"Action masking enabled: {action_mask_config.enabled}")

            # Create full env config
            env_config = EnvConfig(
                initial_capital=env_dict.get("initial_capital", args.initial_capital),
                lookback_window=env_dict.get("lookback_window", 60),
                max_episode_steps=len(data) - 61,  # Full test period
                cost_config=cost_config,
                reward_config=reward_config,
                action_mask_config=action_mask_config,
            )
            logger.info(f"Environment config loaded successfully")

        # Run backtest
        logger.info("Running backtest...")
        engine = BacktestEngine(config)
        result = engine.run(
            agent=agent,
            data=data,
            features=features,
            symbol=args.symbol,
            env_config=env_config,  # ← PASS ENV CONFIG
        )

        # Calculate metrics
        logger.info("Calculating performance metrics...")
        calculator = MetricsCalculator()
        metrics = calculator.calculate(result)

        # Update result with metrics
        result.metrics = metrics

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate reports
        logger.info("Generating reports...")
        report = BacktestReport()

        # Print summary to console
        report.print_summary(result, metrics)

        # Save JSON
        json_path = output_dir / "backtest.json"
        report.save_json(result, metrics, json_path)

        # Save text summary
        summary_path = output_dir / "backtest_summary.txt"
        report.save_text_summary(result, metrics, summary_path)

        # Create visualizations
        analyzer = BacktestAnalyzer()

        logger.info("Creating visualizations...")

        # Equity curve
        analyzer.plot_equity_curve(
            result,
            save_path=output_dir / "equity_curve.png",
        )

        # Drawdown
        analyzer.plot_drawdown(
            result,
            save_path=output_dir / "drawdown.png",
        )

        # Returns distribution
        analyzer.plot_returns_distribution(
            result,
            save_path=output_dir / "returns_distribution.png",
        )

        # Trade analysis
        if result.trades:
            analyzer.plot_trade_analysis(
                result,
                save_path=output_dir / "trade_analysis.png",
            )

        # Dashboard (if requested)
        if args.dashboard:
            logger.info("Creating comprehensive dashboard...")
            analyzer.create_dashboard(
                result,
                output_path=output_dir / "dashboard.png",
                metrics=metrics,
            )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Backtest Complete!")
        logger.info("=" * 60)
        logger.info(f"Reports saved to: {output_dir.absolute()}")
        logger.info("")
        logger.info("Key Metrics:")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
        logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
        logger.info("")

        return 0

    except KeyboardInterrupt:
        logger.warning("Backtest interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
