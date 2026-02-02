#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
from tradebox.features.technical import TechnicalFeatures


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

        # Extract features
        logger.info("Extracting technical features...")
        feature_extractor = TechnicalFeatures()
        features = feature_extractor.extract(data)
        logger.info(f"Extracted {len(features.columns)} features")

        # Create backtest config
        config = BacktestConfig(initial_capital=args.initial_capital)

        # Run backtest
        logger.info("Running backtest...")
        engine = BacktestEngine(config)
        result = engine.run(
            agent=agent,
            data=data,
            features=features,
            symbol=args.symbol,
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
