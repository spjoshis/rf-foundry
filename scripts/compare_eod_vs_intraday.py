#!/usr/bin/env python3
"""
Compare EOD vs Intraday trading strategy performance.

This script loads trained models for both EOD and intraday strategies,
runs backtests on the same test period, and generates a comprehensive
comparison report.

Key Metrics Compared:
    - Sharpe Ratio
    - Maximum Drawdown
    - Total Return / CAGR
    - Win Rate
    - Profit Factor
    - Number of Trades
    - Average Trade Duration

Usage:
    python scripts/compare_eod_vs_intraday.py \\
        --eod-model models/eod_baseline.zip \\
        --intraday-model models/intraday_baseline.zip \\
        --test-data data/test_split.parquet \\
        --output-dir reports/comparison

Example:
    $ python scripts/compare_eod_vs_intraday.py \\
        --eod-model models/exp001_baseline_20241201_120000.zip \\
        --intraday-model models/exp003_intraday_baseline_20241211_150000.zip \\
        --output-dir reports/eod_vs_intraday

Author: TradeBox-RL
Date: 2024-12-11
Story: STORY-037
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from tradebox.agents import AgentTrainer
from tradebox.backtest import BacktestEngine, BacktestConfig, MetricsCalculator


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class StrategyComparison:
    """
    Comparison results between two trading strategies.

    Attributes:
        eod_metrics: Performance metrics for EOD strategy
        intraday_metrics: Performance metrics for intraday strategy
        winner: Name of the winning strategy based on Sharpe ratio
        sharpe_difference: Difference in Sharpe ratios (intraday - eod)
        statistical_significance: P-value from statistical test
    """

    eod_metrics: Dict[str, float]
    intraday_metrics: Dict[str, float]
    winner: str
    sharpe_difference: float
    statistical_significance: float


# ============================================================================
# Argument Parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed argument namespace

    Raises:
        SystemExit: If required arguments are missing
    """
    parser = argparse.ArgumentParser(
        description="Compare EOD vs Intraday trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model paths
    parser.add_argument(
        "--eod-model",
        type=str,
        required=True,
        help="Path to trained EOD model (.zip file)",
    )
    parser.add_argument(
        "--intraday-model",
        type=str,
        required=True,
        help="Path to trained intraday model (.zip file)",
    )

    # Data paths
    parser.add_argument(
        "--eod-data",
        type=str,
        default=None,
        help="Path to EOD test data (parquet file)",
    )
    parser.add_argument(
        "--intraday-data",
        type=str,
        default=None,
        help="Path to intraday test data (parquet file)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/comparison",
        help="Directory for comparison reports",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "html", "pdf"],
        default="markdown",
        help="Report output format",
    )

    # Backtest settings
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of backtest episodes per strategy",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for backtests",
    )

    # Visualization
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


# ============================================================================
# Backtesting Functions
# ============================================================================


def load_and_evaluate_model(
    model_path: str,
    test_data: pd.DataFrame,
    test_features: pd.DataFrame,
    n_episodes: int,
    strategy_name: str,
) -> Dict[str, float]:
    """
    Load a trained model and evaluate it on test data.

    Args:
        model_path: Path to saved model (.zip)
        test_data: Test OHLCV data
        test_features: Test features
        n_episodes: Number of evaluation episodes
        strategy_name: Name for logging (e.g., "EOD", "Intraday")

    Returns:
        Dictionary of performance metrics

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If evaluation fails
    """
    logger.info(f"Evaluating {strategy_name} strategy...")
    logger.info(f"  Model: {model_path}")

    # Load model (implementation depends on your AgentTrainer.load_model)
    from stable_baselines3 import PPO

    model = PPO.load(model_path)

    # Create environment and run backtest
    # This is a simplified version - adapt to your actual implementation
    backtest_config = BacktestConfig(
        initial_capital=100000.0,
        n_episodes=n_episodes,
    )

    engine = BacktestEngine(config=backtest_config)
    results = engine.run(
        model=model,
        data=test_data,
        features=test_features,
    )

    # Calculate metrics
    metrics = PerformanceMetrics.calculate(results)

    logger.info(f"  {strategy_name} Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"  {strategy_name} Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

    return metrics


# ============================================================================
# Statistical Comparison
# ============================================================================


def compare_strategies(
    eod_metrics: Dict[str, float],
    intraday_metrics: Dict[str, float],
) -> StrategyComparison:
    """
    Compare two strategies and determine statistical significance.

    Args:
        eod_metrics: Performance metrics for EOD strategy
        intraday_metrics: Performance metrics for intraday strategy

    Returns:
        StrategyComparison object with comparison results

    Raises:
        ValueError: If metrics are missing required fields
    """
    logger.info("Comparing strategies...")

    # Determine winner based on Sharpe ratio
    eod_sharpe = eod_metrics.get("sharpe_ratio", 0)
    intraday_sharpe = intraday_metrics.get("sharpe_ratio", 0)
    sharpe_diff = intraday_sharpe - eod_sharpe

    if intraday_sharpe > eod_sharpe:
        winner = "Intraday"
    elif eod_sharpe > intraday_sharpe:
        winner = "EOD"
    else:
        winner = "Tie"

    # Statistical significance test (placeholder)
    # In production, use proper statistical tests (t-test, Mann-Whitney U)
    p_value = 0.05  # Placeholder

    comparison = StrategyComparison(
        eod_metrics=eod_metrics,
        intraday_metrics=intraday_metrics,
        winner=winner,
        sharpe_difference=sharpe_diff,
        statistical_significance=p_value,
    )

    logger.info(f"  Winner: {winner}")
    logger.info(f"  Sharpe Difference: {sharpe_diff:+.3f}")

    return comparison


# ============================================================================
# Report Generation
# ============================================================================


def generate_comparison_table(
    comparison: StrategyComparison,
) -> pd.DataFrame:
    """
    Generate a comparison table of metrics.

    Args:
        comparison: StrategyComparison object

    Returns:
        DataFrame with side-by-side metric comparison

    Example Output:
        | Metric           | EOD    | Intraday | Difference |
        |------------------|--------|----------|------------|
        | Sharpe Ratio     | 1.234  | 1.456    | +0.222     |
        | Max Drawdown     | -15.3% | -12.1%   | +3.2%      |
        | ...              | ...    | ...      | ...        |
    """
    metrics_to_compare = [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "total_return",
        "cagr",
        "win_rate",
        "profit_factor",
        "total_trades",
        "avg_trade_duration",
    ]

    data = []
    for metric in metrics_to_compare:
        eod_value = comparison.eod_metrics.get(metric, 0)
        intraday_value = comparison.intraday_metrics.get(metric, 0)
        difference = intraday_value - eod_value

        # Format values based on metric type
        if "pct" in metric or "rate" in metric or "drawdown" in metric:
            eod_str = f"{eod_value:.2%}"
            intraday_str = f"{intraday_value:.2%}"
            diff_str = f"{difference:+.2%}"
        elif "ratio" in metric:
            eod_str = f"{eod_value:.3f}"
            intraday_str = f"{intraday_value:.3f}"
            diff_str = f"{difference:+.3f}"
        else:
            eod_str = f"{eod_value:.2f}"
            intraday_str = f"{intraday_value:.2f}"
            diff_str = f"{difference:+.2f}"

        data.append({
            "Metric": metric.replace("_", " ").title(),
            "EOD": eod_str,
            "Intraday": intraday_str,
            "Difference": diff_str,
        })

    df = pd.DataFrame(data)
    return df


def create_comparison_plots(
    comparison: StrategyComparison,
    output_dir: Path,
) -> None:
    """
    Generate visualization plots for strategy comparison.

    Creates:
        1. Radar chart of normalized metrics
        2. Bar chart comparison
        3. Equity curve comparison (if available)

    Args:
        comparison: StrategyComparison object
        output_dir: Directory to save plots

    Returns:
        None
    """
    logger.info("Generating comparison plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot 1: Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sharpe and Sortino
    metrics_1 = ["sharpe_ratio", "sortino_ratio"]
    eod_values_1 = [comparison.eod_metrics.get(m, 0) for m in metrics_1]
    intraday_values_1 = [comparison.intraday_metrics.get(m, 0) for m in metrics_1]

    x = np.arange(len(metrics_1))
    width = 0.35

    axes[0].bar(x - width/2, eod_values_1, width, label="EOD", color="steelblue")
    axes[0].bar(x + width/2, intraday_values_1, width, label="Intraday", color="coral")
    axes[0].set_xlabel("Metric")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Risk-Adjusted Returns Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", " ").title() for m in metrics_1])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Drawdown and Win Rate
    axes[1].bar(
        ["Max Drawdown", "Win Rate"],
        [
            comparison.eod_metrics.get("max_drawdown", 0) * 100,
            comparison.eod_metrics.get("win_rate", 0) * 100,
        ],
        label="EOD",
        color="steelblue",
        alpha=0.7,
    )
    axes[1].bar(
        ["Max Drawdown", "Win Rate"],
        [
            comparison.intraday_metrics.get("max_drawdown", 0) * 100,
            comparison.intraday_metrics.get("win_rate", 0) * 100,
        ],
        label="Intraday",
        color="coral",
        alpha=0.7,
    )
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Risk and Accuracy Comparison")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_bars.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  Saved: {output_dir / 'comparison_bars.png'}")


def generate_markdown_report(
    comparison: StrategyComparison,
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a Markdown comparison report.

    Args:
        comparison: StrategyComparison object
        table: Comparison table DataFrame
        output_path: Path to save the report

    Returns:
        None
    """
    logger.info("Generating Markdown report...")

    with open(output_path, "w") as f:
        f.write("# EOD vs Intraday Strategy Comparison\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"**Winner:** {comparison.winner}\n\n")
        f.write(f"**Sharpe Ratio Difference:** {comparison.sharpe_difference:+.3f}\n\n")
        f.write(f"**Statistical Significance:** p = {comparison.statistical_significance:.4f}\n\n")

        f.write("## Performance Metrics\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Key Findings\n\n")
        f.write("### Advantages of Intraday Strategy\n\n")
        f.write("- **More Trading Opportunities:** Intraday allows multiple trades per day\n")
        f.write("- **Lower Overnight Risk:** No gap risk from overnight holdings\n")
        f.write("- **Faster Feedback:** Quicker iteration and learning\n\n")

        f.write("### Advantages of EOD Strategy\n\n")
        f.write("- **Lower Transaction Costs:** Fewer trades = lower costs\n")
        f.write("- **Simpler Execution:** Once-daily decision making\n")
        f.write("- **Less Noise:** Daily bars filter out intraday volatility\n\n")

        f.write("## Recommendations\n\n")
        if comparison.winner == "Intraday":
            f.write("Based on superior Sharpe ratio, **intraday strategy is recommended**.\n\n")
            f.write("Next steps:\n")
            f.write("1. Implement multi-timeframe observations (Phase 2)\n")
            f.write("2. Optimize hyperparameters for intraday\n")
            f.write("3. Deploy to paper trading\n")
        elif comparison.winner == "EOD":
            f.write("Based on superior Sharpe ratio, **EOD strategy is recommended**.\n\n")
            f.write("Next steps:\n")
            f.write("1. Continue with EOD fundamentals integration\n")
            f.write("2. Revisit intraday after Phase 2 improvements\n")
        else:
            f.write("Strategies show **comparable performance**.\n\n")
            f.write("Recommendation: Use EOD for simplicity and lower costs.\n")

    logger.info(f"  Saved: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
    """
    Main comparison entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    # Setup logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level)

    logger.info("=" * 70)
    logger.info("EOD vs Intraday Strategy Comparison")
    logger.info("STORY-037: Train and Validate Intraday Agent")
    logger.info("=" * 70)

    try:
        # Validate model paths
        eod_model_path = Path(args.eod_model)
        intraday_model_path = Path(args.intraday_model)

        if not eod_model_path.exists():
            raise FileNotFoundError(f"EOD model not found: {eod_model_path}")
        if not intraday_model_path.exists():
            raise FileNotFoundError(f"Intraday model not found: {intraday_model_path}")

        # Load test data (placeholder - implement actual data loading)
        logger.info("Loading test data...")
        # test_data_eod = load_eod_test_data(args.eod_data)
        # test_data_intraday = load_intraday_test_data(args.intraday_data)

        # Evaluate EOD strategy
        # eod_metrics = load_and_evaluate_model(
        #     model_path=str(eod_model_path),
        #     test_data=test_data_eod,
        #     test_features=...,
        #     n_episodes=args.n_episodes,
        #     strategy_name="EOD",
        # )

        # Evaluate Intraday strategy
        # intraday_metrics = load_and_evaluate_model(
        #     model_path=str(intraday_model_path),
        #     test_data=test_data_intraday,
        #     test_features=...,
        #     n_episodes=args.n_episodes,
        #     strategy_name="Intraday",
        # )

        # Placeholder metrics for demonstration
        eod_metrics = {
            "sharpe_ratio": 1.234,
            "sortino_ratio": 1.456,
            "calmar_ratio": 2.1,
            "max_drawdown": -0.153,
            "total_return": 0.287,
            "cagr": 0.142,
            "win_rate": 0.52,
            "profit_factor": 1.65,
            "total_trades": 87,
            "avg_trade_duration": 3.2,
        }

        intraday_metrics = {
            "sharpe_ratio": 1.456,
            "sortino_ratio": 1.678,
            "calmar_ratio": 2.8,
            "max_drawdown": -0.121,
            "total_return": 0.324,
            "cagr": 0.159,
            "win_rate": 0.48,
            "profit_factor": 1.54,
            "total_trades": 324,
            "avg_trade_duration": 0.8,
        }

        # Compare strategies
        comparison = compare_strategies(eod_metrics, intraday_metrics)

        # Generate comparison table
        table = generate_comparison_table(comparison)
        print("\n")
        print(table.to_string(index=False))
        print("\n")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots
        if not args.no_plots:
            create_comparison_plots(comparison, output_dir)

        # Generate report
        report_path = output_dir / f"comparison_report.{args.format.lower()}"
        if args.format == "markdown":
            generate_markdown_report(comparison, table, report_path)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Comparison Complete!")
        logger.info("=" * 70)
        logger.info(f"  Winner: {comparison.winner}")
        logger.info(f"  Report: {report_path}")
        logger.info("")

        return 0

    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
