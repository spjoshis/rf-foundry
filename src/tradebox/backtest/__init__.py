"""
Backtesting module for RL trading agents.

This module provides comprehensive backtesting capabilities for trained
RL agents, including performance metrics calculation, visualization,
and report generation.

Key Components:
    - BacktestEngine: Run agent on historical data
    - MetricsCalculator: Calculate performance metrics
    - BacktestAnalyzer: Create visualizations
    - BacktestReport: Generate reports

Example:
    >>> from tradebox.backtest import BacktestEngine, MetricsCalculator, BacktestAnalyzer
    >>> from tradebox.agents import PPOAgent
    >>>
    >>> # Load trained agent
    >>> agent = PPOAgent.load("models/ppo_best.zip")
    >>>
    >>> # Run backtest
    >>> engine = BacktestEngine()
    >>> result = engine.run(agent, test_data, test_features, symbol="RELIANCE.NS")
    >>>
    >>> # Calculate metrics
    >>> calculator = MetricsCalculator()
    >>> metrics = calculator.calculate(result)
    >>>
    >>> # Create visualizations
    >>> analyzer = BacktestAnalyzer()
    >>> analyzer.create_dashboard(result, "reports/backtest.png", metrics)
    >>>
    >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    >>> print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
"""

from tradebox.backtest.analyzer import BacktestAnalyzer
from tradebox.backtest.config import BacktestConfig
from tradebox.backtest.engine import BacktestEngine, BacktestResult, Trade
from tradebox.backtest.metrics import MetricsCalculator
from tradebox.backtest.report import BacktestReport

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "MetricsCalculator",
    "BacktestAnalyzer",
    "BacktestReport",
]
