"""Performance metrics calculation for backtesting."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from tradebox.backtest.engine import BacktestResult, Trade


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics from backtest results.

    Computes returns, risk-adjusted metrics, risk metrics, and trading metrics.

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.calculate(backtest_result)
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        >>> print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        >>> print(f"Win Rate: {metrics['win_rate']:.1%}")
    """

    def __init__(self, trading_days_per_year: int = 252) -> None:
        """
        Initialize metrics calculator.

        Args:
            trading_days_per_year: Trading days for annualization (default: 252)
        """
        self.trading_days_per_year = trading_days_per_year

    def calculate(self, result: BacktestResult) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            result: BacktestResult from backtest engine

        Returns:
            Dictionary with all calculated metrics
        """
        logger.info("Calculating performance metrics...")

        metrics = {}

        # Returns metrics
        metrics.update(self._calculate_returns_metrics(result))

        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(result))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(result))

        # Trading metrics
        metrics.update(self._calculate_trading_metrics(result))

        logger.info(
            f"Metrics calculated: Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
            f"Max DD={metrics.get('max_drawdown', 0):.2%}, "
            f"Win Rate={metrics.get('win_rate', 0):.1%}"
        )

        return metrics

    def _calculate_returns_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate returns-based metrics."""
        equity_curve = result.equity_curve
        daily_returns = result.daily_returns

        initial_value = result.config.initial_capital
        final_value = equity_curve.iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # Calculate number of years
        days = len(equity_curve)
        years = days / self.trading_days_per_year

        # CAGR
        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0

        # Annualized return
        annualized_return = daily_returns.mean() * self.trading_days_per_year

        return {
            "total_return": total_return,
            "cagr": cagr,
            "annualized_return": annualized_return,
            "mean_daily_return": daily_returns.mean(),
            "median_daily_return": daily_returns.median(),
        }

    def _calculate_risk_adjusted_metrics(
        self, result: BacktestResult
    ) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        daily_returns = result.daily_returns
        risk_free_rate = result.config.risk_free_rate

        # Daily risk-free rate
        daily_rf = risk_free_rate / self.trading_days_per_year

        # Sharpe Ratio
        excess_returns = daily_returns - daily_rf
        sharpe_ratio = (
            np.sqrt(self.trading_days_per_year) * excess_returns.mean() / excess_returns.std()
            if excess_returns.std() > 0
            else 0.0
        )

        # Sortino Ratio (using downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (
            np.sqrt(self.trading_days_per_year) * (daily_returns.mean() - daily_rf) / downside_std
            if downside_std > 0
            else 0.0
        )

        # Calmar Ratio (CAGR / Max Drawdown)
        cagr = self._calculate_returns_metrics(result)["cagr"]
        max_drawdown = self._calculate_max_drawdown(result.equity_curve)
        calmar_ratio = abs(cagr / max_drawdown) if max_drawdown < 0 else 0.0

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
        }

    def _calculate_risk_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate risk metrics."""
        equity_curve = result.equity_curve
        daily_returns = result.daily_returns

        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Average Drawdown
        drawdowns = self._calculate_drawdowns(equity_curve)
        avg_drawdown = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0

        # Volatility
        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(self.trading_days_per_year)

        # Value at Risk (VaR) 95% and 99%
        var_95 = daily_returns.quantile(0.05)
        var_99 = daily_returns.quantile(0.01)

        # Conditional VaR (CVaR) - average of returns below VaR
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        cvar_99 = daily_returns[daily_returns <= var_99].mean()

        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "daily_volatility": daily_volatility,
            "annualized_volatility": annualized_volatility,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
        }

    def _calculate_trading_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        trades = result.trades

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_trade_duration_days": 0.0,
            }

        # Filter completed trades
        completed_trades = [t for t in trades if t.pnl is not None]
        total_trades = len(completed_trades)

        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_trade_duration_days": 0.0,
            }

        # Win/Loss analysis
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        avg_win = (
            np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        )
        avg_loss = (
            abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0.0
        )

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Profit Factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Average trade duration
        durations = [t.duration_days for t in completed_trades if t.duration_days is not None]
        avg_duration = np.mean(durations) if durations else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "profit_factor": profit_factor,
            "avg_trade_duration_days": avg_duration,
        }

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Maximum drawdown as negative percentage
        """
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()
        return float(max_drawdown)

    def _calculate_drawdowns(self, equity_curve: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Series of drawdowns at each point
        """
        cummax = equity_curve.cummax()
        drawdowns = (equity_curve - cummax) / cummax
        return drawdowns

    def compare_to_benchmark(
        self,
        result: BacktestResult,
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        """
        Compare strategy to benchmark.

        Args:
            result: BacktestResult from strategy
            benchmark_returns: Daily returns of benchmark

        Returns:
            Dictionary with comparison metrics
        """
        strategy_returns = result.daily_returns

        # Align returns
        aligned_strategy, aligned_benchmark = strategy_returns.align(
            benchmark_returns, join="inner"
        )

        if len(aligned_strategy) == 0:
            logger.warning("No overlapping data for benchmark comparison")
            return {}

        # Calculate alpha and beta
        covariance = np.cov(aligned_strategy, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0

        risk_free_rate = result.config.risk_free_rate / self.trading_days_per_year
        alpha = (
            aligned_strategy.mean()
            - risk_free_rate
            - beta * (aligned_benchmark.mean() - risk_free_rate)
        )

        # Annualize alpha
        alpha_annualized = alpha * self.trading_days_per_year

        # Information Ratio
        active_returns = aligned_strategy - aligned_benchmark
        tracking_error = active_returns.std()
        information_ratio = (
            np.sqrt(self.trading_days_per_year) * active_returns.mean() / tracking_error
            if tracking_error > 0
            else 0.0
        )

        # Statistical significance (t-test)
        t_stat, p_value = stats.ttest_1samp(active_returns, 0.0)

        return {
            "alpha": alpha_annualized,
            "beta": beta,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error * np.sqrt(self.trading_days_per_year),
            "correlation": float(np.corrcoef(aligned_strategy, aligned_benchmark)[0, 1]),
            "outperformance_pvalue": p_value,
            "is_significant": p_value < 0.05,
        }
