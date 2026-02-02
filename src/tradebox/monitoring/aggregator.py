"""Metrics aggregation and computation."""

from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from tradebox.monitoring.store import MetricsStore


class MetricsAggregator:
    """
    Computes aggregated metrics and summaries.

    Provides methods for computing derived metrics like Sharpe ratio,
    max drawdown, win rate, etc. from raw metrics data.

    Example:
        >>> store = MetricsStore("data/metrics.db")
        >>> aggregator = MetricsAggregator(store)
        >>>
        >>> # Compute Sharpe ratio
        >>> sharpe = aggregator.compute_sharpe_ratio(returns, periods_per_year=252)
        >>>
        >>> # Compute daily summary
        >>> summary = aggregator.compute_daily_summary(date.today())
    """

    def __init__(self, store: MetricsStore):
        """
        Initialize metrics aggregator.

        Args:
            store: MetricsStore instance

        Example:
            >>> store = MetricsStore()
            >>> aggregator = MetricsAggregator(store)
        """
        self.store = store
        logger.info("MetricsAggregator initialized")

    def compute_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.06,
    ) -> float:
        """
        Compute annualized Sharpe ratio.

        Args:
            returns: Series of returns (not percentages)
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
            risk_free_rate: Annual risk-free rate (default: 6%)

        Returns:
            Annualized Sharpe ratio

        Example:
            >>> returns = pd.Series([0.01, -0.005, 0.02, 0.015])
            >>> sharpe = aggregator.compute_sharpe_ratio(returns)
        """
        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to period rate
        rf_period = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = returns - rf_period

        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0

        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(
            periods_per_year
        )

        return float(sharpe)

    def compute_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.06,
    ) -> float:
        """
        Compute annualized Sortino ratio (uses only downside deviation).

        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized Sortino ratio

        Example:
            >>> sortino = aggregator.compute_sortino_ratio(returns)
        """
        if len(returns) < 2:
            return 0.0

        rf_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_period

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(
            periods_per_year
        )

        return float(sortino)

    def compute_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Compute maximum drawdown percentage.

        Args:
            portfolio_values: Series of portfolio values

        Returns:
            Maximum drawdown as percentage (negative value)

        Example:
            >>> values = pd.Series([100000, 105000, 102000, 98000, 103000])
            >>> max_dd = aggregator.compute_max_drawdown(values)
            >>> print(f"Max drawdown: {max_dd:.2f}%")
        """
        if len(portfolio_values) < 2:
            return 0.0

        # Calculate running maximum
        running_max = portfolio_values.cummax()

        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max

        # Return maximum drawdown as percentage
        max_dd = drawdown.min() * 100

        return float(max_dd)

    def compute_win_rate(self, trades_df: pd.DataFrame) -> float:
        """
        Compute win rate from trades.

        Args:
            trades_df: DataFrame with trade metrics (must have filled_price column)

        Returns:
            Win rate as decimal (0.0 to 1.0)

        Example:
            >>> trades = store.query_recent_trades(n=100)
            >>> win_rate = aggregator.compute_win_rate(trades)
            >>> print(f"Win rate: {win_rate:.1%}")
        """
        if trades_df.empty:
            return 0.0

        # Calculate P&L for each trade
        # For buys: later sells should have higher price
        # For sells: P&L already realized
        # Simplified: assume all filled trades contribute to P&L

        # Group by symbol and calculate P&L
        # This is simplified - in reality need to match buys with sells
        buys = trades_df[trades_df["side"] == "buy"]
        sells = trades_df[trades_df["side"] == "sell"]

        if len(sells) == 0:
            return 0.0

        # Count profitable sells (simplified)
        # In reality, need to match with corresponding buys
        winning_trades = len(sells[sells["slippage_pct"] < 0])  # Negative slippage = better than expected
        total_trades = len(sells)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return float(win_rate)

    def compute_daily_summary(self, target_date: date) -> dict:
        """
        Compute daily summary metrics.

        Args:
            target_date: Date to compute summary for

        Returns:
            Dictionary with daily summary metrics

        Example:
            >>> from datetime import date
            >>> summary = aggregator.compute_daily_summary(date.today())
            >>> print(summary)
        """
        # Get portfolio snapshots for the day
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())

        portfolio_df = self.store.query_portfolio_history(start=start, end=end)

        # Get trades for the day
        trades_df = self.store.query_recent_trades(n=1000)  # Get all recent
        if not trades_df.empty:
            trades_df["date"] = pd.to_datetime(trades_df["timestamp"]).dt.date
            day_trades = trades_df[trades_df["date"] == target_date]
        else:
            day_trades = pd.DataFrame()

        # Get system metrics for the day
        system_df = self.store.query_system_metrics(days=1)
        if not system_df.empty:
            system_df["date"] = pd.to_datetime(system_df["timestamp"]).dt.date
            day_system = system_df[system_df["date"] == target_date]
        else:
            day_system = pd.DataFrame()

        # Compute metrics
        total_trades = len(day_trades)
        win_rate = self.compute_win_rate(day_trades) if not day_trades.empty else 0.0

        # Average return
        if not portfolio_df.empty and "daily_return_pct" in portfolio_df.columns:
            avg_return = portfolio_df["daily_return_pct"].mean()
        else:
            avg_return = 0.0

        # Sharpe and max drawdown (if available)
        if not portfolio_df.empty:
            sharpe = portfolio_df["sharpe_ratio"].iloc[-1] if "sharpe_ratio" in portfolio_df.columns else None
            max_dd = portfolio_df["max_drawdown_pct"].iloc[-1] if "max_drawdown_pct" in portfolio_df.columns else None
            total_pnl = portfolio_df["realized_pnl"].iloc[-1] if "realized_pnl" in portfolio_df.columns else 0.0
        else:
            sharpe = None
            max_dd = None
            total_pnl = 0.0

        # Error count
        if not day_system.empty:
            error_count = len(day_system[day_system["metric_type"] == "error"])
        else:
            error_count = 0

        summary = {
            "date": target_date,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_return_pct": avg_return,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "total_pnl": total_pnl,
            "uptime_pct": 100.0,  # TODO: Implement uptime tracking
            "error_count": error_count,
        }

        logger.debug(f"Computed daily summary for {target_date}: {summary}")
        return summary

    def save_daily_summary(self, target_date: date) -> None:
        """
        Compute and save daily summary to database.

        Args:
            target_date: Date to compute summary for

        Example:
            >>> aggregator.save_daily_summary(date.today())
        """
        summary = self.compute_daily_summary(target_date)

        cursor = self.store.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO daily_summaries (
                date, total_trades, win_rate, avg_return_pct,
                sharpe_ratio, max_drawdown_pct, total_pnl,
                uptime_pct, error_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary["date"].isoformat(),
                summary["total_trades"],
                summary["win_rate"],
                summary["avg_return_pct"],
                summary["sharpe_ratio"],
                summary["max_drawdown_pct"],
                summary["total_pnl"],
                summary["uptime_pct"],
                summary["error_count"],
            ),
        )

        self.store.conn.commit()
        logger.info(f"Saved daily summary for {target_date}")
