"""High-level query interface for metrics."""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

import pandas as pd
from loguru import logger

from tradebox.monitoring.store import MetricsStore
from tradebox.monitoring.aggregator import MetricsAggregator


class MetricsQuery:
    """
    High-level query interface for dashboards.

    Provides convenient methods for querying and aggregating metrics
    for visualization in dashboards.

    Example:
        >>> query = MetricsQuery("data/metrics.db")
        >>>
        >>> # Get latest portfolio
        >>> portfolio = query.get_latest_portfolio()
        >>> print(f"Value: ₹{portfolio['total_value']:,.0f}")
        >>>
        >>> # Get portfolio history
        >>> history = query.get_portfolio_history(days=30)
        >>>
        >>> # Get recent trades
        >>> trades = query.get_recent_trades(n=20)
    """

    def __init__(self, db_path: str = "data/metrics.db"):
        """
        Initialize metrics query interface.

        Args:
            db_path: Path to metrics database

        Example:
            >>> query = MetricsQuery("data/metrics.db")
        """
        self.store = MetricsStore(db_path)
        self.aggregator = MetricsAggregator(self.store)
        logger.info(f"MetricsQuery initialized: {db_path}")

    def get_latest_portfolio(self) -> Optional[Dict]:
        """
        Get latest portfolio snapshot.

        Returns:
            Dictionary with latest portfolio metrics or None

        Example:
            >>> portfolio = query.get_latest_portfolio()
            >>> if portfolio:
            ...     print(f"Portfolio value: ₹{portfolio['total_value']:,.0f}")
        """
        return self.store.get_latest_portfolio()

    def get_portfolio_history(
        self, days: int = 30, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get portfolio history.

        Args:
            days: Number of days to fetch (default: 30)
            start: Start datetime (optional, overrides days)
            end: End datetime (optional)

        Returns:
            DataFrame with portfolio history

        Example:
            >>> df = query.get_portfolio_history(days=30)
            >>> print(df[['timestamp', 'total_value', 'daily_return_pct']])
        """
        return self.store.query_portfolio_history(days=days, start=start, end=end)

    def get_recent_trades(self, n: int = 20, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get recent trades.

        Args:
            n: Number of trades to fetch
            symbol: Filter by symbol (optional)

        Returns:
            DataFrame with recent trades

        Example:
            >>> trades = query.get_recent_trades(n=10)
            >>> print(trades[['timestamp', 'symbol', 'side', 'quantity']])
        """
        return self.store.query_recent_trades(n=n, symbol=symbol)

    def get_active_positions(self) -> pd.DataFrame:
        """
        Get active positions from latest portfolio snapshot.

        This is a placeholder - in reality, positions should be tracked
        separately or derived from trades.

        Returns:
            DataFrame with active positions

        Example:
            >>> positions = query.get_active_positions()
            >>> print(positions[['symbol', 'quantity', 'avg_price']])
        """
        # TODO: Implement proper position tracking
        # For now, return empty DataFrame
        return pd.DataFrame(columns=["symbol", "quantity", "avg_price", "current_price", "unrealized_pnl"])

    def get_model_performance(self, days: int = 7) -> Dict:
        """
        Get model performance metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with model performance metrics

        Example:
            >>> perf = query.get_model_performance(days=7)
            >>> print(f"Action distribution: {perf['action_distribution']}")
        """
        df = self.store.query_model_metrics(days=days)

        if df.empty:
            return {
                "action_distribution": {},
                "avg_confidence": 0.0,
                "total_predictions": 0,
            }

        # Action distribution
        action_dist = df["action"].value_counts().to_dict()

        # Average confidence
        avg_confidence = df["confidence"].mean() if "confidence" in df.columns else 0.0

        return {
            "action_distribution": action_dist,
            "avg_confidence": float(avg_confidence),
            "total_predictions": len(df),
        }

    def get_system_status(self) -> Dict:
        """
        Get system health status.

        Returns:
            Dictionary with system health metrics

        Example:
            >>> status = query.get_system_status()
            >>> print(f"Uptime: {status['uptime_pct']:.1f}%")
            >>> print(f"Error count (24h): {status['error_count_24h']}")
        """
        # Get system metrics for last 24 hours
        df = self.store.query_system_metrics(days=1)

        if df.empty:
            return {
                "uptime_pct": 100.0,
                "avg_latency_ms": 0.0,
                "error_count_24h": 0,
                "circuit_breaker_active": False,
            }

        # Calculate average latency
        latency_df = df[df["metric_type"] == "latency"]
        avg_latency = latency_df["metric_value"].mean() if not latency_df.empty else 0.0

        # Count errors
        error_count = len(df[df["metric_type"] == "error"])

        # TODO: Get circuit breaker status from orchestrator state
        circuit_breaker_active = False

        return {
            "uptime_pct": 100.0,  # TODO: Implement uptime tracking
            "avg_latency_ms": float(avg_latency),
            "error_count_24h": error_count,
            "circuit_breaker_active": circuit_breaker_active,
        }

    def get_error_timeline(self, days: int = 7) -> pd.DataFrame:
        """
        Get error timeline.

        Args:
            days: Number of days to fetch

        Returns:
            DataFrame with daily error counts

        Example:
            >>> errors = query.get_error_timeline(days=7)
            >>> print(errors[['date', 'count']])
        """
        df = self.store.query_system_metrics(days=days)

        if df.empty:
            return pd.DataFrame(columns=["date", "count"])

        # Filter errors
        errors = df[df["metric_type"] == "error"].copy()

        if errors.empty:
            return pd.DataFrame(columns=["date", "count"])

        # Group by date
        errors["date"] = pd.to_datetime(errors["timestamp"]).dt.date
        error_counts = errors.groupby("date").size().reset_index(name="count")

        return error_counts

    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """
        Get recent alerts.

        Args:
            n: Number of alerts to fetch

        Returns:
            List of alert dictionaries

        Example:
            >>> alerts = query.get_recent_alerts(n=5)
            >>> for alert in alerts:
            ...     print(f"{alert['timestamp']}: {alert['message']}")
        """
        # Get recent error/warning system metrics
        df = self.store.query_system_metrics(days=1)

        if df.empty:
            return []

        # Filter warnings and errors
        alerts_df = df[df["metric_type"].isin(["error", "warning"])].copy()

        if alerts_df.empty:
            return []

        # Sort by timestamp descending and take top n
        alerts_df = alerts_df.sort_values("timestamp", ascending=False).head(n)

        # Convert to list of dicts
        alerts = []
        for _, row in alerts_df.iterrows():
            alerts.append({
                "timestamp": row["timestamp"],
                "severity": row["metric_type"],
                "message": row["message"] or row["metric_name"],
                "component": row["component"],
            })

        return alerts

    def get_trades(self, start_date: date, end_date: date, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Get trades within date range.

        Args:
            start_date: Start date
            end_date: End date
            symbol: Filter by symbol (optional)

        Returns:
            DataFrame with trades

        Example:
            >>> from datetime import date, timedelta
            >>> end = date.today()
            >>> start = end - timedelta(days=7)
            >>> trades = query.get_trades(start, end)
        """
        # Convert dates to datetime
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        # Query trades
        if symbol:
            df = self.store.query_recent_trades(n=10000, symbol=symbol)
        else:
            df = self.store.query_recent_trades(n=10000)

        if df.empty:
            return df

        # Filter by date range
        df = df[
            (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
        ].copy()

        return df

    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Get comprehensive performance summary.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with performance metrics

        Example:
            >>> summary = query.get_performance_summary(days=30)
            >>> print(f"Sharpe: {summary['sharpe_ratio']:.2f}")
            >>> print(f"Max DD: {summary['max_drawdown_pct']:.2f}%")
        """
        # Get portfolio history
        portfolio_df = self.store.query_portfolio_history(days=days)

        if portfolio_df.empty:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "total_return_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }

        # Calculate metrics
        if "daily_return_pct" in portfolio_df.columns:
            returns = portfolio_df["daily_return_pct"] / 100.0  # Convert to decimal
            sharpe = self.aggregator.compute_sharpe_ratio(returns)
            sortino = self.aggregator.compute_sortino_ratio(returns)
        else:
            sharpe = 0.0
            sortino = 0.0

        # Max drawdown
        max_dd = self.aggregator.compute_max_drawdown(portfolio_df["total_value"])

        # Total return
        if len(portfolio_df) >= 2:
            start_value = portfolio_df["total_value"].iloc[0]
            end_value = portfolio_df["total_value"].iloc[-1]
            total_return_pct = ((end_value - start_value) / start_value) * 100
        else:
            total_return_pct = 0.0

        # Win rate and trade count
        trades_df = self.store.query_recent_trades(n=10000)
        win_rate = self.aggregator.compute_win_rate(trades_df)
        total_trades = len(trades_df)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd,
            "total_return_pct": float(total_return_pct),
            "win_rate": win_rate,
            "total_trades": total_trades,
        }

    def close(self) -> None:
        """Close database connection."""
        self.store.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
