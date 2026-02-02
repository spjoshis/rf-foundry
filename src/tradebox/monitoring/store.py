"""Persistent storage for trading metrics using SQLite."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from tradebox.monitoring.events import (
    PortfolioMetricsEvent,
    TradeMetricsEvent,
    ModelMetricsEvent,
    SystemMetricsEvent,
)
from tradebox.monitoring.schemas import ALL_SCHEMAS


class MetricsStore:
    """
    Persistent storage for trading metrics using SQLite.

    Features:
    - Fast writes (<1ms per event)
    - Time-series optimized with indexes
    - Thread-safe with WAL mode
    - Automatic schema creation

    Example:
        >>> store = MetricsStore("data/metrics.db")
        >>> event = PortfolioMetricsEvent(...)
        >>> store.insert_portfolio_metrics(event)
        >>> df = store.query_portfolio_history(days=30)
    """

    def __init__(self, db_path: str = "data/metrics.db"):
        """
        Initialize metrics store.

        Args:
            db_path: Path to SQLite database file

        Example:
            >>> store = MetricsStore("data/metrics.db")
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,  # Allow multi-threading
        )

        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        # Create tables
        self._create_tables()

        logger.info(f"MetricsStore initialized: {db_path}")

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        for schema in ALL_SCHEMAS:
            cursor.executescript(schema)

        self.conn.commit()
        logger.debug("Database tables created/verified")

    def insert_portfolio_metrics(self, event: PortfolioMetricsEvent) -> None:
        """
        Insert portfolio metrics event.

        Args:
            event: PortfolioMetricsEvent instance

        Example:
            >>> event = PortfolioMetricsEvent(...)
            >>> store.insert_portfolio_metrics(event)
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO portfolio_metrics (
                timestamp, total_value, cash, positions_value,
                unrealized_pnl, realized_pnl, daily_return_pct,
                sharpe_ratio, max_drawdown_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.timestamp.isoformat(),
                event.total_value,
                event.cash,
                event.positions_value,
                event.unrealized_pnl,
                event.realized_pnl,
                event.daily_return_pct,
                event.sharpe_ratio,
                event.max_drawdown_pct,
            ),
        )

        self.conn.commit()
        logger.debug(f"Inserted portfolio metrics: value=₹{event.total_value:,.0f}")

    def insert_trade_metrics(self, event: TradeMetricsEvent) -> None:
        """
        Insert trade execution metrics event.

        Args:
            event: TradeMetricsEvent instance

        Example:
            >>> event = TradeMetricsEvent(...)
            >>> store.insert_trade_metrics(event)
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO trade_metrics (
                trade_id, timestamp, symbol, side, quantity,
                intended_price, filled_price, slippage_pct,
                commission, latency_ms, order_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.trade_id,
                event.timestamp.isoformat(),
                event.symbol,
                event.side,
                event.quantity,
                event.intended_price,
                event.filled_price,
                event.slippage_pct,
                event.commission,
                event.latency_ms,
                event.order_status,
            ),
        )

        self.conn.commit()
        logger.debug(
            f"Inserted trade metrics: {event.side} {event.quantity} {event.symbol}"
        )

    def insert_model_metrics(self, event: ModelMetricsEvent) -> None:
        """
        Insert model prediction metrics event.

        Args:
            event: ModelMetricsEvent instance

        Example:
            >>> event = ModelMetricsEvent(...)
            >>> store.insert_model_metrics(event)
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_metrics (
                timestamp, symbol, action, confidence, observation, reward
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.timestamp.isoformat(),
                event.symbol,
                event.action,
                event.confidence,
                event.observation,
                event.reward,
            ),
        )

        self.conn.commit()
        logger.debug(f"Inserted model metrics: action={event.action} {event.symbol}")

    def insert_system_metrics(self, event: SystemMetricsEvent) -> None:
        """
        Insert system health metrics event.

        Args:
            event: SystemMetricsEvent instance

        Example:
            >>> event = SystemMetricsEvent(...)
            >>> store.insert_system_metrics(event)
        """
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO system_metrics (
                timestamp, metric_name, metric_value, metric_type, component, message
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.timestamp.isoformat(),
                event.metric_name,
                event.metric_value,
                event.metric_type,
                event.component,
                event.message,
            ),
        )

        self.conn.commit()
        logger.debug(
            f"Inserted system metrics: {event.metric_name}={event.metric_value}"
        )

    def query_portfolio_history(
        self, days: int = 30, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Query portfolio history.

        Args:
            days: Number of days to fetch (default: 30)
            start: Start datetime (optional, overrides days)
            end: End datetime (optional)

        Returns:
            DataFrame with portfolio history

        Example:
            >>> df = store.query_portfolio_history(days=30)
            >>> print(df[['timestamp', 'total_value', 'daily_return_pct']])
        """
        if start is not None:
            query = """
                SELECT * FROM portfolio_metrics
                WHERE timestamp >= ?
            """
            params = [start.isoformat()]

            if end is not None:
                query += " AND timestamp <= ?"
                params.append(end.isoformat())

            query += " ORDER BY timestamp ASC"

        else:
            query = """
                SELECT * FROM portfolio_metrics
                WHERE timestamp >= datetime('now', ? || ' days')
                ORDER BY timestamp ASC
            """
            params = [f"-{days}"]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def query_recent_trades(self, n: int = 20, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Query recent trades.

        Args:
            n: Number of trades to fetch
            symbol: Filter by symbol (optional)

        Returns:
            DataFrame with recent trades

        Example:
            >>> df = store.query_recent_trades(n=10)
            >>> print(df[['timestamp', 'symbol', 'side', 'filled_price']])
        """
        if symbol is not None:
            query = """
                SELECT * FROM trade_metrics
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = [symbol, n]
        else:
            query = """
                SELECT * FROM trade_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params = [n]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def query_model_metrics(self, days: int = 7, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Query model prediction metrics.

        Args:
            days: Number of days to fetch
            symbol: Filter by symbol (optional)

        Returns:
            DataFrame with model metrics

        Example:
            >>> df = store.query_model_metrics(days=7)
            >>> action_dist = df['action'].value_counts()
        """
        if symbol is not None:
            query = """
                SELECT * FROM model_metrics
                WHERE timestamp >= datetime('now', ? || ' days')
                  AND symbol = ?
                ORDER BY timestamp DESC
            """
            params = [f"-{days}", symbol]
        else:
            query = """
                SELECT * FROM model_metrics
                WHERE timestamp >= datetime('now', ? || ' days')
                ORDER BY timestamp DESC
            """
            params = [f"-{days}"]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def query_system_metrics(
        self, days: int = 7, component: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query system health metrics.

        Args:
            days: Number of days to fetch
            component: Filter by component (optional)

        Returns:
            DataFrame with system metrics

        Example:
            >>> df = store.query_system_metrics(days=1, component="broker")
            >>> avg_latency = df[df['metric_name']=='api_latency']['metric_value'].mean()
        """
        if component is not None:
            query = """
                SELECT * FROM system_metrics
                WHERE timestamp >= datetime('now', ? || ' days')
                  AND component = ?
                ORDER BY timestamp DESC
            """
            params = [f"-{days}", component]
        else:
            query = """
                SELECT * FROM system_metrics
                WHERE timestamp >= datetime('now', ? || ' days')
                ORDER BY timestamp DESC
            """
            params = [f"-{days}"]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def get_latest_portfolio(self) -> Optional[dict]:
        """
        Get latest portfolio snapshot.

        Returns:
            Dictionary with latest portfolio metrics or None

        Example:
            >>> latest = store.get_latest_portfolio()
            >>> print(f"Portfolio value: ₹{latest['total_value']:,.0f}")
        """
        cursor = self.conn.cursor()

        result = cursor.execute(
            """
            SELECT * FROM portfolio_metrics
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).fetchone()

        if result is None:
            return None

        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))

    def query_trades_by_date(self, start_date, end_date) -> pd.DataFrame:
        """
        Query trades by date range.

        Args:
            start_date: Start date (date or datetime)
            end_date: End date (date or datetime)

        Returns:
            DataFrame with trades in date range

        Example:
            >>> df = store.query_trades_by_date(start_date, end_date)
        """
        query = """
            SELECT * FROM trade_metrics
            WHERE DATE(timestamp) >= DATE(?)
              AND DATE(timestamp) <= DATE(?)
            ORDER BY timestamp DESC
        """
        params = [str(start_date), str(end_date)]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def query_model_metrics_by_date(self, start_date, end_date) -> pd.DataFrame:
        """
        Query model metrics by date range.

        Args:
            start_date: Start date (date or datetime)
            end_date: End date (date or datetime)

        Returns:
            DataFrame with model metrics in date range

        Example:
            >>> df = store.query_model_metrics_by_date(start_date, end_date)
        """
        query = """
            SELECT * FROM model_metrics
            WHERE DATE(timestamp) >= DATE(?)
              AND DATE(timestamp) <= DATE(?)
            ORDER BY timestamp DESC
        """
        params = [str(start_date), str(end_date)]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def query_system_errors(self, days: int = 7) -> pd.DataFrame:
        """
        Query system errors for recent days.

        Args:
            days: Number of days to fetch

        Returns:
            DataFrame with system errors

        Example:
            >>> df = store.query_system_errors(days=1)
        """
        query = """
            SELECT * FROM system_errors
            WHERE timestamp >= datetime('now', ? || ' days')
            ORDER BY timestamp DESC
        """
        params = [f"-{days}"]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def query_system_errors_by_date(self, start_date, end_date) -> pd.DataFrame:
        """
        Query system errors by date range.

        Args:
            start_date: Start date (date or datetime)
            end_date: End date (date or datetime)

        Returns:
            DataFrame with system errors in date range

        Example:
            >>> df = store.query_system_errors_by_date(start_date, end_date)
        """
        query = """
            SELECT * FROM system_errors
            WHERE DATE(timestamp) >= DATE(?)
              AND DATE(timestamp) <= DATE(?)
            ORDER BY timestamp DESC
        """
        params = [str(start_date), str(end_date)]

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def insert_system_error(
        self,
        component: str,
        severity: str,
        message: str,
        stack_trace: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Insert system error event.

        Args:
            component: Component name (e.g., "workflow", "broker", "data")
            severity: Error severity (INFO/WARNING/ERROR/CRITICAL)
            message: Error message
            stack_trace: Stack trace (optional)
            timestamp: Error timestamp (optional, defaults to now)

        Example:
            >>> store.insert_system_error(
            ...     component="broker",
            ...     severity="ERROR",
            ...     message="Order placement failed",
            ...     stack_trace=traceback.format_exc()
            ... )
        """
        if timestamp is None:
            timestamp = datetime.now()

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO system_errors (
                timestamp, component, severity, message, stack_trace
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                timestamp.isoformat(),
                component,
                severity,
                message,
                stack_trace,
            ),
        )

        self.conn.commit()
        logger.debug(f"Inserted system error: [{severity}] {component} - {message}")

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.info("MetricsStore closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
