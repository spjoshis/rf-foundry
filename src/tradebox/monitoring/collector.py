"""Metrics collector for trading operations."""

from datetime import datetime
from typing import Optional

from loguru import logger

from tradebox.monitoring.events import (
    PortfolioMetricsEvent,
    TradeMetricsEvent,
    ModelMetricsEvent,
    SystemMetricsEvent,
)
from tradebox.monitoring.store import MetricsStore


class MetricsCollector:
    """
    Collects and stores metrics from trading operations.

    Uses Observer pattern to receive events from components and
    persists them to the metrics store.

    Example:
        >>> store = MetricsStore("data/metrics.db")
        >>> collector = MetricsCollector(store)
        >>>
        >>> # Record portfolio snapshot
        >>> event = PortfolioMetricsEvent(...)
        >>> collector.record_portfolio_snapshot(event)
        >>>
        >>> # Record trade
        >>> trade = TradeMetricsEvent(...)
        >>> collector.record_trade(trade)
    """

    def __init__(self, store: Optional[MetricsStore] = None):
        """
        Initialize metrics collector.

        Args:
            store: MetricsStore instance (creates default if None)

        Example:
            >>> store = MetricsStore("data/metrics.db")
            >>> collector = MetricsCollector(store)
        """
        if store is None:
            store = MetricsStore()

        self.store = store
        self._event_count = 0

        logger.info("MetricsCollector initialized")

    def record_portfolio_snapshot(self, event: PortfolioMetricsEvent) -> None:
        """
        Record portfolio metrics snapshot.

        Args:
            event: PortfolioMetricsEvent instance

        Example:
            >>> event = PortfolioMetricsEvent(
            ...     timestamp=datetime.now(),
            ...     total_value=105000.0,
            ...     cash=50000.0,
            ...     positions_value=55000.0,
            ...     unrealized_pnl=5000.0,
            ...     realized_pnl=0.0
            ... )
            >>> collector.record_portfolio_snapshot(event)
        """
        try:
            self.store.insert_portfolio_metrics(event)
            self._event_count += 1
        except Exception as e:
            logger.error(f"Failed to record portfolio metrics: {e}")

    def record_trade(self, event: TradeMetricsEvent) -> None:
        """
        Record trade execution metrics.

        Args:
            event: TradeMetricsEvent instance

        Example:
            >>> event = TradeMetricsEvent(
            ...     trade_id="ORD12345",
            ...     timestamp=datetime.now(),
            ...     symbol="RELIANCE.NS",
            ...     side="buy",
            ...     quantity=10,
            ...     intended_price=2500.0,
            ...     filled_price=2502.0,
            ...     slippage_pct=0.08,
            ...     commission=15.0,
            ...     latency_ms=250.0,
            ...     order_status="filled"
            ... )
            >>> collector.record_trade(event)
        """
        try:
            self.store.insert_trade_metrics(event)
            self._event_count += 1
        except Exception as e:
            logger.error(f"Failed to record trade metrics: {e}")

    def record_model_prediction(self, event: ModelMetricsEvent) -> None:
        """
        Record model prediction metrics.

        Args:
            event: ModelMetricsEvent instance

        Example:
            >>> event = ModelMetricsEvent(
            ...     timestamp=datetime.now(),
            ...     symbol="TCS.NS",
            ...     action=1,
            ...     confidence=0.85
            ... )
            >>> collector.record_model_prediction(event)
        """
        try:
            self.store.insert_model_metrics(event)
            self._event_count += 1
        except Exception as e:
            logger.error(f"Failed to record model metrics: {e}")

    def record_system_metric(self, event: SystemMetricsEvent) -> None:
        """
        Record system health metric.

        Args:
            event: SystemMetricsEvent instance

        Example:
            >>> event = SystemMetricsEvent(
            ...     timestamp=datetime.now(),
            ...     metric_name="api_latency",
            ...     metric_value=120.0,
            ...     metric_type="latency",
            ...     component="broker"
            ... )
            >>> collector.record_system_metric(event)
        """
        try:
            self.store.insert_system_metrics(event)
            self._event_count += 1
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")

    def get_event_count(self) -> int:
        """
        Get total number of events collected.

        Returns:
            Total event count

        Example:
            >>> count = collector.get_event_count()
            >>> print(f"Collected {count} events")
        """
        return self._event_count

    def reset_event_count(self) -> None:
        """Reset event counter."""
        self._event_count = 0
