"""
Monitoring and metrics collection module.

This module provides infrastructure for collecting, storing, and querying
trading metrics including portfolio performance, trade execution quality,
model predictions, and system health.

Key Components:
    - MetricsCollector: Collects metrics from trading operations
    - MetricsStore: Persists metrics to SQLite database
    - MetricsAggregator: Computes aggregated metrics and summaries
    - MetricsQuery: High-level query interface for dashboards
    - Event dataclasses: Structured metric events

Example:
    >>> from tradebox.monitoring import MetricsCollector, MetricsStore
    >>> from tradebox.monitoring import PortfolioMetricsEvent
    >>>
    >>> # Initialize
    >>> store = MetricsStore("data/metrics.db")
    >>> collector = MetricsCollector(store)
    >>>
    >>> # Record metrics
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

from tradebox.monitoring.collector import MetricsCollector
from tradebox.monitoring.events import (
    PortfolioMetricsEvent,
    TradeMetricsEvent,
    ModelMetricsEvent,
    SystemMetricsEvent,
)
from tradebox.monitoring.store import MetricsStore
from tradebox.monitoring.aggregator import MetricsAggregator
from tradebox.monitoring.query import MetricsQuery

__all__ = [
    "MetricsCollector",
    "MetricsStore",
    "MetricsAggregator",
    "MetricsQuery",
    "PortfolioMetricsEvent",
    "TradeMetricsEvent",
    "ModelMetricsEvent",
    "SystemMetricsEvent",
]
