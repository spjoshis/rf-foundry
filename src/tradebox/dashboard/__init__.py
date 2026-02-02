"""
Real-time dashboard for TradeBox-RL.

This module provides a Streamlit-based web dashboard for monitoring
trading operations in real-time.

Key Components:
    - Main app: Multi-page dashboard with navigation
    - Overview page: Portfolio summary and recent activity
    - Trading page: Trade history and execution quality
    - Model page: Model performance and predictions
    - System page: System health and alerts

Usage:
    Launch dashboard:
    $ python scripts/dashboard.py

    Or directly:
    $ streamlit run src/tradebox/dashboard/app.py
"""

from tradebox.dashboard.charts import ChartBuilder
from tradebox.dashboard.utils import format_currency, format_percentage, calculate_color

__all__ = [
    "ChartBuilder",
    "format_currency",
    "format_percentage",
    "calculate_color",
]
