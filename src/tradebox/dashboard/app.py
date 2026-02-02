"""
Main Streamlit dashboard application for TradeBox-RL.

This is the entry point for the web dashboard. It provides navigation
between different pages and handles auto-refresh.

Usage:
    streamlit run src/tradebox/dashboard/app.py

    Or via CLI:
    python scripts/dashboard.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tradebox.monitoring import MetricsQuery
from tradebox.dashboard.pages import overview, trading, model, system


def main():
    """Main dashboard application."""
    # Page configuration
    st.set_page_config(
        page_title="TradeBox-RL Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            padding-bottom: 1rem;
            border-bottom: 2px solid #1f77b4;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<h1 class="main-header">📊 TradeBox-RL</h1>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["📈 Overview", "💰 Trading", "🤖 Model", "⚙️ System"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Settings
        st.subheader("Settings")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)

        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (seconds)",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
            )

        # Database path
        db_path = st.text_input(
            "Metrics Database",
            value="data/metrics.db",
            help="Path to metrics database",
        )

        st.markdown("---")

        # Info
        st.caption("TradeBox-RL v1.0")
        st.caption("Real-time Trading Dashboard")

    # Initialize metrics query
    try:
        metrics = MetricsQuery(db_path)
    except Exception as e:
        st.error(f"Failed to connect to metrics database: {e}")
        st.stop()

    # Auto-refresh logic
    if auto_refresh:
        # Create placeholder for refresh counter
        refresh_placeholder = st.empty()

        # Countdown timer
        for remaining in range(refresh_interval, 0, -1):
            with refresh_placeholder.container():
                st.info(f"🔄 Auto-refreshing in {remaining} seconds...")
            time.sleep(1)

        refresh_placeholder.empty()

        # Trigger rerun
        st.rerun()

    # Render selected page
    if page == "📈 Overview":
        overview.render(metrics)
    elif page == "💰 Trading":
        trading.render(metrics)
    elif page == "🤖 Model":
        model.render(metrics)
    elif page == "⚙️ System":
        system.render(metrics)


if __name__ == "__main__":
    main()
