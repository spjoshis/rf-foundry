"""Dashboard pages.

Each page module exports a render(metrics: MetricsQuery) function
that displays the page content using Streamlit.

Available pages:
- overview: Portfolio overview, positions, recent trades
- trading: Trade execution analytics, fill quality, slippage
- model: RL agent performance, action distribution, confidence
- system: System health, errors, API latency, database stats
"""

from tradebox.dashboard.pages import overview, trading, model, system

__all__ = ["overview", "trading", "model", "system"]
