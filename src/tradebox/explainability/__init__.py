"""
Explainability module for RL trading agents.

This module provides tools to understand and explain why the trading agent
makes specific decisions. It includes:

- Attention weight visualization (which price bars the agent focused on)
- Gradient-based attribution (feature contribution to decisions)
- Trade explanation generation (human-readable summaries)
- Interactive visualizations (charts, heatmaps, dashboards)

Key Components:
    - TradeExplainer: Main class for generating explanations
    - AttentionAnalyzer: Analyzes and visualizes attention weights
    - AttributionEngine: Computes feature attributions using Captum
    - TextGenerator: Generates human-readable trade summaries
    - Visualizers: Creates charts and interactive dashboards

Example:
    >>> from tradebox.explainability import TradeExplainer
    >>> from tradebox.agents.ppo_agent import PPOAgent
    >>>
    >>> agent = PPOAgent.load("models/best_model")
    >>> explainer = TradeExplainer(agent)
    >>>
    >>> # Explain a trade
    >>> explanation = explainer.explain(observation, action='BUY')
    >>> print(explanation['summary'])
    "BUY: Strong bullish breakout in bars 55-59. RSI oversold (35)."
    >>>
    >>> # Visualize attention
    >>> explainer.visualize_attention(observation, save_path="attention.png")
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies and heavy imports at package level
if TYPE_CHECKING:
    from .trade_explainer import TradeExplainer
    from .attention_viz import AttentionAnalyzer
    from .text_generator import TradeExplainTextGenerator

__all__ = [
    "TradeExplainer",
    "AttentionAnalyzer",
    "TradeExplainTextGenerator",
]
