"""
Attention weight visualization module.

This module provides tools to visualize attention weights overlaid on
candlestick charts, helping understand which price bars the agent focused on
when making trading decisions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from pathlib import Path


class AttentionAnalyzer:
    """
    Analyzer and visualizer for attention weights in trading agent.

    This class provides methods to:
    - Extract and process attention weights
    - Visualize attention over candlestick charts
    - Interpret attention patterns
    - Generate attention heatmaps

    Example:
        >>> analyzer = AttentionAnalyzer()
        >>> fig = analyzer.plot_attention_on_candlestick(
        ...     price_data=ohlcv_data,
        ...     attention_weights=attn_weights,
        ...     action="BUY",
        ...     save_path="attention_viz.png"
        ... )
    """

    def __init__(self):
        """Initialize attention analyzer."""
        self.pattern_descriptions = {
            "momentum_driven": "Momentum-driven (focused on recent bars)",
            "mean_reversion": "Mean Reversion (comparing to distant levels)",
            "breakout_detection": "Breakout Detection (mid-window reference)",
            "mixed_signals": "Mixed Signals",
        }

    def plot_attention_on_candlestick(
        self,
        price_data: np.ndarray,
        attention_weights: np.ndarray,
        action: str = "HOLD",
        confidence: float = 0.0,
        top_indicators: Optional[List[Dict]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> Figure:
        """
        Plot attention weights overlaid on candlestick chart.

        Args:
            price_data: (lookback, 5) array with OHLCV data
            attention_weights: (num_heads, lookback, lookback) attention weights
                              or (lookback,) averaged attention scores
            action: Action taken ("BUY", "SELL", "HOLD")
            confidence: Action confidence (0-1)
            top_indicators: List of top indicator dicts
            save_path: Optional path to save figure
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object
        """
        # Process attention weights
        if attention_weights.ndim == 3:
            # Average across heads and get last bar's attention
            avg_attention = attention_weights.mean(axis=0)
            recent_bar_attention = avg_attention[-1, :]
        else:
            recent_bar_attention = attention_weights

        lookback = len(price_data)

        # Create figure with subplots
        fig, (ax_candle, ax_attn) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
        )

        # Top panel: Candlestick chart with attention overlay
        self._plot_candlesticks(ax_candle, price_data, recent_bar_attention, action)

        # Bottom panel: Attention scores bar chart
        self._plot_attention_bars(ax_attn, recent_bar_attention)

        # Add title with action and confidence
        pattern = self._detect_pattern(recent_bar_attention, lookback)
        pattern_desc = self.pattern_descriptions.get(pattern, pattern)

        title = f"{action} Decision (Confidence: {confidence*100:.1f}%)\n"
        title += f"Attention Pattern: {pattern_desc}"

        if top_indicators:
            indicator_str = ", ".join([f"{ind['name']}={ind['value']:.1f}" for ind in top_indicators[:3]])
            title += f"\nTop Indicators: {indicator_str}"

        fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved attention visualization to {save_path}")

        return fig

    def _plot_candlesticks(
        self,
        ax: plt.Axes,
        price_data: np.ndarray,
        attention_scores: np.ndarray,
        action: str,
    ):
        """Plot candlestick chart with attention overlay."""
        lookback = len(price_data)
        indices = np.arange(lookback)

        # Extract OHLC
        opens = price_data[:, 0]
        highs = price_data[:, 1]
        lows = price_data[:, 2]
        closes = price_data[:, 3]

        # Normalize attention scores for visualization
        attention_norm = attention_scores / (attention_scores.max() + 1e-8)

        # Plot candlesticks with attention-based coloring
        for i in indices:
            color = "green" if closes[i] >= opens[i] else "red"
            alpha = 0.3 + 0.7 * attention_norm[i]  # Higher attention = more opaque

            # Candlestick body
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            rect = mpatches.Rectangle(
                (i - 0.3, body_bottom),
                0.6,
                body_height,
                facecolor=color,
                edgecolor="black",
                alpha=alpha,
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # Wicks
            ax.plot(
                [i, i], [lows[i], highs[i]],
                color="black",
                linewidth=0.5,
                alpha=alpha,
            )

            # Highlight top-5 attended bars with yellow background
            if attention_norm[i] > np.percentile(attention_norm, 80):
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color="yellow", zorder=-1)

        # Set limits and labels
        ax.set_xlim(-1, lookback)
        y_min = lows.min() * 0.995
        y_max = highs.max() * 1.005
        ax.set_ylim(y_min, y_max)

        ax.set_ylabel("Price", fontsize=12)
        ax.set_title(f"Candlestick Chart (Last {lookback} Bars)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add action annotation on last bar
        action_colors = {"BUY": "green", "SELL": "red", "HOLD": "gray"}
        ax.annotate(
            action,
            xy=(lookback - 1, closes[-1]),
            xytext=(lookback - 1, closes[-1] * 1.02),
            fontsize=12,
            fontweight="bold",
            color=action_colors.get(action, "black"),
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=action_colors.get(action, "gray"), alpha=0.3),
        )

        # Legend
        green_patch = mpatches.Patch(color="green", alpha=0.7, label="Bullish candle")
        red_patch = mpatches.Patch(color="red", alpha=0.7, label="Bearish candle")
        yellow_patch = mpatches.Patch(color="yellow", alpha=0.2, label="High attention (top 20%)")
        ax.legend(handles=[green_patch, red_patch, yellow_patch], loc="upper left", fontsize=10)

    def _plot_attention_bars(self, ax: plt.Axes, attention_scores: np.ndarray):
        """Plot attention scores as bar chart."""
        lookback = len(attention_scores)
        indices = np.arange(lookback)

        # Color bars by magnitude
        colors = plt.cm.YlOrRd(attention_scores / (attention_scores.max() + 1e-8))

        bars = ax.bar(indices, attention_scores, color=colors, edgecolor="black", linewidth=0.5)

        # Highlight top-5 bars
        top5_indices = attention_scores.argsort()[-5:]
        for idx in top5_indices:
            bars[idx].set_edgecolor("blue")
            bars[idx].set_linewidth(2)

        ax.set_xlabel("Bar Index (0 = oldest, {} = most recent)".format(lookback - 1), fontsize=11)
        ax.set_ylabel("Attention Score", fontsize=11)
        ax.set_title("Agent Attention Focus", fontsize=12)
        ax.set_xlim(-1, lookback)
        ax.grid(True, alpha=0.3, axis="y")

    def _detect_pattern(self, attention_scores: np.ndarray, lookback: int) -> str:
        """Detect attention pattern type."""
        recent_focus = attention_scores[-10:].mean()
        distant_focus = attention_scores[:30].mean() if lookback >= 30 else 0.0
        middle_focus = attention_scores[30:50].mean() if lookback >= 50 else 0.0

        if recent_focus > 0.4:
            return "momentum_driven"
        elif distant_focus > recent_focus * 1.5:
            return "mean_reversion"
        elif middle_focus > 0.3:
            return "breakout_detection"
        else:
            return "mixed_signals"

    def create_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Figure:
        """
        Create attention heatmap showing all heads and positions.

        Args:
            attention_weights: (num_heads, lookback, lookback) attention matrix
            save_path: Optional path to save figure
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        num_heads = attention_weights.shape[0]

        # Create subplots for each head
        fig, axes = plt.subplots(1, num_heads, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for head_idx in range(num_heads):
            ax = axes[head_idx]

            # Plot heatmap
            sns.heatmap(
                attention_weights[head_idx],
                cmap="YlOrRd",
                ax=ax,
                cbar=True,
                square=True,
                linewidths=0.5,
                linecolor="gray",
            )

            ax.set_title(f"Head {head_idx + 1}", fontsize=12)
            ax.set_xlabel("Key Position", fontsize=10)
            ax.set_ylabel("Query Position", fontsize=10)

        fig.suptitle("Attention Weights per Head", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def analyze_attention_distribution(
        self, attention_scores: np.ndarray
    ) -> Dict:
        """
        Analyze attention distribution and return statistics.

        Args:
            attention_scores: (lookback,) attention scores

        Returns:
            Dict with distribution statistics
        """
        lookback = len(attention_scores)

        # Find critical bars (top 20%)
        threshold = np.percentile(attention_scores, 80)
        critical_bars = np.where(attention_scores > threshold)[0]

        # Compute focus distribution
        recent_focus = attention_scores[-10:].mean()
        middle_focus = attention_scores[30:50].mean() if lookback >= 50 else 0.0
        distant_focus = attention_scores[:30].mean() if lookback >= 30 else 0.0

        # Entropy (measure of focus spread)
        attention_probs = attention_scores / (attention_scores.sum() + 1e-8)
        entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8))

        return {
            "critical_bars": critical_bars.tolist(),
            "num_critical_bars": len(critical_bars),
            "focus_distribution": {
                "recent": float(recent_focus),
                "middle": float(middle_focus),
                "distant": float(distant_focus),
            },
            "entropy": float(entropy),
            "max_attention": float(attention_scores.max()),
            "mean_attention": float(attention_scores.mean()),
            "pattern_type": self._detect_pattern(attention_scores, lookback),
        }
