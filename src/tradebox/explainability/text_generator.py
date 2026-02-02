"""
Text summary generator for trade explanations.

This module generates human-readable summaries of trading decisions
based on attention patterns, indicator signals, and portfolio state.
"""

from typing import Dict, List


class TradeExplainTextGenerator:
    """
    Generates human-readable explanations for trading decisions.

    This class provides template-based text generation for creating
    clear, concise explanations that traders can quickly understand.

    Example:
        >>> generator = TradeExplainTextGenerator()
        >>> summary = generator.generate(explanation_dict)
        >>> print(summary)
        "BUY: Strong bullish breakout in bars 55-59. RSI oversold (35), MACD bullish crossover."
    """

    def __init__(self):
        """Initialize text generator with templates."""
        self.action_templates = {
            "BUY": [
                "{action} executed with {confidence}% confidence. {primary_reason} {indicators} {portfolio}",
                "{action} signal triggered ({confidence}% confident). {primary_reason} Confirmations: {indicators} {portfolio}",
            ],
            "SELL": [
                "{action} executed with {confidence}% confidence. {primary_reason} {indicators} {profit_context}",
                "{action} triggered ({confidence}% confident). {primary_reason} {indicators} {profit_context}",
            ],
            "HOLD": [
                "{action}: {primary_reason} {indicators} {portfolio}",
                "Position maintained. {primary_reason} {indicators}",
            ],
        }

        self.pattern_descriptions = {
            "momentum_driven": "Strong momentum pattern in recent bars {bars}",
            "breakout": "Breakout pattern detected at bar {bar}",
            "mean_reversion": "Mean reversion signal from bars {bars}",
            "breakout_detection": "Consolidation breakout pattern",
            "mixed_signals": "Mixed technical signals",
        }

    def generate(self, explanation: Dict) -> str:
        """
        Generate human-readable summary from explanation dict.

        Args:
            explanation: Explanation dict from TradeExplainer containing:
                - action: str
                - confidence: float
                - price_pattern_analysis: Dict
                - indicator_analysis: Dict
                - portfolio_state: Dict

        Returns:
            Human-readable summary string
        """
        action = explanation["action"]
        confidence = int(explanation["confidence"] * 100)

        # Extract components
        price_analysis = explanation.get("price_pattern_analysis", {})
        indicator_analysis = explanation.get("indicator_analysis", {})
        portfolio = explanation.get("portfolio_state", {})

        # Build reason string
        primary_reason = self._build_primary_reason(price_analysis)
        indicators = self._build_indicator_string(indicator_analysis)
        portfolio_context = self._build_portfolio_context(action, portfolio)

        # Select and format template
        template = self.action_templates.get(action, ["{action}: {primary_reason}"])[0]

        # Format template
        summary = template.format(
            action=action,
            confidence=confidence,
            primary_reason=primary_reason,
            indicators=indicators,
            portfolio=portfolio_context,
            profit_context=self._build_profit_context(portfolio),
        )

        return summary

    def _build_primary_reason(self, price_analysis: Dict) -> str:
        """Build primary reason string from price pattern analysis."""
        pattern = price_analysis.get("pattern_detected", "unknown")
        primary_bars = price_analysis.get("attention_focus", {}).get("primary_bars", [])

        if not primary_bars:
            return "Price pattern analysis."

        # Get bar range string
        if len(primary_bars) >= 3:
            bar_range = f"{min(primary_bars)}-{max(primary_bars)}"
        else:
            bar_range = ", ".join(map(str, primary_bars))

        # Format pattern description
        pattern_clean = pattern.replace("_", " ").title()
        return f"{pattern_clean} pattern detected in bars {bar_range}."

    def _build_indicator_string(self, indicator_analysis: Dict) -> str:
        """Build indicator signals string."""
        top_contributors = indicator_analysis.get("top_contributors", [])

        if not top_contributors:
            return ""

        # Format top 3 indicators
        indicator_parts = []
        for ind in top_contributors[:3]:
            name = ind["name"]
            value = ind["value"]
            signal = ind["signal"].replace("_", " ")

            # Shorten indicator names for readability
            name_short = name.replace("Close_", "").replace("_", "")

            indicator_parts.append(f"{name_short} {signal} ({value:.1f})")

        return "Indicators: " + ", ".join(indicator_parts) + "."

    def _build_portfolio_context(self, action: str, portfolio: Dict) -> str:
        """Build portfolio context string."""
        if action == "BUY":
            cash = portfolio.get("cash_pct", 1.0)
            return f"Cash: {cash*100:.0f}%."
        elif action == "SELL":
            position = portfolio.get("position_pct", 0.0)
            return f"Position: {position*100:.0f}%."
        else:
            return ""

    def _build_profit_context(self, portfolio: Dict) -> str:
        """Build profit context for SELL actions."""
        unrealized_pnl = portfolio.get("unrealized_pnl", 0.0)
        if abs(unrealized_pnl) < 0.01:
            return ""
        return f"Realized P&L: {unrealized_pnl:+.1f}%."

    def generate_detailed(self, explanation: Dict) -> str:
        """
        Generate detailed multi-line explanation.

        Args:
            explanation: Explanation dict

        Returns:
            Detailed explanation with multiple sections
        """
        lines = []

        # Header
        action = explanation["action"]
        confidence = explanation["confidence"] * 100
        lines.append(f"{'='*60}")
        lines.append(f"{action} DECISION - Confidence: {confidence:.1f}%")
        lines.append(f"{'='*60}")
        lines.append("")

        # Price Pattern Section
        price_analysis = explanation.get("price_pattern_analysis", {})
        if price_analysis:
            lines.append("PRICE PATTERN ANALYSIS:")
            pattern = price_analysis.get("pattern_detected", "unknown").replace("_", " ").title()
            lines.append(f"  Pattern: {pattern}")

            focus = price_analysis.get("attention_focus", {})
            primary_bars = focus.get("primary_bars", [])
            if primary_bars:
                bars_str = ", ".join(map(str, primary_bars[:5]))
                lines.append(f"  Focus Bars: {bars_str}")

            dist = price_analysis.get("focus_distribution", {})
            if dist:
                lines.append(f"  Recent Focus: {dist.get('recent', 0)*100:.1f}%")
                lines.append(f"  Middle Focus: {dist.get('middle', 0)*100:.1f}%")
                lines.append(f"  Distant Focus: {dist.get('distant', 0)*100:.1f}%")
            lines.append("")

        # Indicator Section
        indicator_analysis = explanation.get("indicator_analysis", {})
        top_contributors = indicator_analysis.get("top_contributors", [])
        if top_contributors:
            lines.append("TOP INDICATORS:")
            for i, ind in enumerate(top_contributors[:5], 1):
                name = ind["name"]
                value = ind["value"]
                signal = ind["signal"].replace("_", " ").title()
                lines.append(f"  {i}. {name}: {value:.2f} ({signal})")
            lines.append("")

        # Portfolio Section
        portfolio = explanation.get("portfolio_state", {})
        if portfolio:
            lines.append("PORTFOLIO STATE:")
            lines.append(f"  Position: {portfolio.get('position_pct', 0)*100:.1f}%")
            lines.append(f"  Cash: {portfolio.get('cash_pct', 1.0)*100:.1f}%")
            unrealized_pnl = portfolio.get("unrealized_pnl", 0.0)
            if abs(unrealized_pnl) > 0.01:
                lines.append(f"  Unrealized P&L: {unrealized_pnl:+.1f}%")
            lines.append("")

        # Summary
        lines.append("SUMMARY:")
        lines.append(f"  {explanation.get('summary', 'No summary available.')}")
        lines.append(f"{'='*60}")

        return "\n".join(lines)
