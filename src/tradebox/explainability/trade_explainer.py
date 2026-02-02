"""
Main trade explainer class for generating comprehensive explanations.

This module provides the TradeExplainer class which brings together all
explainability methods to generate comprehensive explanations for why the
agent executed specific trades.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch as th
from pathlib import Path

from tradebox.agents.ppo_agent import PPOAgent


class TradeExplainer:
    """
    Main class for explaining trading agent decisions.

    This class provides a unified interface for generating comprehensive
    explanations including:
    - Attention weight analysis (which bars the agent focused on)
    - Feature attribution (which indicators contributed)
    - Text summaries (human-readable explanations)
    - Visualizations (charts, heatmaps)

    Example:
        >>> agent = PPOAgent.load("models/best_model")
        >>> explainer = TradeExplainer(agent)
        >>>
        >>> # Explain a trade
        >>> explanation = explainer.explain(observation, action='BUY')
        >>> print(explanation['summary'])
        >>> print(explanation['attention_focus'])
        >>> print(explanation['top_indicators'])
        >>>
        >>> # Save visualization
        >>> explainer.visualize_trade(observation, "trade_analysis.png")
    """

    def __init__(
        self,
        agent: PPOAgent,
        feature_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        """
        Initialize trade explainer.

        Args:
            agent: Trained PPO agent to explain
            feature_names: Optional list of feature names for indicators
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.agent = agent
        self.feature_names = feature_names
        self.device = device

        # Get the policy network
        self.policy = agent.model.policy

        # Enable intermediate capture in feature extractor if available
        if hasattr(self.policy.features_extractor, "capture_intermediates"):
            self.policy.features_extractor.capture_intermediates = True

        # Action mapping
        self.action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def explain(
        self,
        observation: Dict[str, np.ndarray],
        action: Optional[Union[int, str]] = None,
        method: str = "attention",
        include_viz: bool = False,
    ) -> Dict:
        """
        Generate comprehensive explanation for a trading decision.

        Args:
            observation: Observation dict with keys:
                - "price": (lookback, 5) OHLCV data
                - "indicators": (n_indicators,) technical indicators
                - "portfolio": (state_dim,) portfolio state
            action: Action taken (0/1/2 or "HOLD"/"BUY"/"SELL")
                    If None, uses agent's predicted action
            method: Explanation method:
                - "attention": Attention weights only (fastest)
                - "saliency": Add saliency maps (fast)
                - "integrated_gradients": Add integrated gradients (slower)
            include_viz: Whether to generate visualization arrays

        Returns:
            Dictionary with explanation components:
            {
                "action": str,  # "BUY", "SELL", "HOLD"
                "action_idx": int,  # 0, 1, 2
                "confidence": float,  # Action probability
                "price_pattern_analysis": {...},
                "indicator_analysis": {...},
                "portfolio_state": {...},
                "summary": str,  # Human-readable summary
                "visualization": Optional[np.ndarray]  # If include_viz=True
            }
        """
        # Prepare observation tensors
        obs_tensor = self._prepare_observation(observation)

        # Get agent's action and probabilities
        with th.no_grad():
            action_logits = self.policy.get_distribution(obs_tensor).distribution.logits
            action_probs = th.softmax(action_logits, dim=-1)
            predicted_action = action_probs.argmax(dim=-1).item()

        # Use predicted action if not specified
        if action is None:
            action_idx = predicted_action
        elif isinstance(action, str):
            action_idx = {v: k for k, v in self.action_names.items()}[action.upper()]
        else:
            # Convert numpy array or tensor to int
            action_idx = int(action) if hasattr(action, '__iter__') else action

        action_name = self.action_names[action_idx]
        confidence = action_probs[0, action_idx].item()

        # Build explanation dict
        explanation = {
            "action": action_name,
            "action_idx": action_idx,
            "confidence": confidence,
            "predicted_action": self.action_names[predicted_action],
        }

        # 1. Price pattern analysis (attention weights)
        price_analysis = self._analyze_price_patterns(observation, obs_tensor)
        explanation["price_pattern_analysis"] = price_analysis

        # 2. Indicator analysis
        indicator_analysis = self._analyze_indicators(observation, method=method)
        explanation["indicator_analysis"] = indicator_analysis

        # 3. Portfolio state
        portfolio_analysis = self._analyze_portfolio_state(observation)
        explanation["portfolio_state"] = portfolio_analysis

        # 4. Generate text summary
        explanation["summary"] = self._generate_summary(explanation)

        # 5. Optional visualization
        if include_viz:
            explanation["visualization"] = self._create_visualization(
                observation, explanation
            )

        return explanation

    def _prepare_observation(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, th.Tensor]:
        """Convert numpy observation to torch tensors."""
        obs_tensor = {}
        for key, value in observation.items():
            if value is not None:
                # Add batch dimension if needed
                if value.ndim == 1:
                    value = value[np.newaxis, :]
                elif value.ndim == 2 and key == "price":
                    value = value[np.newaxis, :, :]

                obs_tensor[key] = th.from_numpy(value).float().to(self.device)

        return obs_tensor

    def _analyze_price_patterns(
        self, observation: Dict[str, np.ndarray], obs_tensor: Dict[str, th.Tensor]
    ) -> Dict:
        """
        Analyze price patterns using attention weights.

        Returns:
            Dict with:
            - attention_focus: Primary bars the agent focused on
            - attention_scores: Attention weights for each bar
            - pattern_detected: Inferred pattern type
            - confidence: Pattern confidence
        """
        # Run forward pass to cache intermediates
        with th.no_grad():
            _ = self.policy.features_extractor(obs_tensor)

        # Get cached intermediates
        intermediates = self.policy.features_extractor.get_intermediates()

        if "attention_weights" not in intermediates or intermediates["attention_weights"] is None:
            return {
                "attention_focus": {"primary_bars": [], "attention_scores": []},
                "pattern_detected": "unknown",
                "confidence": 0.0,
                "note": "Attention mechanism not available",
            }

        # Extract attention weights
        attn_weights = intermediates["attention_weights"]  # (B, num_heads, L, L)

        # Average across heads and get attention for last bar (most recent query)
        # Shape: (num_heads, L, L) -> (L, L) -> (L,)
        avg_attention = attn_weights[0].mean(dim=0)  # Average across heads
        recent_bar_attention = avg_attention[-1, :].numpy()  # Attention from last bar

        # Find top attended bars
        top_k = min(5, len(recent_bar_attention))
        top_indices = recent_bar_attention.argsort()[-top_k:][::-1]
        top_scores = recent_bar_attention[top_indices]

        # Infer pattern type based on attention distribution
        lookback = len(recent_bar_attention)
        recent_focus = recent_bar_attention[-10:].mean()  # Last 10 bars
        distant_focus = recent_bar_attention[:30].mean()  # First 30 bars
        middle_focus = recent_bar_attention[30:50].mean() if lookback >= 50 else 0.0

        if recent_focus > 0.4:
            pattern = "momentum_driven"
        elif distant_focus > recent_focus * 1.5:
            pattern = "mean_reversion"
        elif middle_focus > 0.3:
            pattern = "breakout_detection"
        else:
            pattern = "mixed_signals"

        return {
            "attention_focus": {
                "primary_bars": top_indices.tolist(),
                "attention_scores": top_scores.tolist(),
            },
            "pattern_detected": pattern,
            "confidence": float(recent_bar_attention.max()),
            "focus_distribution": {
                "recent": float(recent_focus),
                "middle": float(middle_focus),
                "distant": float(distant_focus),
            },
        }

    def _analyze_indicators(
        self, observation: Dict[str, np.ndarray], method: str = "attention"
    ) -> Dict:
        """
        Analyze technical indicator contributions.

        For now, returns indicator values. Future versions will add
        gradient-based attribution (saliency, integrated gradients).

        Args:
            observation: Observation dict
            method: Attribution method ("attention", "saliency", "integrated_gradients")

        Returns:
            Dict with indicator analysis
        """
        indicators = observation.get("indicators", np.array([]))

        if indicators is None or len(indicators) == 0:
            return {"top_contributors": [], "note": "No indicators available"}

        # For now, return top indicators by absolute value
        # TODO: Implement gradient-based attribution in Phase 2
        top_k = min(5, len(indicators))
        abs_values = np.abs(indicators)
        top_indices = abs_values.argsort()[-top_k:][::-1]

        top_contributors = []
        for idx in top_indices:
            name = self.feature_names[idx] if self.feature_names else f"indicator_{idx}"
            value = float(indicators[idx])

            # Infer signal based on indicator name and value
            signal = self._infer_signal(name, value)

            top_contributors.append(
                {
                    "name": name,
                    "value": value,
                    "contribution": float(abs_values[idx]),  # Placeholder
                    "signal": signal,
                }
            )

        return {
            "top_contributors": top_contributors,
            "note": f"Using {method} method (gradient attribution coming in Phase 2)",
        }

    def _infer_signal(self, indicator_name: str, value: float) -> str:
        """Infer signal type from indicator name and value."""
        name_lower = indicator_name.lower()

        # RSI signals
        if "rsi" in name_lower:
            if value < 30:
                return "oversold"
            elif value > 70:
                return "overbought"
            else:
                return "neutral"

        # MACD signals
        if "macd" in name_lower:
            if value > 0:
                return "bullish_crossover"
            elif value < 0:
                return "bearish_crossover"
            else:
                return "neutral"

        # Bollinger Bands
        if "bb_position" in name_lower or "bb_pos" in name_lower:
            if value < 0.2:
                return "near_lower_band"
            elif value > 0.8:
                return "near_upper_band"
            else:
                return "middle_range"

        # Volume
        if "volume" in name_lower and "ratio" in name_lower:
            if value > 1.5:
                return "high_volume"
            elif value < 0.5:
                return "low_volume"
            else:
                return "average_volume"

        # Default
        return "positive" if value > 0 else "negative"

    def _analyze_portfolio_state(self, observation: Dict[str, np.ndarray]) -> Dict:
        """Analyze portfolio state influence."""
        portfolio = observation.get("portfolio", np.array([]))

        if portfolio is None or len(portfolio) < 4:
            return {
                "position_pct": 0.0,
                "cash_pct": 1.0,
                "unrealized_pnl": 0.0,
                "risk_appetite": "unknown",
            }

        # Typical portfolio state: [position%, price_dev%, unrealized_pnl%, cash%]
        position_pct = float(portfolio[0])
        cash_pct = float(portfolio[3]) if len(portfolio) > 3 else 1.0
        unrealized_pnl = float(portfolio[2]) if len(portfolio) > 2 else 0.0

        # Infer risk appetite
        if position_pct > 0.7:
            risk = "conservative"  # Already heavily invested
        elif cash_pct < 0.2:
            risk = "aggressive"  # Low cash reserves
        else:
            risk = "neutral"

        return {
            "position_pct": position_pct,
            "cash_pct": cash_pct,
            "unrealized_pnl": unrealized_pnl,
            "risk_appetite": risk,
            "contribution": 0.05,  # Placeholder
        }

    def _generate_summary(self, explanation: Dict) -> str:
        """Generate human-readable summary of the explanation."""
        action = explanation["action"]
        confidence = explanation["confidence"]

        price_analysis = explanation["price_pattern_analysis"]
        indicator_analysis = explanation["indicator_analysis"]
        portfolio = explanation["portfolio_state"]

        # Build summary
        summary_parts = []

        # Action and confidence
        summary_parts.append(
            f"{action} executed with {confidence*100:.0f}% confidence."
        )

        # Price pattern
        pattern = price_analysis.get("pattern_detected", "unknown")
        primary_bars = price_analysis.get("attention_focus", {}).get("primary_bars", [])
        if primary_bars:
            bar_str = ", ".join(map(str, primary_bars[:3]))
            summary_parts.append(
                f"Primary driver: {pattern.replace('_', ' ')} pattern detected "
                f"(attention focused on bars {bar_str})."
            )

        # Top indicators
        top_indicators = indicator_analysis.get("top_contributors", [])[:3]
        if top_indicators:
            indicator_strs = [
                f"{ind['name']} {ind['signal']} ({ind['value']:.1f})"
                for ind in top_indicators
            ]
            summary_parts.append(f"Supporting signals: {', '.join(indicator_strs)}.")

        # Portfolio context
        if action == "BUY":
            cash = portfolio.get("cash_pct", 1.0)
            summary_parts.append(f"Cash available: {cash*100:.0f}%.")
        elif action == "SELL":
            pnl = portfolio.get("unrealized_pnl", 0.0)
            summary_parts.append(f"Unrealized P&L: {pnl:.1f}%.")

        return " ".join(summary_parts)

    def _create_visualization(
        self, observation: Dict[str, np.ndarray], explanation: Dict
    ) -> Optional[np.ndarray]:
        """
        Create visualization arrays (placeholder for Phase 1).

        This will be implemented with attention heatmaps over candlesticks
        in the attention_viz module.
        """
        # TODO: Implement in attention_viz.py
        return None

    def get_attention_weights(self, observation: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Get raw attention weights for an observation.

        Args:
            observation: Observation dict

        Returns:
            Attention weights array of shape (num_heads, lookback, lookback)
            or None if attention is not available
        """
        obs_tensor = self._prepare_observation(observation)

        with th.no_grad():
            _ = self.policy.features_extractor(obs_tensor)

        intermediates = self.policy.features_extractor.get_intermediates()
        attn_weights = intermediates.get("attention_weights")

        if attn_weights is not None:
            return attn_weights[0].numpy()  # Remove batch dim
        return None
