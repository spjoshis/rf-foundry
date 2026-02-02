"""Regime-conditioned action masking for trading environments."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from loguru import logger


@dataclass
class ActionMaskConfig:
    """
    Configuration for regime-conditioned action masking.

    Action masking dynamically restricts the action space based on detected
    market regime and directional bias to prevent statistically poor trades.

    Attributes:
        enabled: Whether action masking is enabled (default: False)
        regime_column: Column name for regime state in features (default: "regime_state")
        trend_bias_column: Column name for trend bias in features (default: "trend_bias")
        ranging_state: Regime state value for ranging market (default: 0)
        transition_state: Regime state value for transition market (default: 1)
        trending_state: Regime state value for trending market (default: 2)
        allow_hold_always: Whether Hold action is always allowed (default: True)

    Example:
        >>> config = ActionMaskConfig(enabled=True)
        >>> print(f"Masking enabled: {config.enabled}")
        Masking enabled: True
    """

    enabled: bool = False
    regime_column: str = "regime_state"
    trend_bias_column: str = "trend_bias"
    ranging_state: int = 0
    transition_state: int = 1
    trending_state: int = 2
    allow_hold_always: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.ranging_state == self.trending_state:
            raise ValueError(
                f"ranging_state ({self.ranging_state}) and trending_state "
                f"({self.trending_state}) must be different"
            )
        if self.ranging_state == self.transition_state:
            raise ValueError(
                f"ranging_state ({self.ranging_state}) and transition_state "
                f"({self.transition_state}) must be different"
            )
        if self.trending_state == self.transition_state:
            raise ValueError(
                f"trending_state ({self.trending_state}) and transition_state "
                f"({self.transition_state}) must be different"
            )

        if self.enabled:
            logger.info(
                f"ActionMaskConfig initialized: enabled={self.enabled}, "
                f"regime_col='{self.regime_column}', trend_col='{self.trend_bias_column}'"
            )


class RegimeActionMask:
    """
    Generate action masks based on market regime and trend bias.

    Mask Logic:
    -----------
    Regime      | Trend Bias | Allowed Actions
    ------------|------------|------------------
    Ranging     | Any        | Hold
    Transition  | Any        | Hold, Buy, Sell
    Trending    | +1 (up)    | Hold, Buy
    Trending    | -1 (down)  | Hold, Sell
    Trending    | 0 (neutral)| Hold

    Actions:
        0: Hold
        1: Buy
        2: Sell

    Example:
        >>> config = ActionMaskConfig(enabled=True)
        >>> masker = RegimeActionMask(config)
        >>> mask = masker.get_mask(regime_state=2, trend_bias=1)
        >>> print(mask)  # [True, True, False] - Hold and Buy allowed
        [True, True, False]
    """

    def __init__(self, config: Optional[ActionMaskConfig] = None):
        """
        Initialize regime action mask generator.

        Args:
            config: Action mask configuration (uses defaults if None)
        """
        self.config = config or ActionMaskConfig()

        if self.config.enabled:
            logger.info(
                f"RegimeActionMask initialized: masking enabled, "
                f"regime states=[{self.config.ranging_state}, "
                f"{self.config.transition_state}, {self.config.trending_state}]"
            )

    def get_mask(
        self,
        regime_state: int,
        trend_bias: int,
    ) -> np.ndarray:
        """
        Generate action mask for current regime and trend.

        Args:
            regime_state: Current regime (0=ranging, 1=transition, 2=trending)
            trend_bias: Trend direction (-1=down, 0=neutral, +1=up)

        Returns:
            Boolean mask array [Hold_valid, Buy_valid, Sell_valid]
            True means action is allowed, False means action is masked

        Raises:
            ValueError: If regime_state or trend_bias are invalid
        """
        # If masking disabled, allow all actions
        if not self.config.enabled:
            return np.ones(3, dtype=bool)

        # Validate inputs
        valid_regimes = {
            self.config.ranging_state,
            self.config.transition_state,
            self.config.trending_state
        }
        if regime_state not in valid_regimes:
            raise ValueError(
                f"Invalid regime_state: {regime_state}. "
                f"Expected one of {valid_regimes}"
            )

        if trend_bias not in {-1, 0, 1}:
            raise ValueError(
                f"Invalid trend_bias: {trend_bias}. Expected -1, 0, or 1"
            )

        # Initialize mask: [Hold, Buy, Sell]
        mask = np.zeros(3, dtype=bool)

        # Hold is always allowed (if configured)
        if self.config.allow_hold_always:
            mask[0] = True

        # Apply regime-specific rules
        if regime_state == self.config.ranging_state:
            # Ranging: Only Hold allowed
            # mask[0] already set to True above
            pass

        elif regime_state == self.config.transition_state:
            # Transition: All actions allowed
            mask[:] = True

        elif regime_state == self.config.trending_state:
            # Trending: Direction-dependent
            if trend_bias == 1:
                # Uptrend: Hold + Buy
                mask[1] = True
            elif trend_bias == -1:
                # Downtrend: Hold + Sell
                mask[2] = True
            # else: trend_bias == 0, only Hold (already set)

        logger.debug(
            f"Mask generated: regime={regime_state}, bias={trend_bias}, "
            f"mask={mask} → [Hold={mask[0]}, Buy={mask[1]}, Sell={mask[2]}]"
        )

        return mask

    def is_action_valid(
        self,
        action: int,
        regime_state: int,
        trend_bias: int,
    ) -> bool:
        """
        Check if a specific action is valid for current regime/trend.

        Args:
            action: Action to validate (0=Hold, 1=Buy, 2=Sell)
            regime_state: Current regime state
            trend_bias: Current trend bias

        Returns:
            True if action is allowed, False if masked

        Raises:
            ValueError: If action is invalid (not in {0, 1, 2})
        """
        if action not in {0, 1, 2}:
            raise ValueError(f"Invalid action: {action}. Expected 0, 1, or 2")

        mask = self.get_mask(regime_state, trend_bias)
        return bool(mask[action])

    def get_valid_actions(
        self,
        regime_state: int,
        trend_bias: int,
    ) -> list[int]:
        """
        Get list of valid action indices for current regime/trend.

        Args:
            regime_state: Current regime state
            trend_bias: Current trend bias

        Returns:
            List of valid action indices

        Example:
            >>> masker = RegimeActionMask(ActionMaskConfig(enabled=True))
            >>> masker.get_valid_actions(regime_state=2, trend_bias=1)
            [0, 1]  # Hold and Buy
        """
        mask = self.get_mask(regime_state, trend_bias)
        return [i for i, valid in enumerate(mask) if valid]

    def get_mask_statistics(
        self,
        regime_states: np.ndarray,
        trend_biases: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute statistics about action masking for a sequence.

        Args:
            regime_states: Array of regime states
            trend_biases: Array of trend biases

        Returns:
            Dictionary with masking statistics:
                - hold_allowed_pct: % of timesteps where Hold is allowed
                - buy_allowed_pct: % of timesteps where Buy is allowed
                - sell_allowed_pct: % of timesteps where Sell is allowed
                - all_allowed_pct: % of timesteps where all actions allowed
                - restricted_pct: % of timesteps with some restrictions

        Example:
            >>> masker = RegimeActionMask(ActionMaskConfig(enabled=True))
            >>> regimes = np.array([0, 1, 2, 2])
            >>> biases = np.array([0, 0, 1, -1])
            >>> stats = masker.get_mask_statistics(regimes, biases)
        """
        if len(regime_states) != len(trend_biases):
            raise ValueError(
                f"regime_states ({len(regime_states)}) and trend_biases "
                f"({len(trend_biases)}) must have same length"
            )

        n_steps = len(regime_states)
        if n_steps == 0:
            return {
                "hold_allowed_pct": 0.0,
                "buy_allowed_pct": 0.0,
                "sell_allowed_pct": 0.0,
                "all_allowed_pct": 0.0,
                "restricted_pct": 0.0,
            }

        # Generate masks for all timesteps
        masks = np.array([
            self.get_mask(int(regime), int(bias))
            for regime, bias in zip(regime_states, trend_biases)
        ])

        # Compute statistics
        hold_allowed = masks[:, 0].sum() / n_steps * 100
        buy_allowed = masks[:, 1].sum() / n_steps * 100
        sell_allowed = masks[:, 2].sum() / n_steps * 100
        all_allowed = masks.all(axis=1).sum() / n_steps * 100
        restricted = (~masks.all(axis=1)).sum() / n_steps * 100

        return {
            "hold_allowed_pct": float(hold_allowed),
            "buy_allowed_pct": float(buy_allowed),
            "sell_allowed_pct": float(sell_allowed),
            "all_allowed_pct": float(all_allowed),
            "restricted_pct": float(restricted),
        }
