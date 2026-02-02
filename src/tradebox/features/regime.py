"""Market regime detection for trading strategies."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RegimeConfig:
    """
    Configuration for regime detection.

    Supports trend-based, volatility-based, and volume-based regime classification.

    Attributes:
        regime_type: Type of regime detection ("trend", "volatility", "volume")
        trending_threshold: ADX threshold for trending market (default: 25.0)
        ranging_threshold: ADX threshold for ranging market (default: 20.0)
        use_directional_bias: Use +DI/-DI for trend direction (default: True)
        di_diff_threshold: Min DI difference for directional bias (default: 5.0)
        encode_as_onehot: One-hot encoding vs single value (default: False)
        smooth_regime: Apply regime persistence filter (default: False)
        min_regime_duration: Min bars to confirm regime change (default: 5)
    """

    regime_type: Literal["trend", "volatility", "volume"] = "trend"
    trending_threshold: float = 25.0
    ranging_threshold: float = 20.0
    use_directional_bias: bool = True
    di_diff_threshold: float = 5.0
    encode_as_onehot: bool = False
    smooth_regime: bool = False
    min_regime_duration: int = 5

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.ranging_threshold >= self.trending_threshold:
            raise ValueError(
                f"ranging_threshold ({self.ranging_threshold}) must be < "
                f"trending_threshold ({self.trending_threshold})"
            )
        if self.di_diff_threshold < 0:
            raise ValueError(f"di_diff_threshold must be >= 0, got {self.di_diff_threshold}")
        if self.min_regime_duration < 1:
            raise ValueError(f"min_regime_duration must be >= 1, got {self.min_regime_duration}")


class RegimeDetector:
    """
    Detect and classify market regimes from technical indicators.

    Supports:
    - Trend-based regime (ADX): ranging, transition, trending
    - Directional bias (+DI/-DI): uptrend, downtrend, neutral
    - Regime smoothing to prevent rapid flipping

    Example:
        >>> config = RegimeConfig(trending_threshold=25.0)
        >>> detector = RegimeDetector(config)
        >>> regime_df = detector.detect(indicators)
        >>> print(regime_df.columns)
        ['regime_state', 'regime_strength', 'trend_bias', 'regime_persistence']
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize regime detector.

        Args:
            config: Regime detection configuration (uses defaults if None)
        """
        self.config = config or RegimeConfig()
        logger.info(
            f"RegimeDetector initialized: type={self.config.regime_type}, "
            f"thresholds=[{self.config.ranging_threshold}, {self.config.trending_threshold}]"
        )

    def detect(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime for each timestep.

        Args:
            indicators: DataFrame with ADX, Plus_DI, Minus_DI columns

        Returns:
            DataFrame with regime features:
            - regime_state: int [0=range, 1=transition, 2=trending]
            - regime_strength: float [0-1] (normalized ADX)
            - trend_bias: int [-1=down, 0=neutral, 1=up] (if use_directional_bias)
            - regime_persistence: int (bars in current regime)

        Raises:
            ValueError: If required indicator columns are missing
        """
        if self.config.regime_type == "trend":
            return self._detect_trend_regime(indicators)
        else:
            raise NotImplementedError(
                f"Regime type '{self.config.regime_type}' not yet implemented"
            )

    def _detect_trend_regime(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Detect trend-based regime using ADX and directional indicators.

        Regime States:
        - 0 (Ranging): ADX < ranging_threshold (weak trend)
        - 1 (Transition): ranging_threshold <= ADX < trending_threshold
        - 2 (Trending): ADX >= trending_threshold (strong trend)

        Args:
            indicators: DataFrame with ADX, Plus_DI, Minus_DI columns

        Returns:
            DataFrame with regime features
        """
        # Validate required columns
        required_cols = ["ADX"]
        if self.config.use_directional_bias:
            required_cols.extend(["Plus_DI", "Minus_DI"])

        missing_cols = [col for col in required_cols if col not in indicators.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for trend regime detection: {missing_cols}"
            )

        # Extract ADX values
        adx = indicators["ADX"].fillna(0)

        # Classify regime state based on ADX thresholds
        regime_state = self._classify_trend_regime(adx)

        # Apply smoothing if enabled
        if self.config.smooth_regime:
            regime_state = self._apply_regime_smoothing(regime_state)

        # Compute regime strength (normalized ADX)
        regime_strength = adx / 100.0  # ADX range: 0-100 → [0, 1]

        # Compute directional bias (optional)
        if self.config.use_directional_bias:
            plus_di = indicators["Plus_DI"].fillna(0)
            minus_di = indicators["Minus_DI"].fillna(0)
            trend_bias = self._compute_directional_bias(plus_di, minus_di)
        else:
            trend_bias = pd.Series(0, index=indicators.index)

        # Compute regime persistence (bars in current regime)
        regime_persistence = self._compute_regime_persistence(regime_state)

        # Build output DataFrame
        regime_df = pd.DataFrame({
            "regime_state": regime_state.astype(int),
            "regime_strength": regime_strength.astype(np.float32),
            "trend_bias": trend_bias.astype(int),
            "regime_persistence": regime_persistence.astype(int),
        }, index=indicators.index)

        logger.debug(
            f"Regime detection complete: "
            f"{(regime_state == 0).sum()} ranging, "
            f"{(regime_state == 1).sum()} transition, "
            f"{(regime_state == 2).sum()} trending"
        )

        return regime_df

    def _classify_trend_regime(self, adx: pd.Series) -> pd.Series:
        """
        Classify regime state based on ADX thresholds.

        Args:
            adx: ADX indicator series

        Returns:
            Regime state series: 0 (range), 1 (transition), 2 (trending)
        """
        regime_state = pd.Series(1, index=adx.index)  # Default: transition

        # Ranging: ADX < ranging_threshold
        regime_state[adx < self.config.ranging_threshold] = 0

        # Trending: ADX >= trending_threshold
        regime_state[adx >= self.config.trending_threshold] = 2

        return regime_state

    def _compute_directional_bias(
        self,
        plus_di: pd.Series,
        minus_di: pd.Series
    ) -> pd.Series:
        """
        Compute trend direction from directional indicators.

        Args:
            plus_di: +DI (positive directional indicator)
            minus_di: -DI (negative directional indicator)

        Returns:
            Trend bias: -1 (downtrend), 0 (neutral), 1 (uptrend)
        """
        di_diff = plus_di - minus_di

        trend_bias = pd.Series(0, index=plus_di.index)  # Default: neutral

        # Uptrend: +DI significantly > -DI
        trend_bias[di_diff > self.config.di_diff_threshold] = 1

        # Downtrend: -DI significantly > +DI
        trend_bias[di_diff < -self.config.di_diff_threshold] = -1

        return trend_bias

    def _apply_regime_smoothing(self, regime_state: pd.Series) -> pd.Series:
        """
        Filter out regime changes below minimum duration.

        Prevents rapid flipping between regimes by requiring
        a minimum number of bars to confirm a regime change.

        Args:
            regime_state: Raw regime state series

        Returns:
            Smoothed regime state series
        """
        smoothed = regime_state.copy()
        current_regime = regime_state.iloc[0]
        regime_start = 0

        for i in range(1, len(regime_state)):
            if regime_state.iloc[i] != current_regime:
                # Check if previous regime was long enough
                if i - regime_start < self.config.min_regime_duration:
                    # Revert to previous regime (too short)
                    smoothed.iloc[regime_start:i] = current_regime
                else:
                    # Accept regime change
                    current_regime = regime_state.iloc[i]
                    regime_start = i

        return smoothed

    def _compute_regime_persistence(self, regime_state: pd.Series) -> pd.Series:
        """
        Compute number of bars in current regime.

        Args:
            regime_state: Regime state series

        Returns:
            Persistence series (bars in current regime)
        """
        persistence = pd.Series(1, index=regime_state.index)

        for i in range(1, len(regime_state)):
            if regime_state.iloc[i] == regime_state.iloc[i-1]:
                persistence.iloc[i] = persistence.iloc[i-1] + 1
            else:
                persistence.iloc[i] = 1

        return persistence

    def get_regime_summary(self, regime_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return statistics about regime distribution.

        Args:
            regime_df: Output from detect() method

        Returns:
            Dictionary with regime statistics
        """
        regime_state = regime_df["regime_state"]

        total_bars = len(regime_state)
        ranging_bars = (regime_state == 0).sum()
        transition_bars = (regime_state == 1).sum()
        trending_bars = (regime_state == 2).sum()

        return {
            "total_bars": total_bars,
            "ranging_pct": ranging_bars / total_bars * 100,
            "transition_pct": transition_bars / total_bars * 100,
            "trending_pct": trending_bars / total_bars * 100,
            "avg_regime_duration": regime_df["regime_persistence"].mean(),
            "max_regime_duration": regime_df["regime_persistence"].max(),
        }
