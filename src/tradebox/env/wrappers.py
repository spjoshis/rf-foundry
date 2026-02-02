"""
Gymnasium environment wrappers for observation transformation.

This module provides wrappers to convert flat Box observation spaces
into structured Dict observation spaces suitable for CNN-based feature extraction.
"""

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DictObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert flat Box observations into Dict observations.

    Splits flat observations into structured components:
    - price: Raw OHLCV data (channels × time)
    - indicators: Derived technical indicators (flat vector, current values)
    - portfolio: Portfolio state (position, cash%, entry_price, unrealized_pnl)

    This enables CNN-based feature extraction on raw price data while
    preserving indicator and portfolio information.

    Attributes:
        lookback_window: Number of historical bars in price window
        n_price_channels: Number of price channels (typically 5 for OHLCV)
        n_indicators: Number of technical indicator features
        n_portfolio: Number of portfolio state features (typically 4)

    Example:
        >>> from tradebox.env import TradingEnv
        >>> env = TradingEnv(data, features, config)
        >>> wrapped_env = DictObservationWrapper(env, data, lookback_window=60)
        >>> obs, info = wrapped_env.reset()
        >>> print(obs.keys())  # dict_keys(['price', 'indicators', 'portfolio'])
    """

    def __init__(
        self,
        env: gym.Env,
        data: "pd.DataFrame",
        lookback_window: int,
        n_price_channels: int = 5,
    ) -> None:
        """
        Initialize observation wrapper.

        Args:
            env: Base trading environment with flat Box observation space
            data: OHLCV DataFrame (needed to extract raw price data)
            lookback_window: Number of bars in lookback window
            n_price_channels: Number of price channels (5 for OHLCV)

        Raises:
            ValueError: If observation space is not Box or dimensions don't match
        """
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise ValueError(
                f"DictObservationWrapper requires Box observation space, "
                f"got {type(env.observation_space)}"
            )

        self.data = data
        self.lookback_window = lookback_window
        self.n_price_channels = n_price_channels

        # Calculate component dimensions from flat observation space
        flat_obs_dim = env.observation_space.shape[0]

        # Portfolio state is always last 4 features
        self.n_portfolio = 4

        # Technical features are windowed (lookback × n_indicators)
        # Fundamental features are static (n_fundamental)
        # So: flat_obs_dim = lookback × n_tech + n_fund + 4

        # We need to infer n_tech and n_fund
        # Access environment's internal feature columns
        if hasattr(env, "_technical_cols") and hasattr(env, "_fundamental_cols"):
            self.n_tech_features = len(env._technical_cols)
            self.n_fund_features = len(env._fundamental_cols)
        else:
            # Fallback: Assume no fundamentals (backward compatibility)
            n_features_before_portfolio = flat_obs_dim - self.n_portfolio
            self.n_tech_features = n_features_before_portfolio // lookback_window
            self.n_fund_features = 0

        # Total indicators = all technical (latest) + all fundamental
        self.n_indicators = self.n_tech_features + self.n_fund_features

        # Define new Dict observation space
        # Note: TradingCNNExtractor expects price shape (lookback, channels), not (channels, lookback)
        self.observation_space = spaces.Dict({
            "price": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(lookback_window, n_price_channels),
                dtype=np.float32,
            ),
            "indicators": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_indicators,),
                dtype=np.float32,
            ),
            "portfolio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_portfolio,),
                dtype=np.float32,
            ),
        })

    def observation(self, flat_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert flat observation to dict observation.

        Args:
            flat_obs: Flat observation from base environment
                Shape: (lookback × n_tech + n_fund + 4,)

        Returns:
            Dict observation with keys:
                - "price": (lookback_window, n_price_channels)
                - "indicators": (n_indicators,)
                - "portfolio": (4,)
        """
        # Split flat observation into components
        n_tech_windowed = self.lookback_window * self.n_tech_features
        n_fund_static = self.n_fund_features

        # Extract technical window, fundamental static, and portfolio
        tech_window = flat_obs[:n_tech_windowed]
        fund_static = flat_obs[n_tech_windowed:n_tech_windowed + n_fund_static]
        portfolio = flat_obs[-self.n_portfolio:]

        # Reshape technical features: (lookback × n_tech,) → (lookback, n_tech)
        tech_reshaped = tech_window.reshape(self.lookback_window, self.n_tech_features)

        # Extract raw price data from environment's current position
        current_step = self.env.current_step
        lookback_start = current_step - self.lookback_window

        # Get OHLCV data: (lookback_window, 5)
        # TradingCNNExtractor expects (lookback, channels) format
        price_window = self.data.iloc[
            lookback_start:current_step, :self.n_price_channels
        ].values

        # Extract latest technical indicator values (last timestep)
        latest_indicators = tech_reshaped[-1, :]  # (n_tech,)

        # Combine latest technical + all fundamental → indicators
        if n_fund_static > 0:
            indicators = np.concatenate([latest_indicators, fund_static])
        else:
            indicators = latest_indicators

        # Return dict observation
        return {
            "price": price_window.astype(np.float32),
            "indicators": indicators.astype(np.float32),
            "portfolio": portfolio.astype(np.float32),
        }


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to normalize Dict observations for CNN training.

    Applies different normalization strategies to each component:
    - Price: Log returns normalization (to make stationarity)
    - Indicators: Z-score normalization (assumed pre-normalized in features)
    - Portfolio: Custom scaling for each component

    Note: This is optional and might not be needed if your features
    are already normalized. Use for improved CNN training stability.

    Example:
        >>> env = DictObservationWrapper(base_env, data, lookback_window=60)
        >>> normalized_env = NormalizeObservationWrapper(env)
    """

    def __init__(
        self,
        env: gym.Env,
        normalize_price: bool = True,
        price_norm_type: str = "log_returns",
    ) -> None:
        """
        Initialize normalization wrapper.

        Args:
            env: Environment with Dict observation space
            normalize_price: Whether to normalize price data
            price_norm_type: Type of price normalization
                - "log_returns": Convert to log returns (default)
                - "zscore": Z-score normalization
                - "minmax": Min-max scaling to [-1, 1]
        """
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError(
                f"NormalizeObservationWrapper requires Dict observation space, "
                f"got {type(env.observation_space)}"
            )

        self.normalize_price = normalize_price
        self.price_norm_type = price_norm_type

    def observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize dict observation.

        Args:
            obs: Dict observation with price, indicators, portfolio

        Returns:
            Normalized dict observation
        """
        normalized = obs.copy()

        if self.normalize_price and "price" in obs:
            price = obs["price"]  # (time, channels)

            if self.price_norm_type == "log_returns":
                # Convert to log returns (preserve first value for context)
                log_price = np.log(price + 1e-8)  # Avoid log(0)
                # Calculate returns: log(p_t) - log(p_{t-1})
                log_returns = np.diff(log_price, axis=0)  # (time-1, channels)
                # Prepend zeros for first timestep
                log_returns = np.concatenate([
                    np.zeros((1, price.shape[1])),
                    log_returns
                ], axis=0)
                normalized["price"] = log_returns.astype(np.float32)

            elif self.price_norm_type == "zscore":
                # Z-score normalization per channel
                mean = price.mean(axis=0, keepdims=True)
                std = price.std(axis=0, keepdims=True) + 1e-8
                normalized["price"] = ((price - mean) / std).astype(np.float32)

            elif self.price_norm_type == "minmax":
                # Min-max scaling to [-1, 1] per channel
                min_val = price.min(axis=0, keepdims=True)
                max_val = price.max(axis=0, keepdims=True)
                range_val = max_val - min_val + 1e-8
                normalized["price"] = (2 * (price - min_val) / range_val - 1).astype(np.float32)

        # Indicators and portfolio are assumed already normalized
        return normalized


def make_cnn_compatible_env(
    env: gym.Env,
    data: "pd.DataFrame",
    lookback_window: int,
    normalize: bool = False,
) -> gym.Env:
    """
    Helper function to wrap environment for CNN compatibility.

    Args:
        env: Base trading environment
        data: OHLCV DataFrame
        lookback_window: Lookback window size
        normalize: Whether to apply normalization

    Returns:
        Wrapped environment with Dict observation space

    Example:
        >>> from tradebox.env import TradingEnv, make_cnn_compatible_env
        >>> base_env = TradingEnv(data, features, config)
        >>> cnn_env = make_cnn_compatible_env(base_env, data, lookback_window=60)
    """
    # Convert to Dict observation space
    env = DictObservationWrapper(env, data, lookback_window)

    # Optionally add normalization
    if normalize:
        env = NormalizeObservationWrapper(env, normalize_price=True)

    return env
