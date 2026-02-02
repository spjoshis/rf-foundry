"""
CNN-based feature extractor for trading RL agents.

This module provides a sophisticated CNN architecture that learns directly from
raw OHLCV price data, treating charts as images and learning candlestick patterns,
trends, and market structure. Indicators are used as auxiliary context rather than
primary features.

Key Features:
    - Multi-channel CNN for OHLCV data
    - Temporal pattern recognition (trends, consolidations, breakouts)
    - Attention mechanism for important price levels
    - Indicator context integration
    - Portfolio state embedding

Architecture Philosophy:
    - Raw price action is primary (CNN on OHLCV)
    - Technical indicators are secondary context
    - Portfolio state is integrated at final layer
    - Learn true chart patterns, not indicator stacking

Example:
    >>> from tradebox.models.trading_cnn_extractor import TradingCNNExtractor
    >>> from gymnasium import spaces
    >>>
    >>> obs_space = spaces.Dict({
    ...     "price": spaces.Box(0, np.inf, shape=(60, 5)),  # 60 bars × OHLCV
    ...     "indicators": spaces.Box(-np.inf, np.inf, shape=(27,)),
    ...     "portfolio": spaces.Box(-np.inf, np.inf, shape=(4,))
    ... })
    >>>
    >>> extractor = TradingCNNExtractor(obs_space, features_dim=256)
    >>> features = extractor(obs_dict)  # (batch_size, 256)
"""

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class PricePatternCNN(nn.Module):
    """
    CNN for learning price patterns from OHLCV candlestick data.

    Architecture:
        - Multi-scale convolutions to capture different timeframe patterns
        - Residual connections for gradient flow
        - Attention mechanism for important price levels
        - Learns: trends, consolidations, breakouts, volatility patterns

    Input: (batch_size, lookback, 5) where 5 = [Open, High, Low, Close, Volume]
    Output: (batch_size, embedding_dim)
    """

    def __init__(
        self,
        lookback_window: int = 60,
        in_channels: int = 5,
        embedding_dim: int = 128,
        use_attention: bool = True,
        capture_attention: bool = False,
    ):
        """
        Initialize price pattern CNN.

        Args:
            lookback_window: Number of historical bars (e.g., 60)
            in_channels: Number of input channels (5 for OHLCV)
            embedding_dim: Output embedding dimension
            use_attention: Whether to use attention mechanism
            capture_attention: Whether to capture and store attention weights for explainability
        """
        super().__init__()

        self.lookback_window = lookback_window
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.use_attention = use_attention
        self.capture_attention = capture_attention
        self._attention_weights = None  # Cache for attention weights

        # Input: (B, lookback, 5) -> permute to (B, 5, lookback) for Conv1d

        # Multi-scale feature extraction
        # Short-term patterns (recent 5-10 bars)
        self.conv_short = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Medium-term patterns (10-30 bars)
        self.conv_medium = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Long-term patterns (30-60 bars)
        self.conv_long = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),  # 64*3 = 192
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Optional attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=4,
                batch_first=False,
            )

        # Final embedding
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, price_data: th.Tensor) -> th.Tensor:
        """
        Extract features from price data.

        Args:
            price_data: (batch_size, lookback, 5) OHLCV data

        Returns:
            (batch_size, embedding_dim) price pattern embeddings
        """
        # Permute to (B, C, L) for Conv1d
        x = price_data.permute(0, 2, 1)  # (B, 5, lookback)

        # Multi-scale convolutions
        short = self.conv_short(x)  # (B, 64, lookback)
        medium = self.conv_medium(x)  # (B, 64, lookback)
        long = self.conv_long(x)  # (B, 64, lookback)

        # Concatenate multi-scale features
        multi_scale = th.cat([short, medium, long], dim=1)  # (B, 192, lookback)

        # Fuse features
        fused = self.fusion(multi_scale)  # (B, 128, lookback)

        # Apply attention if enabled
        if self.use_attention:
            # Permute to (L, B, C) for attention
            fused_t = fused.permute(2, 0, 1)  # (lookback, B, 128)
            attn_out, attn_weights = self.attention(
                fused_t, fused_t, fused_t,
                need_weights=True,
                average_attn_weights=False  # Get all heads for detailed analysis
            )

            # Cache attention weights for explainability if enabled
            if self.capture_attention:
                # attn_weights shape: (B, num_heads, lookback, lookback)
                self._attention_weights = attn_weights.detach().cpu()

            fused = attn_out.permute(1, 2, 0)  # (B, 128, lookback)

        # Global pooling
        pooled = self.global_pool(fused).squeeze(-1)  # (B, 128)

        # Final embedding
        embedding = self.embedding(pooled)  # (B, embedding_dim)

        return embedding


class IndicatorContextEncoder(nn.Module):
    """
    MLP encoder for technical indicators as auxiliary context.

    Indicators provide complementary information but are not the primary
    decision driver. The CNN learns raw patterns; indicators add context.
    """

    def __init__(
        self,
        n_indicators: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ):
        """
        Initialize indicator encoder.

        Args:
            n_indicators: Number of technical indicators
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_indicators, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, indicators: th.Tensor) -> th.Tensor:
        """
        Encode indicators.

        Args:
            indicators: (batch_size, n_indicators)

        Returns:
            (batch_size, embedding_dim) indicator embeddings
        """
        return self.encoder(indicators)


class PortfolioStateEncoder(nn.Module):
    """
    Encoder for portfolio state (position, cash, PnL, etc.).

    Portfolio state influences risk-taking behavior and position sizing.
    """

    def __init__(
        self,
        state_dim: int = 4,
        embedding_dim: int = 16,
    ):
        """
        Initialize portfolio state encoder.

        Args:
            state_dim: Portfolio state dimension (e.g., 4 for position, cash, pnl, price_dev)
            embedding_dim: Output embedding dimension
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, portfolio_state: th.Tensor) -> th.Tensor:
        """
        Encode portfolio state.

        Args:
            portfolio_state: (batch_size, state_dim)

        Returns:
            (batch_size, embedding_dim) portfolio embeddings
        """
        return self.encoder(portfolio_state)


class TradingCNNExtractor(BaseFeaturesExtractor):
    """
    Complete CNN-based feature extractor for trading RL.

    Combines:
        1. PricePatternCNN: Learns from raw OHLCV data (primary)
        2. IndicatorContextEncoder: Encodes technical indicators (context)
        3. PortfolioStateEncoder: Encodes portfolio state

    Architecture Philosophy:
        - Price patterns drive decisions (highest weight)
        - Indicators provide context (medium weight)
        - Portfolio state modulates risk (lower weight)

    This follows the principle: "Trade what you see (charts),
    not what you think (indicators)."

    Example:
        >>> obs_space = spaces.Dict({
        ...     "price": spaces.Box(0, np.inf, shape=(60, 5)),
        ...     "indicators": spaces.Box(-np.inf, np.inf, shape=(27,)),
        ...     "portfolio": spaces.Box(-np.inf, np.inf, shape=(4,))
        ... })
        >>> extractor = TradingCNNExtractor(obs_space, features_dim=256)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_embedding_dim: int = 128,
        indicator_embedding_dim: int = 32,
        portfolio_embedding_dim: int = 16,
        fundamental_embedding_dim: int = 16,
        use_attention: bool = True,
        capture_intermediates: bool = False,
    ):
        """
        Initialize trading CNN extractor.

        Args:
            observation_space: Dict observation space with keys:
                - "price": (lookback, 5) for OHLCV
                - "indicators": (n_indicators,) for technical indicators
                - "portfolio": (state_dim,) for portfolio state
                - "fundamentals": (n_fundamentals,) [optional] for fundamental features
            features_dim: Total output feature dimension
            cnn_embedding_dim: CNN output dimension
            indicator_embedding_dim: Indicator encoder output dimension
            portfolio_embedding_dim: Portfolio encoder output dimension
            fundamental_embedding_dim: Fundamental encoder output dimension
            use_attention: Whether to use attention in CNN
            capture_intermediates: Whether to cache intermediate embeddings for explainability

        Raises:
            ValueError: If observation space is not Dict or missing required keys
        """
        # Call parent constructor
        super().__init__(observation_space, features_dim=features_dim)

        # Explainability flags
        self.capture_intermediates = capture_intermediates
        self._intermediates = {}  # Cache for intermediate activations

        # Validate observation space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                f"TradingCNNExtractor requires Dict observation space, "
                f"got {type(observation_space)}"
            )

        required_keys = {"price", "indicators", "portfolio"}
        if not required_keys.issubset(observation_space.spaces.keys()):
            raise ValueError(
                f"Observation space must contain keys {required_keys}, "
                f"got {observation_space.spaces.keys()}"
            )

        # Extract dimensions
        price_shape = observation_space["price"].shape
        if len(price_shape) != 2 or price_shape[1] != 5:
            raise ValueError(
                f"Price space should be (lookback, 5) for OHLCV, got {price_shape}"
            )

        lookback_window = price_shape[0]
        n_indicators = observation_space["indicators"].shape[0]
        portfolio_dim = observation_space["portfolio"].shape[0]

        # Check for optional fundamentals
        self.has_fundamentals = "fundamentals" in observation_space.spaces
        n_fundamentals = (
            observation_space["fundamentals"].shape[0] if self.has_fundamentals else 0
        )

        # Create sub-modules
        self.price_cnn = PricePatternCNN(
            lookback_window=lookback_window,
            in_channels=5,
            embedding_dim=cnn_embedding_dim,
            use_attention=use_attention,
            capture_attention=capture_intermediates,  # Enable attention capture for explainability
        )

        self.indicator_encoder = IndicatorContextEncoder(
            n_indicators=n_indicators,
            embedding_dim=indicator_embedding_dim,
        )

        self.portfolio_encoder = PortfolioStateEncoder(
            state_dim=portfolio_dim,
            embedding_dim=portfolio_embedding_dim,
        )

        # Optional fundamental encoder
        if self.has_fundamentals:
            self.fundamental_encoder = IndicatorContextEncoder(
                n_indicators=n_fundamentals,
                hidden_dim=32,
                embedding_dim=fundamental_embedding_dim,
            )
            combined_dim = (
                cnn_embedding_dim
                + indicator_embedding_dim
                + portfolio_embedding_dim
                + fundamental_embedding_dim
            )
        else:
            self.fundamental_encoder = None
            combined_dim = (
                cnn_embedding_dim + indicator_embedding_dim + portfolio_embedding_dim
            )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Dict with keys:
                - "price": (batch_size, lookback, 5)
                - "indicators": (batch_size, n_indicators)
                - "portfolio": (batch_size, portfolio_dim)
                - "fundamentals": (batch_size, n_fundamentals) [optional]

        Returns:
            (batch_size, features_dim) combined features
        """
        # Extract embeddings from each component
        price_emb = self.price_cnn(observations["price"])
        indicator_emb = self.indicator_encoder(observations["indicators"])
        portfolio_emb = self.portfolio_encoder(observations["portfolio"])

        # Cache intermediate embeddings for explainability if enabled
        if self.capture_intermediates:
            self._intermediates = {
                "price_embedding": price_emb.detach().cpu(),
                "indicator_embedding": indicator_emb.detach().cpu(),
                "portfolio_embedding": portfolio_emb.detach().cpu(),
                "attention_weights": self.price_cnn._attention_weights if self.price_cnn.use_attention else None,
            }

        # Concatenate embeddings
        if self.has_fundamentals and "fundamentals" in observations:
            fundamental_emb = self.fundamental_encoder(observations["fundamentals"])

            # Cache fundamental embedding
            if self.capture_intermediates:
                self._intermediates["fundamental_embedding"] = fundamental_emb.detach().cpu()

            combined = th.cat(
                [price_emb, indicator_emb, fundamental_emb, portfolio_emb], dim=1
            )
        else:
            combined = th.cat([price_emb, indicator_emb, portfolio_emb], dim=1)

        # Final fusion
        features = self.fusion(combined)

        return features

    def get_intermediates(self) -> Dict[str, Optional[th.Tensor]]:
        """
        Get cached intermediate activations for explainability.

        Returns:
            Dict with keys:
                - "price_embedding": (batch_size, cnn_embedding_dim)
                - "indicator_embedding": (batch_size, indicator_embedding_dim)
                - "portfolio_embedding": (batch_size, portfolio_embedding_dim)
                - "fundamental_embedding": (batch_size, fundamental_embedding_dim) [if available]
                - "attention_weights": (batch_size, num_heads, lookback, lookback) [if attention enabled]

        Note:
            Only available if capture_intermediates=True was set during initialization.
            Returns empty dict if capture_intermediates=False.
        """
        return self._intermediates if self.capture_intermediates else {}
