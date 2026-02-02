"""
Hybrid CNN-MLP feature extractor for Stable-Baselines3 policies.

This module provides feature extractors that combine CNN-based price pattern
recognition with MLP processing of technical indicators and portfolio state.
"""

from typing import Dict, Literal, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tradebox.models.trading_cnn import (
    MultiScalePriceCNN,
    ResidualPriceCNN,
    SimplePriceCNN,
)


class HybridCNNExtractor(BaseFeaturesExtractor):
    """
    Hybrid feature extractor combining CNN (price) + MLP (indicators + portfolio).

    This extractor processes three types of observations:
    1. Price data (OHLCV): CNN for spatial-temporal pattern extraction
    2. Technical indicators: MLP for derived feature processing
    3. Portfolio state: MLP for position encoding

    All embeddings are concatenated and optionally fused through a final layer.

    Architecture:
        price (B, 5, T) → PriceCNN → price_emb (B, price_embed_dim)
        indicators (B, N) → IndicatorMLP → ind_emb (B, ind_embed_dim)
        portfolio (B, 4) → PortfolioMLP → port_emb (B, port_embed_dim)
        Concatenate → [price_emb, ind_emb, port_emb] → features (B, features_dim)

    Attributes:
        observation_space: Gymnasium Dict observation space
        features_dim: Output feature dimension (sum of all embeddings or fused)
        price_cnn: CNN for price data extraction
        indicator_mlp: MLP for processing technical indicators
        portfolio_mlp: MLP for encoding portfolio state
        fusion_layer: Optional layer to fuse all embeddings

    Example:
        >>> from gymnasium import spaces
        >>> obs_space = spaces.Dict({
        ...     "price": spaces.Box(0, np.inf, shape=(5, 60)),
        ...     "indicators": spaces.Box(-np.inf, np.inf, shape=(27,)),
        ...     "portfolio": spaces.Box(-np.inf, np.inf, shape=(4,))
        ... })
        >>> extractor = HybridCNNExtractor(obs_space, features_dim=256)
        >>> obs = {
        ...     "price": torch.randn(32, 5, 60),
        ...     "indicators": torch.randn(32, 27),
        ...     "portfolio": torch.randn(32, 4)
        ... }
        >>> features = extractor(obs)  # (32, 256)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        cnn_type: Literal["simple", "multiscale", "residual"] = "multiscale",
        price_embed_dim: int = 128,
        ind_embed_dim: int = 64,
        port_embed_dim: int = 32,
        use_fusion: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize hybrid CNN-MLP feature extractor.

        Args:
            observation_space: Dict observation space with keys:
                - "price": Box(shape=(5, lookback_window))
                - "indicators": Box(shape=(n_indicators,))
                - "portfolio": Box(shape=(4,))
            features_dim: Output feature dimension (passed to SB3 policy)
            cnn_type: Type of CNN to use ("simple", "multiscale", "residual")
            price_embed_dim: Embedding dimension for price CNN
            ind_embed_dim: Embedding dimension for indicator MLP
            port_embed_dim: Embedding dimension for portfolio MLP
            use_fusion: If True, add fusion layer after concatenation
            dropout: Dropout probability for regularization

        Raises:
            ValueError: If observation_space is invalid
        """
        # Validate observation space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                f"HybridCNNExtractor requires Dict observation space, "
                f"got {type(observation_space)}"
            )

        required_keys = {"price", "indicators", "portfolio"}
        if not required_keys.issubset(observation_space.spaces.keys()):
            raise ValueError(
                f"Observation space must contain {required_keys}, "
                f"got {set(observation_space.spaces.keys())}"
            )

        # Initialize BaseFeaturesExtractor
        super().__init__(observation_space, features_dim=features_dim)

        # Extract dimensions from observation space
        price_space = observation_space["price"]
        ind_space = observation_space["indicators"]
        port_space = observation_space["portfolio"]

        n_price_channels = price_space.shape[0]  # Typically 5 (OHLCV)
        n_indicators = ind_space.shape[0]
        n_portfolio_features = port_space.shape[0]  # Typically 4

        # Build price CNN based on type
        if cnn_type == "simple":
            self.price_cnn = SimplePriceCNN(
                input_channels=n_price_channels,
                embed_dim=price_embed_dim,
                dropout=dropout,
            )
        elif cnn_type == "multiscale":
            self.price_cnn = MultiScalePriceCNN(
                input_channels=n_price_channels,
                base_channels=32,
                embed_dim=price_embed_dim,
                dropout=dropout,
            )
        elif cnn_type == "residual":
            self.price_cnn = ResidualPriceCNN(
                input_channels=n_price_channels,
                hidden_channels=64,
                embed_dim=price_embed_dim,
                n_blocks=3,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Invalid cnn_type: {cnn_type}. "
                "Must be 'simple', 'multiscale', or 'residual'"
            )

        # Build indicator MLP
        self.indicator_mlp = nn.Sequential(
            nn.Linear(n_indicators, ind_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ind_embed_dim * 2, ind_embed_dim),
            nn.ReLU(),
        )

        # Build portfolio MLP
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(n_portfolio_features, port_embed_dim),
            nn.ReLU(),
        )

        # Calculate combined embedding dimension
        combined_dim = price_embed_dim + ind_embed_dim + port_embed_dim

        # Optional fusion layer
        if use_fusion:
            self.fusion_layer = nn.Sequential(
                nn.Linear(combined_dim, features_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            # No fusion, just pass through
            self.fusion_layer = nn.Identity()
            # Update features_dim to match combined_dim if no fusion
            self._features_dim = combined_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from dict observation.

        Args:
            observations: Dictionary with keys:
                - "price": Tensor of shape (batch, 5, lookback_window)
                - "indicators": Tensor of shape (batch, n_indicators)
                - "portfolio": Tensor of shape (batch, 4)

        Returns:
            Feature tensor of shape (batch, features_dim)

        Raises:
            KeyError: If required observation keys are missing
        """
        # Extract each component
        price = observations["price"]  # (B, 5, T)
        indicators = observations["indicators"]  # (B, N_ind)
        portfolio = observations["portfolio"]  # (B, 4)

        # Pass through respective networks
        price_emb = self.price_cnn(price)  # (B, price_embed_dim)
        ind_emb = self.indicator_mlp(indicators)  # (B, ind_embed_dim)
        port_emb = self.portfolio_mlp(portfolio)  # (B, port_embed_dim)

        # Concatenate all embeddings
        combined = torch.cat([price_emb, ind_emb, port_emb], dim=1)

        # Optional fusion
        features = self.fusion_layer(combined)

        return features


class CNNOnlyExtractor(BaseFeaturesExtractor):
    """
    CNN-only feature extractor (price data only, no indicators).

    Useful for ablation studies or when you want the agent to learn
    purely from raw price patterns without hand-crafted indicators.

    Attributes:
        observation_space: Gymnasium Dict observation space with "price" key
        features_dim: Output feature dimension
        price_cnn: CNN for price data extraction

    Example:
        >>> obs_space = spaces.Dict({
        ...     "price": spaces.Box(0, np.inf, shape=(5, 60)),
        ... })
        >>> extractor = CNNOnlyExtractor(obs_space, features_dim=128)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        cnn_type: Literal["simple", "multiscale", "residual"] = "multiscale",
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize CNN-only feature extractor.

        Args:
            observation_space: Dict observation space with "price" key
            features_dim: Output feature dimension
            cnn_type: Type of CNN architecture
            dropout: Dropout probability
        """
        super().__init__(observation_space, features_dim=features_dim)

        price_space = observation_space["price"]
        n_price_channels = price_space.shape[0]

        # Build price CNN
        if cnn_type == "simple":
            self.price_cnn = SimplePriceCNN(
                input_channels=n_price_channels,
                embed_dim=features_dim,
                dropout=dropout,
            )
        elif cnn_type == "multiscale":
            self.price_cnn = MultiScalePriceCNN(
                input_channels=n_price_channels,
                embed_dim=features_dim,
                dropout=dropout,
            )
        elif cnn_type == "residual":
            self.price_cnn = ResidualPriceCNN(
                input_channels=n_price_channels,
                embed_dim=features_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Invalid cnn_type: {cnn_type}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from price data only."""
        price = observations["price"]
        return self.price_cnn(price)


class MLPOnlyExtractor(BaseFeaturesExtractor):
    """
    MLP-only feature extractor (indicators + portfolio, no price CNN).

    This is essentially the baseline approach (what you currently have).
    Useful for comparison with CNN-based extractors.

    Attributes:
        observation_space: Gymnasium Dict observation space
        features_dim: Output feature dimension
        indicator_mlp: MLP for indicators
        portfolio_mlp: MLP for portfolio state

    Example:
        >>> obs_space = spaces.Dict({
        ...     "indicators": spaces.Box(-np.inf, np.inf, shape=(27,)),
        ...     "portfolio": spaces.Box(-np.inf, np.inf, shape=(4,))
        ... })
        >>> extractor = MLPOnlyExtractor(obs_space, features_dim=128)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 128,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize MLP-only feature extractor.

        Args:
            observation_space: Dict observation space
            features_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions (default: [256, 128])
            dropout: Dropout probability
        """
        super().__init__(observation_space, features_dim=features_dim)

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Get total input dimension
        total_input_dim = 0
        for key, space in observation_space.spaces.items():
            if key == "price":
                # Flatten price if present
                total_input_dim += space.shape[0] * space.shape[1]
            else:
                total_input_dim += space.shape[0]

        # Build MLP
        layers = []
        in_dim = total_input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, features_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features using MLP on flattened observations."""
        # Flatten all observations and concatenate
        flattened = []
        for key in sorted(observations.keys()):
            obs = observations[key]
            if len(obs.shape) > 2:
                # Flatten spatial dimensions (e.g., price)
                obs = obs.reshape(obs.shape[0], -1)
            flattened.append(obs)

        combined = torch.cat(flattened, dim=1)
        return self.mlp(combined)
