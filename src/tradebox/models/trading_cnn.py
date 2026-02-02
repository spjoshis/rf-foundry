"""
Enhanced CNN architecture for trading pattern recognition.

This module provides multi-scale CNN architectures for extracting spatial-temporal
patterns from raw OHLCV price data. Designed specifically for trading RL agents.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class MultiScalePriceCNN(nn.Module):
    """
    Multi-scale 1D CNN for extracting patterns from OHLCV price data.

    Uses parallel convolution paths with different kernel sizes to capture
    patterns at multiple time scales (short-term spikes, medium-term trends,
    long-term momentum).

    Architecture:
        - Path 1: Small kernels (3) - Local candle patterns (1-3 bars)
        - Path 2: Medium kernels (7) - Short-term trends (1-2 weeks EOD / 30-60 min intraday)
        - Path 3: Large kernels (15) - Medium-term patterns (2-4 weeks EOD / 1-2 hours intraday)
        - Concatenate all paths and project to embedding

    Attributes:
        input_channels: Number of input channels (typically 5 for OHLCV)
        base_channels: Base number of convolutional channels (default: 32)
        embed_dim: Output embedding dimension (default: 128)
        dropout: Dropout probability (default: 0.1)
        use_batch_norm: Whether to use batch normalization (default: True)

    Example:
        >>> cnn = MultiScalePriceCNN(input_channels=5, embed_dim=128)
        >>> x = torch.randn(32, 5, 60)  # (batch, channels, time)
        >>> embedding = cnn(x)  # (32, 128)
    """

    def __init__(
        self,
        input_channels: int = 5,
        base_channels: int = 32,
        embed_dim: int = 128,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ) -> None:
        """
        Initialize multi-scale CNN.

        Args:
            input_channels: Number of input channels (5 for OHLCV)
            base_channels: Number of filters in first conv layer
            embed_dim: Output embedding dimension
            dropout: Dropout probability for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels
        self.embed_dim = embed_dim

        # Short-term path (kernel=3, small receptive field)
        self.short_path = self._build_conv_path(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=3,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Medium-term path (kernel=7, medium receptive field)
        self.medium_path = self._build_conv_path(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=7,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Long-term path (kernel=15, large receptive field)
        self.long_path = self._build_conv_path(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=15,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Combine all paths (3 × base_channels × 2 layers)
        combined_channels = base_channels * 2 * 3
        self.projection = nn.Sequential(
            nn.Linear(combined_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def _build_conv_path(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_batch_norm: bool,
        dropout: float,
    ) -> nn.Sequential:
        """
        Build a convolutional path with 2 conv layers.

        Args:
            in_channels: Input channels
            out_channels: Output channels for first conv
            kernel_size: Convolution kernel size
            use_batch_norm: Whether to use batch norm
            dropout: Dropout probability

        Returns:
            Sequential module for the conv path
        """
        padding = kernel_size // 2  # Same padding

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels * 2, kernel_size, padding=padding),
        ])

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels * 2))

        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale CNN.

        Args:
            x: Input tensor of shape (batch, channels, time)
               For OHLCV: (batch, 5, lookback_window)

        Returns:
            Embedding tensor of shape (batch, embed_dim)
        """
        # Apply each path in parallel
        short = self.short_path(x)  # (B, base_channels*2, T)
        medium = self.medium_path(x)  # (B, base_channels*2, T)
        long = self.long_path(x)  # (B, base_channels*2, T)

        # Global pooling
        short = self.global_pool(short).squeeze(-1)  # (B, base_channels*2)
        medium = self.global_pool(medium).squeeze(-1)  # (B, base_channels*2)
        long = self.global_pool(long).squeeze(-1)  # (B, base_channels*2)

        # Concatenate all scales
        combined = torch.cat([short, medium, long], dim=1)  # (B, base_channels*2*3)

        # Project to embedding dimension
        embedding = self.projection(combined)  # (B, embed_dim)

        return embedding


class ResidualPriceCNN(nn.Module):
    """
    Residual CNN for deeper feature extraction from price data.

    Uses residual connections to enable training deeper networks without
    vanishing gradients. Better for complex pattern recognition.

    Architecture:
        Input → Conv Block 1 (+ residual) → Conv Block 2 (+ residual) →
        Global Pool → Projection → Embedding

    Attributes:
        input_channels: Number of input channels (typically 5 for OHLCV)
        hidden_channels: Number of channels in hidden layers (default: 64)
        embed_dim: Output embedding dimension (default: 128)
        n_blocks: Number of residual blocks (default: 3)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> cnn = ResidualPriceCNN(input_channels=5, hidden_channels=64)
        >>> x = torch.randn(32, 5, 60)
        >>> embedding = cnn(x)  # (32, 128)
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_channels: int = 64,
        embed_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize residual CNN.

        Args:
            input_channels: Number of input channels (5 for OHLCV)
            hidden_channels: Number of filters in conv layers
            embed_dim: Output embedding dimension
            n_blocks: Number of residual blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim

        # Initial projection to hidden channels
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual CNN.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Embedding tensor of shape (batch, embed_dim)
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x).squeeze(-1)
        x = self.projection(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Implements: output = F(x) + x, where F is a 2-layer conv block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize residual block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection projection if dimensions don't match
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # Residual connection
        out = self.relu(out)

        return out


class SimplePriceCNN(nn.Module):
    """
    Simple 3-layer CNN for price data (lightweight baseline).

    A streamlined architecture for faster training and inference.
    Good starting point before trying more complex architectures.

    Attributes:
        input_channels: Number of input channels (typically 5 for OHLCV)
        hidden_channels: List of channel sizes for each layer
        embed_dim: Output embedding dimension (default: 64)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> cnn = SimplePriceCNN(input_channels=5, embed_dim=64)
        >>> x = torch.randn(32, 5, 60)
        >>> embedding = cnn(x)  # (32, 64)
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_channels: Optional[List[int]] = None,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize simple CNN.

        Args:
            input_channels: Number of input channels (5 for OHLCV)
            hidden_channels: Channel sizes for conv layers (default: [32, 64, 64])
            embed_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 64]

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim

        layers = []
        in_ch = input_channels

        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool1d(1))

        self.net = nn.Sequential(*layers)
        self.projection = nn.Linear(hidden_channels[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through simple CNN.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Embedding tensor of shape (batch, embed_dim)
        """
        x = self.net(x)  # (B, hidden_channels[-1], 1)
        x = x.squeeze(-1)  # (B, hidden_channels[-1])
        x = self.projection(x)  # (B, embed_dim)
        return x
