"""
Neural network models and feature extractors for RL agents.

This module provides CNN architectures and feature extractors for trading RL agents,
enabling pattern recognition from raw OHLCV price data.
"""

from tradebox.models.trading_cnn import (
    MultiScalePriceCNN,
    ResidualPriceCNN,
    SimplePriceCNN,
)
from tradebox.models.hybrid_extractor import (
    HybridCNNExtractor,
    CNNOnlyExtractor,
    MLPOnlyExtractor,
)
from tradebox.models.trading_cnn_extractor import (
    TradingCNNExtractor,
    PricePatternCNN,
    IndicatorContextEncoder,
    PortfolioStateEncoder,
)

__all__ = [
    # CNN architectures
    "SimplePriceCNN",
    "MultiScalePriceCNN",
    "ResidualPriceCNN",
    # Feature extractors
    "HybridCNNExtractor",
    "CNNOnlyExtractor",
    "MLPOnlyExtractor",
    # Production CNN extractor (recommended)
    "TradingCNNExtractor",
    "PricePatternCNN",
    "IndicatorContextEncoder",
    "PortfolioStateEncoder",
]
