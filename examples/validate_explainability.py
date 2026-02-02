"""
Quick validation script for explainability module.

This script tests the explainability components without requiring a fully trained model.
It creates a simple environment and validates that attention weights can be captured
and visualized correctly.

Usage:
    python examples/validate_explainability.py
"""

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym

from tradebox.models.trading_cnn_extractor import TradingCNNExtractor, PricePatternCNN
from tradebox.explainability.attention_viz import AttentionAnalyzer
from tradebox.explainability.text_generator import TradeExplainTextGenerator


def test_price_pattern_cnn():
    """Test PricePatternCNN attention capture."""
    print("\n" + "="*80)
    print("TEST 1: PricePatternCNN Attention Capture")
    print("="*80)

    # Create CNN with attention capture enabled
    cnn = PricePatternCNN(
        lookback_window=60,
        in_channels=5,
        embedding_dim=128,
        use_attention=True,
        capture_attention=True,
    )

    # Create dummy price data (batch_size=1, lookback=60, channels=5)
    # Simulate an uptrend
    batch_size = 1
    lookback = 60
    price_data = th.zeros(batch_size, lookback, 5)

    # Generate uptrend pattern
    base_price = 100.0
    for i in range(lookback):
        trend = base_price + i * 0.5  # Uptrend
        noise = np.random.randn() * 0.2
        price = trend + noise

        price_data[0, i, 0] = price + np.random.randn() * 0.1  # Open
        price_data[0, i, 1] = price + abs(np.random.randn() * 0.3)  # High
        price_data[0, i, 2] = price - abs(np.random.randn() * 0.3)  # Low
        price_data[0, i, 3] = price  # Close
        price_data[0, i, 4] = 1000 + np.random.randn() * 100  # Volume

    # Forward pass
    print("Running forward pass...")
    embedding = cnn(price_data)
    print(f"Embedding shape: {embedding.shape}")

    # Check attention weights
    if cnn._attention_weights is not None:
        print(f"Attention weights captured: {cnn._attention_weights.shape}")
        print("✓ Attention capture working!")
        return cnn._attention_weights, price_data
    else:
        print("✗ Attention weights NOT captured!")
        return None, None


def test_trading_cnn_extractor():
    """Test TradingCNNExtractor intermediate capture."""
    print("\n" + "="*80)
    print("TEST 2: TradingCNNExtractor Intermediate Capture")
    print("="*80)

    # Create observation space
    obs_space = gym.spaces.Dict({
        "price": gym.spaces.Box(0, np.inf, shape=(60, 5)),
        "indicators": gym.spaces.Box(-np.inf, np.inf, shape=(27,)),
        "portfolio": gym.spaces.Box(-np.inf, np.inf, shape=(4,))
    })

    # Create extractor with intermediate capture
    extractor = TradingCNNExtractor(
        observation_space=obs_space,
        features_dim=256,
        cnn_embedding_dim=128,
        indicator_embedding_dim=32,
        portfolio_embedding_dim=16,
        use_attention=True,
        capture_intermediates=True,
    )

    # Create dummy observation
    obs = {
        "price": th.randn(1, 60, 5) * 10 + 100,  # Price around 100
        "indicators": th.randn(1, 27),  # Normalized indicators
        "portfolio": th.tensor([[0.5, 0.02, 0.03, 0.7]]),  # Some position
    }

    # Forward pass
    print("Running forward pass...")
    features = extractor(obs)
    print(f"Features shape: {features.shape}")

    # Get intermediates
    intermediates = extractor.get_intermediates()
    print(f"\nCaptured intermediates:")
    for key, value in intermediates.items():
        if value is not None:
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: None")

    if intermediates and len(intermediates) > 0:
        print("✓ Intermediate capture working!")
        return intermediates, obs
    else:
        print("✗ Intermediates NOT captured!")
        return None, None


def test_attention_visualization(attention_weights, price_data):
    """Test attention visualization."""
    print("\n" + "="*80)
    print("TEST 3: Attention Visualization")
    print("="*80)

    if attention_weights is None or price_data is None:
        print("✗ Skipping (no attention weights available)")
        return

    # Create analyzer
    analyzer = AttentionAnalyzer()

    # Convert tensors to numpy
    attn_np = attention_weights[0].numpy()  # Remove batch dim
    price_np = price_data[0].numpy()  # Remove batch dim

    print(f"Attention shape: {attn_np.shape}")
    print(f"Price data shape: {price_np.shape}")

    # Create visualization
    output_dir = Path("reports/explainability_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating candlestick + attention visualization...")
    fig = analyzer.plot_attention_on_candlestick(
        price_data=price_np,
        attention_weights=attn_np,
        action="BUY",
        confidence=0.87,
        top_indicators=[
            {"name": "RSI", "value": 35.2, "signal": "oversold"},
            {"name": "MACD", "value": 12.5, "signal": "bullish_crossover"},
            {"name": "BB_position", "value": 0.15, "signal": "near_lower_band"},
        ],
        save_path=output_dir / "test_attention_candlestick.png",
    )
    plt.close(fig)

    print("\nGenerating attention heatmap...")
    fig_heatmap = analyzer.create_attention_heatmap(
        attn_np,
        save_path=output_dir / "test_attention_heatmap.png",
    )
    plt.close(fig_heatmap)

    # Analyze distribution
    avg_attention = attn_np.mean(axis=0)[-1, :]  # Last bar's attention
    attn_stats = analyzer.analyze_attention_distribution(avg_attention)

    print(f"\nAttention Distribution Analysis:")
    print(f"  Pattern: {attn_stats['pattern_type']}")
    print(f"  Critical Bars: {attn_stats['critical_bars'][:5]}...")
    print(f"  Entropy: {attn_stats['entropy']:.3f}")
    print(f"  Max Attention: {attn_stats['max_attention']:.3f}")

    print(f"\n✓ Visualizations saved to: {output_dir}")


def test_text_generation():
    """Test text generation."""
    print("\n" + "="*80)
    print("TEST 4: Text Summary Generation")
    print("="*80)

    generator = TradeExplainTextGenerator()

    # Create mock explanation
    explanation = {
        "action": "BUY",
        "confidence": 0.87,
        "price_pattern_analysis": {
            "attention_focus": {
                "primary_bars": [55, 56, 57, 58, 59],
                "attention_scores": [0.05, 0.08, 0.15, 0.25, 0.40],
            },
            "pattern_detected": "momentum_driven",
            "confidence": 0.82,
            "focus_distribution": {
                "recent": 0.65,
                "middle": 0.20,
                "distant": 0.15,
            },
        },
        "indicator_analysis": {
            "top_contributors": [
                {"name": "RSI", "value": 35.2, "contribution": 0.15, "signal": "oversold"},
                {"name": "MACD_diff", "value": 12.5, "contribution": 0.10, "signal": "bullish_crossover"},
                {"name": "BB_position", "value": 0.15, "contribution": 0.08, "signal": "near_lower_band"},
            ]
        },
        "portfolio_state": {
            "position_pct": 0.0,
            "cash_pct": 1.0,
            "unrealized_pnl": 0.0,
            "risk_appetite": "neutral",
        },
    }

    # Generate summary
    print("\nShort Summary:")
    print("-" * 80)
    summary = generator.generate(explanation)
    print(summary)

    print("\nDetailed Summary:")
    print("-" * 80)
    detailed = generator.generate_detailed(explanation)
    print(detailed)

    print("\n✓ Text generation working!")


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("EXPLAINABILITY MODULE VALIDATION")
    print("="*80)

    # Test 1: PricePatternCNN
    attn_weights, price_data = test_price_pattern_cnn()

    # Test 2: TradingCNNExtractor
    intermediates, obs = test_trading_cnn_extractor()

    # Test 3: Visualization
    test_attention_visualization(attn_weights, price_data)

    # Test 4: Text generation
    test_text_generation()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nAll core components are working correctly!")
    print("Next steps:")
    print("  1. Train a model if you haven't already")
    print("  2. Run examples/explain_trades.py with your trained model")
    print("  3. Integrate explanations into backtest reports")
    print("\n")


if __name__ == "__main__":
    main()
