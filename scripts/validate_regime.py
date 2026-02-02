"""Validation script for regime detection.

This script validates the regime detection implementation by:
1. Downloading sample market data
2. Extracting features including regime detection
3. Visualizing regime states and indicators
4. Printing regime distribution statistics

Usage:
    python scripts/validate_regime.py
    python scripts/validate_regime.py --symbol RELIANCE.NS
    python scripts/validate_regime.py --symbol ^NSEI --start 2020-01-01 --end 2024-12-31
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
from tradebox.features.technical import FeatureConfig
from tradebox.features.regime import RegimeConfig


def validate_regime_detection(
    symbol: str = "^NSEI",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    output_path: str = "regime_validation.png"
):
    """
    Visual validation of regime detection.

    Args:
        symbol: Stock symbol to analyze
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Path to save validation plot
    """
    logger.info(f"Validating regime detection for {symbol} ({start_date} to {end_date})")

    # Load data
    logger.info("Loading market data...")
    loader = YahooDataLoader(cache_dir=Path("data/raw"), use_cache=True)
    data = loader.download(symbol, start_date, end_date)

    if data.empty:
        logger.error(f"No data downloaded for {symbol}")
        return

    logger.info(f"Loaded {len(data)} bars of data")

    # Extract features with regime detection
    logger.info("Extracting features with regime detection...")
    from tradebox.data.loaders.fundamental_loader import FundamentalConfig
    config = FeatureExtractorConfig(
        technical=FeatureConfig(adx_enabled=True, adx_period=14),
        regime=RegimeConfig(
            trending_threshold=25.0,
            ranging_threshold=20.0,
            use_directional_bias=True
        ),
        fundamental=FundamentalConfig(enabled=False)  # Disable fundamentals for validation
    )
    extractor = FeatureExtractor(config)
    features = extractor.extract(symbol, data, fit_normalize=False)  # No normalization for visualization

    logger.info(f"Extracted {len(features.columns)} features")

    # Verify regime features exist
    regime_features = ['ADX', 'Plus_DI', 'Minus_DI', 'regime_state', 'regime_strength', 'trend_bias', 'regime_persistence']
    missing_features = [f for f in regime_features if f not in features.columns]
    if missing_features:
        logger.error(f"Missing regime features: {missing_features}")
        return

    logger.info("All regime features successfully extracted")

    # Create visualization
    logger.info("Creating visualization...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    # Plot 1: Price with regime background
    ax1 = axes[0]
    ax1.plot(features.index, features['Close'], label='Close Price', color='black', linewidth=1)

    # Color background by regime
    regime_colors = {0: 'red', 1: 'yellow', 2: 'green'}
    regime_labels = {0: 'Ranging', 1: 'Transition', 2: 'Trending'}

    for regime, color in regime_colors.items():
        mask = features['regime_state'] == regime
        ax1.fill_between(
            features.index,
            features['Close'].min(),
            features['Close'].max(),
            where=mask,
            alpha=0.15,
            color=color,
            label=regime_labels[regime]
        )

    ax1.set_title(f'{symbol} Price with Regime Zones', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: ADX with thresholds
    ax2 = axes[1]
    ax2.plot(features.index, features['ADX'], label='ADX', color='blue', linewidth=1.5)
    ax2.axhline(20, color='red', linestyle='--', linewidth=1, label='Ranging Threshold (20)')
    ax2.axhline(25, color='green', linestyle='--', linewidth=1, label='Trending Threshold (25)')
    ax2.fill_between(features.index, 0, 20, alpha=0.1, color='red')
    ax2.fill_between(features.index, 25, 100, alpha=0.1, color='green')
    ax2.set_title('ADX Indicator (Trend Strength)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ADX Value', fontsize=12)
    ax2.set_ylim(0, max(features['ADX'].max() * 1.1, 50))
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: +DI/-DI
    ax3 = axes[2]
    ax3.plot(features.index, features['Plus_DI'], label='+DI (Uptrend)', color='green', linewidth=1.5)
    ax3.plot(features.index, features['Minus_DI'], label='-DI (Downtrend)', color='red', linewidth=1.5)
    ax3.set_title('Directional Indicators (Trend Direction)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('DI Value', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Regime state timeline
    ax4 = axes[3]
    ax4.plot(features.index, features['regime_state'], linewidth=2, color='purple')
    ax4.fill_between(features.index, 0, features['regime_state'], alpha=0.3, color='purple')
    ax4.set_title('Regime State Timeline', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Regime State', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Ranging\n(ADX<20)', 'Transition\n(20≤ADX<25)', 'Trending\n(ADX≥25)'])
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved regime validation plot to {output_path}")

    # Print statistics
    print("\n" + "="*60)
    print("REGIME DETECTION VALIDATION RESULTS")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total bars: {len(features)}")
    print("\n" + "-"*60)
    print("REGIME DISTRIBUTION:")
    print("-"*60)

    total = len(features)
    ranging_count = (features['regime_state'] == 0).sum()
    transition_count = (features['regime_state'] == 1).sum()
    trending_count = (features['regime_state'] == 2).sum()

    print(f"  Ranging    (ADX < 20):  {ranging_count:5d} bars ({ranging_count/total*100:5.1f}%)")
    print(f"  Transition (20≤ADX<25): {transition_count:5d} bars ({transition_count/total*100:5.1f}%)")
    print(f"  Trending   (ADX ≥ 25):  {trending_count:5d} bars ({trending_count/total*100:5.1f}%)")

    print("\n" + "-"*60)
    print("REGIME STATISTICS:")
    print("-"*60)
    print(f"  Average regime duration: {features['regime_persistence'].mean():.1f} bars")
    print(f"  Maximum regime duration: {features['regime_persistence'].max():.0f} bars")
    print(f"  Average ADX: {features['ADX'].mean():.2f}")
    print(f"  Maximum ADX: {features['ADX'].max():.2f}")

    print("\n" + "-"*60)
    print("TREND BIAS DISTRIBUTION:")
    print("-"*60)

    uptrend_count = (features['trend_bias'] == 1).sum()
    neutral_count = (features['trend_bias'] == 0).sum()
    downtrend_count = (features['trend_bias'] == -1).sum()

    print(f"  Uptrend   (+DI > -DI): {uptrend_count:5d} bars ({uptrend_count/total*100:5.1f}%)")
    print(f"  Neutral   (mixed):     {neutral_count:5d} bars ({neutral_count/total*100:5.1f}%)")
    print(f"  Downtrend (-DI > +DI): {downtrend_count:5d} bars ({downtrend_count/total*100:5.1f}%)")

    print("\n" + "-"*60)
    print("FEATURE VERIFICATION:")
    print("-"*60)
    feature_names = extractor.get_feature_names()
    print(f"  Technical features: {len(feature_names['technical'])}")
    print(f"  Regime features:    {len(feature_names['regime'])}")
    print(f"  Fundamental features: {len(feature_names['fundamental'])}")
    print(f"  Total features:     {len(feature_names['all'])}")

    print("\n" + "="*60)
    print("✓ Regime detection validation completed successfully!")
    print(f"✓ Visualization saved to: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate regime detection implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_regime.py
  python scripts/validate_regime.py --symbol RELIANCE.NS
  python scripts/validate_regime.py --symbol ^NSEI --start 2020-01-01 --end 2024-12-31
  python scripts/validate_regime.py --symbol TCS.NS --output tcs_regime.png
        """
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="^NSEI",
        help="Stock symbol to analyze (default: ^NSEI)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date YYYY-MM-DD (default: 2020-01-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date YYYY-MM-DD (default: 2024-12-31)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="regime_validation.png",
        help="Output path for validation plot (default: regime_validation.png)"
    )

    args = parser.parse_args()

    validate_regime_detection(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        output_path=args.output
    )
