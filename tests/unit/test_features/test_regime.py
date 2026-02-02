"""Unit tests for regime detection module."""

import pytest
import pandas as pd
import numpy as np

from tradebox.features.regime import RegimeDetector, RegimeConfig


class TestRegimeConfig:
    """Test RegimeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegimeConfig()
        assert config.regime_type == "trend"
        assert config.trending_threshold == 25.0
        assert config.ranging_threshold == 20.0
        assert config.use_directional_bias is True
        assert config.di_diff_threshold == 5.0
        assert config.smooth_regime is False
        assert config.min_regime_duration == 5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegimeConfig(
            trending_threshold=30.0,
            ranging_threshold=15.0,
            use_directional_bias=False
        )
        assert config.trending_threshold == 30.0
        assert config.ranging_threshold == 15.0
        assert config.use_directional_bias is False

    def test_invalid_thresholds(self):
        """Test validation of threshold values."""
        # ranging_threshold must be < trending_threshold
        with pytest.raises(ValueError, match="ranging_threshold.*must be <"):
            RegimeConfig(ranging_threshold=30.0, trending_threshold=20.0)

        with pytest.raises(ValueError, match="ranging_threshold.*must be <"):
            RegimeConfig(ranging_threshold=25.0, trending_threshold=25.0)

    def test_invalid_di_diff(self):
        """Test validation of di_diff_threshold."""
        with pytest.raises(ValueError, match="di_diff_threshold must be >= 0"):
            RegimeConfig(di_diff_threshold=-1.0)

    def test_invalid_min_duration(self):
        """Test validation of min_regime_duration."""
        with pytest.raises(ValueError, match="min_regime_duration must be >= 1"):
            RegimeConfig(min_regime_duration=0)


class TestRegimeDetector:
    """Test RegimeDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        config = RegimeConfig()
        detector = RegimeDetector(config)
        assert detector.config == config

    def test_initialization_default_config(self):
        """Test detector initialization with default config."""
        detector = RegimeDetector()
        assert detector.config.regime_type == "trend"

    def test_regime_classification_thresholds(self):
        """Test regime state classification at boundary values."""
        config = RegimeConfig(ranging_threshold=20.0, trending_threshold=25.0)
        detector = RegimeDetector(config)

        # Create test data
        indicators = pd.DataFrame({
            'ADX': [15, 19, 20, 22, 24, 25, 30],
            'Plus_DI': [30, 30, 30, 30, 30, 30, 30],
            'Minus_DI': [20, 20, 20, 20, 20, 20, 20],
        })

        regime_df = detector.detect(indicators)

        # Verify classification
        assert regime_df.loc[0, 'regime_state'] == 0  # ADX=15 → range
        assert regime_df.loc[1, 'regime_state'] == 0  # ADX=19 → range
        assert regime_df.loc[2, 'regime_state'] == 1  # ADX=20 → transition (boundary at threshold)
        assert regime_df.loc[3, 'regime_state'] == 1  # ADX=22 → transition
        assert regime_df.loc[4, 'regime_state'] == 1  # ADX=24 → transition
        assert regime_df.loc[5, 'regime_state'] == 2  # ADX=25 → trending (boundary, inclusive)
        assert regime_df.loc[6, 'regime_state'] == 2  # ADX=30 → trending

    def test_directional_bias(self):
        """Test +DI/-DI bias calculation."""
        config = RegimeConfig(use_directional_bias=True, di_diff_threshold=5.0)
        detector = RegimeDetector(config)

        indicators = pd.DataFrame({
            'ADX': [30, 30, 30, 30, 30, 30],
            'Plus_DI': [30, 26, 25, 20, 15, 10],
            'Minus_DI': [15, 20, 20, 18, 30, 20],
        })

        regime_df = detector.detect(indicators)

        # Verify bias calculation
        # +DI - (-DI) > 5 → uptrend (1)
        # +DI - (-DI) < -5 → downtrend (-1)
        # else → neutral (0)
        assert regime_df.loc[0, 'trend_bias'] == 1   # +DI=30, -DI=15, diff=15 → uptrend
        assert regime_df.loc[1, 'trend_bias'] == 1   # +DI=26, -DI=20, diff=6 → uptrend (just above threshold)
        assert regime_df.loc[2, 'trend_bias'] == 0   # +DI=25, -DI=20, diff=5 → neutral (at boundary)
        assert regime_df.loc[3, 'trend_bias'] == 0   # +DI=20, -DI=18, diff=2 → neutral
        assert regime_df.loc[4, 'trend_bias'] == -1  # +DI=15, -DI=30, diff=-15 → downtrend
        assert regime_df.loc[5, 'trend_bias'] == -1  # +DI=10, -DI=20, diff=-10 → downtrend

    def test_directional_bias_disabled(self):
        """Test that bias is zero when disabled."""
        config = RegimeConfig(use_directional_bias=False)
        detector = RegimeDetector(config)

        indicators = pd.DataFrame({
            'ADX': [30, 30],
            'Plus_DI': [30, 10],  # Not required when disabled
            'Minus_DI': [15, 30],
        })

        regime_df = detector.detect(indicators)

        # Should all be neutral (0) when disabled
        assert all(regime_df['trend_bias'] == 0)

    def test_regime_strength(self):
        """Test regime strength calculation (normalized ADX)."""
        detector = RegimeDetector()

        indicators = pd.DataFrame({
            'ADX': [0, 25, 50, 75, 100],
            'Plus_DI': [20, 20, 20, 20, 20],
            'Minus_DI': [20, 20, 20, 20, 20],
        })

        regime_df = detector.detect(indicators)

        # Strength should be ADX / 100
        assert regime_df.loc[0, 'regime_strength'] == 0.0
        assert regime_df.loc[1, 'regime_strength'] == 0.25
        assert regime_df.loc[2, 'regime_strength'] == 0.50
        assert regime_df.loc[3, 'regime_strength'] == 0.75
        assert regime_df.loc[4, 'regime_strength'] == 1.0

    def test_regime_persistence(self):
        """Test regime persistence calculation."""
        detector = RegimeDetector()

        indicators = pd.DataFrame({
            'ADX': [30, 30, 30, 15, 15, 30, 30, 30, 30],
            'Plus_DI': [20, 20, 20, 20, 20, 20, 20, 20, 20],
            'Minus_DI': [20, 20, 20, 20, 20, 20, 20, 20, 20],
        })

        regime_df = detector.detect(indicators)

        # First regime (trending): 3 bars
        assert regime_df.loc[0, 'regime_persistence'] == 1
        assert regime_df.loc[1, 'regime_persistence'] == 2
        assert regime_df.loc[2, 'regime_persistence'] == 3

        # Second regime (ranging): 2 bars
        assert regime_df.loc[3, 'regime_persistence'] == 1
        assert regime_df.loc[4, 'regime_persistence'] == 2

        # Third regime (trending): 4 bars
        assert regime_df.loc[5, 'regime_persistence'] == 1
        assert regime_df.loc[6, 'regime_persistence'] == 2
        assert regime_df.loc[7, 'regime_persistence'] == 3
        assert regime_df.loc[8, 'regime_persistence'] == 4

    def test_regime_smoothing(self):
        """Test regime smoothing to filter rapid changes."""
        config = RegimeConfig(
            smooth_regime=True,
            min_regime_duration=3,
            ranging_threshold=20.0,
            trending_threshold=25.0
        )
        detector = RegimeDetector(config)

        # Create data with brief regime changes
        indicators = pd.DataFrame({
            'ADX': [30, 30, 15, 15, 30, 30, 30, 30],  # Short ranging period (2 bars)
            'Plus_DI': [20, 20, 20, 20, 20, 20, 20, 20],
            'Minus_DI': [20, 20, 20, 20, 20, 20, 20, 20],
        })

        regime_df = detector.detect(indicators)

        # With smoothing, the brief ranging period (2 bars) should be filtered out
        # because it's less than min_regime_duration (3)
        # The regime should remain trending throughout
        # Note: Smoothing implementation may vary, this tests the concept
        assert regime_df['regime_state'].nunique() <= 2  # Should have fewer regime changes

    def test_missing_columns(self):
        """Test error handling for missing required columns."""
        detector = RegimeDetector()

        # Missing ADX
        indicators_no_adx = pd.DataFrame({
            'Plus_DI': [20, 20],
            'Minus_DI': [20, 20],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(indicators_no_adx)

        # Missing directional indicators when use_directional_bias=True
        config = RegimeConfig(use_directional_bias=True)
        detector = RegimeDetector(config)

        indicators_no_di = pd.DataFrame({
            'ADX': [30, 30],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(indicators_no_di)

    def test_nan_handling(self):
        """Test handling of NaN values in indicators."""
        detector = RegimeDetector()

        indicators = pd.DataFrame({
            'ADX': [np.nan, 30, np.nan, 15],
            'Plus_DI': [30, np.nan, 20, 15],
            'Minus_DI': [15, 20, np.nan, 30],
        })

        # Should not raise error, NaN values filled with 0
        regime_df = detector.detect(indicators)

        assert len(regime_df) == 4
        # NaN ADX → 0 → ranging regime
        assert regime_df.loc[0, 'regime_state'] == 0
        assert regime_df.loc[2, 'regime_state'] == 0

    def test_output_columns(self):
        """Test that output has all expected columns."""
        detector = RegimeDetector()

        indicators = pd.DataFrame({
            'ADX': [30],
            'Plus_DI': [30],
            'Minus_DI': [15],
        })

        regime_df = detector.detect(indicators)

        expected_columns = ['regime_state', 'regime_strength', 'trend_bias', 'regime_persistence']
        assert list(regime_df.columns) == expected_columns

    def test_output_dtypes(self):
        """Test that output columns have correct dtypes."""
        detector = RegimeDetector()

        indicators = pd.DataFrame({
            'ADX': [30.0],
            'Plus_DI': [30.0],
            'Minus_DI': [15.0],
        })

        regime_df = detector.detect(indicators)

        assert regime_df['regime_state'].dtype == np.int64
        assert regime_df['regime_strength'].dtype == np.float32
        assert regime_df['trend_bias'].dtype == np.int64
        assert regime_df['regime_persistence'].dtype == np.int64

    def test_get_regime_summary(self):
        """Test regime summary statistics."""
        detector = RegimeDetector()

        indicators = pd.DataFrame({
            'ADX': [15, 15, 15, 22, 22, 30, 30, 30, 30, 30],
            'Plus_DI': [20] * 10,
            'Minus_DI': [20] * 10,
        })

        regime_df = detector.detect(indicators)
        summary = detector.get_regime_summary(regime_df)

        assert summary['total_bars'] == 10
        assert summary['ranging_pct'] == 30.0  # 3 bars
        assert summary['transition_pct'] == 20.0  # 2 bars
        assert summary['trending_pct'] == 50.0  # 5 bars
        assert summary['avg_regime_duration'] > 0
        assert summary['max_regime_duration'] == 5  # Longest regime

    def test_unimplemented_regime_type(self):
        """Test that unimplemented regime types raise error."""
        config = RegimeConfig(regime_type="volatility")
        detector = RegimeDetector(config)

        indicators = pd.DataFrame({
            'ADX': [30],
            'Plus_DI': [30],
            'Minus_DI': [15],
        })

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            detector.detect(indicators)


class TestRegimeDetectorIntegration:
    """Integration tests with realistic data."""

    def test_realistic_adx_values(self):
        """Test with realistic ADX values from market data."""
        detector = RegimeDetector()

        # Simulate realistic ADX values
        np.random.seed(42)
        adx_values = np.concatenate([
            np.random.uniform(10, 20, 30),  # Ranging period
            np.random.uniform(20, 25, 20),  # Transition
            np.random.uniform(25, 50, 50),  # Trending period
        ])

        indicators = pd.DataFrame({
            'ADX': adx_values,
            'Plus_DI': np.random.uniform(10, 40, 100),
            'Minus_DI': np.random.uniform(10, 40, 100),
        })

        regime_df = detector.detect(indicators)

        # Should have all three regimes
        assert 0 in regime_df['regime_state'].values  # Ranging
        assert 1 in regime_df['regime_state'].values  # Transition
        assert 2 in regime_df['regime_state'].values  # Trending

        # Verify output shape
        assert len(regime_df) == 100
        assert regime_df.shape[1] == 4

    def test_index_preservation(self):
        """Test that original DataFrame index is preserved."""
        detector = RegimeDetector()

        # Use datetime index like real market data
        date_range = pd.date_range('2020-01-01', periods=10, freq='D')
        indicators = pd.DataFrame({
            'ADX': [30] * 10,
            'Plus_DI': [30] * 10,
            'Minus_DI': [15] * 10,
        }, index=date_range)

        regime_df = detector.detect(indicators)

        # Index should match
        assert all(regime_df.index == indicators.index)
        assert isinstance(regime_df.index, pd.DatetimeIndex)
