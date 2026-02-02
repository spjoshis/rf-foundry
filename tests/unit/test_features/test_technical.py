"""Unit tests for technical feature extraction."""

import pandas as pd
import pytest

from tradebox.features.technical import FeatureConfig, TechnicalFeatures


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Open": [100.0 + i * 0.1 for i in range(300)],
        "High": [105.0 + i * 0.1 for i in range(300)],
        "Low": [95.0 + i * 0.1 for i in range(300)],
        "Close": [102.0 + i * 0.1 for i in range(300)],
        "Volume": [1000000 + i * 1000 for i in range(300)],
    })


class TestFeatureConfig:
    """Tests for FeatureConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = FeatureConfig()

        assert config.version == "1.0"
        assert config.normalize is True
        assert config.trend_enabled is True
        assert config.sma_periods == [20, 50, 200]

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = FeatureConfig(
            normalize=False,
            sma_periods=[10, 20],
            momentum_enabled=False,
        )

        assert config.normalize is False
        assert config.sma_periods == [10, 20]
        assert config.momentum_enabled is False


class TestTechnicalFeatures:
    """Tests for TechnicalFeatures extraction."""

    def test_init(self) -> None:
        """Test initialization."""
        extractor = TechnicalFeatures()

        assert extractor.config is not None
        assert len(extractor.feature_names) == 0

    def test_extract_all_features(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test extracting all feature categories."""
        extractor = TechnicalFeatures()
        features = extractor.extract(sample_ohlcv)

        assert not features.empty
        assert len(features) == len(sample_ohlcv)
        assert len(extractor.get_feature_names()) > 20  # Should have 20-25 features

    def test_extract_validates_input(self) -> None:
        """Test that extract validates input columns."""
        extractor = TechnicalFeatures()
        invalid_df = pd.DataFrame({"Date": [1, 2, 3], "Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required columns"):
            extractor.extract(invalid_df)

    def test_trend_features(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test trend feature extraction."""
        config = FeatureConfig(
            trend_enabled=True,
            momentum_enabled=False,
            volatility_enabled=False,
            volume_enabled=False,
        )
        extractor = TechnicalFeatures(config)
        features = extractor.extract(sample_ohlcv)

        feature_names = extractor.get_feature_names()
        assert any("SMA" in name for name in feature_names)
        assert any("EMA" in name for name in feature_names)
        assert "MACD" in feature_names

    def test_momentum_features(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test momentum feature extraction."""
        config = FeatureConfig(
            trend_enabled=False,
            momentum_enabled=True,
            volatility_enabled=False,
            volume_enabled=False,
        )
        extractor = TechnicalFeatures(config)
        features = extractor.extract(sample_ohlcv)

        feature_names = extractor.get_feature_names()
        assert "RSI" in feature_names
        assert "Stoch_K" in feature_names
        assert "ROC" in feature_names

    def test_volatility_features(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test volatility feature extraction."""
        config = FeatureConfig(
            trend_enabled=False,
            momentum_enabled=False,
            volatility_enabled=True,
            volume_enabled=False,
        )
        extractor = TechnicalFeatures(config)
        features = extractor.extract(sample_ohlcv)

        feature_names = extractor.get_feature_names()
        assert "ATR" in feature_names
        assert "BB_Width" in feature_names
        assert "Returns_Std" in feature_names

    def test_volume_features(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test volume feature extraction."""
        config = FeatureConfig(
            trend_enabled=False,
            momentum_enabled=False,
            volatility_enabled=False,
            volume_enabled=True,
        )
        extractor = TechnicalFeatures(config)
        features = extractor.extract(sample_ohlcv)

        feature_names = extractor.get_feature_names()
        assert "Volume_MA" in feature_names
        assert "Volume_Ratio" in feature_names
        assert "OBV" in feature_names

    def test_normalization(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test feature normalization."""
        config = FeatureConfig(normalize=True)
        extractor = TechnicalFeatures(config)
        features = extractor.extract(sample_ohlcv)

        # Verify normalization was applied (check that normalization params were stored)
        assert len(extractor._normalization_params) > 0

        # Verify all extracted features are present
        for feature in extractor.get_feature_names():
            assert feature in features.columns

    def test_no_normalization(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test without normalization."""
        config = FeatureConfig(normalize=False)
        extractor = TechnicalFeatures(config)
        features = extractor.extract(sample_ohlcv)

        # Features should not be normalized
        assert "SMA_20" in features.columns
        assert features["SMA_20"].max() > 10  # Should be in original scale

    def test_get_feature_metadata(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test feature metadata."""
        extractor = TechnicalFeatures()
        features = extractor.extract(sample_ohlcv)

        metadata = extractor.get_feature_metadata()

        assert "version" in metadata
        assert "n_features" in metadata
        assert "feature_names" in metadata
        assert metadata["n_features"] == len(extractor.get_feature_names())

    def test_insufficient_data_warning(self) -> None:
        """Test warning for insufficient data."""
        # Create data with < 200 rows
        df = pd.DataFrame({
            "Open": [100.0] * 50,
            "High": [105.0] * 50,
            "Low": [95.0] * 50,
            "Close": [102.0] * 50,
            "Volume": [1000000] * 50,
        })

        extractor = TechnicalFeatures()
        # Should complete but log warning
        features = extractor.extract(df)
        assert not features.empty
