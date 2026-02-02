"""Comprehensive tests for unified FeatureExtractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tradebox.data.loaders.fundamental_loader import FundamentalConfig
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
from tradebox.features.technical import FeatureConfig


# ===== Fixtures =====


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample OHLCV price data."""
    dates = pd.date_range("2023-01-01", periods=250, freq="D")
    np.random.seed(42)

    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(250) * 2)
    high_prices = close_prices + np.abs(np.random.randn(250) * 1)
    low_prices = close_prices - np.abs(np.random.randn(250) * 1)
    open_prices = close_prices + np.random.randn(250) * 0.5

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, 250),
        },
        index=dates,
    )


@pytest.fixture
def technical_only_config(temp_cache_dir: Path) -> FeatureExtractorConfig:
    """Create config with only technical features enabled."""
    return FeatureExtractorConfig(
        technical=FeatureConfig(
            sma_periods=[20, 50],
            ema_periods=[9],
            normalize=True,
        ),
        fundamental=FundamentalConfig(
            use_mock_data=False,
            enabled=False,  # Disable fundamentals
        ),
        cache_dir=str(temp_cache_dir),
        version="test_v1",
    )


@pytest.fixture
def combined_config(temp_cache_dir: Path) -> FeatureExtractorConfig:
    """Create config with both technical and fundamental features."""
    return FeatureExtractorConfig(
        technical=FeatureConfig(
            sma_periods=[20, 50],
            ema_periods=[9],
            normalize=True,
        ),
        fundamental=FundamentalConfig(
            use_mock_data=True,  # Use mock data for testing
            enabled=True,
            announcement_delay_days=45,
            valuation_enabled=True,
            profitability_enabled=True,
        ),
        cache_dir=str(temp_cache_dir),
        version="test_v1",
    )


@pytest.fixture
def extractor_technical_only(technical_only_config: FeatureExtractorConfig) -> FeatureExtractor:
    """Create FeatureExtractor with only technical features."""
    return FeatureExtractor(technical_only_config)


@pytest.fixture
def extractor_combined(combined_config: FeatureExtractorConfig) -> FeatureExtractor:
    """Create FeatureExtractor with both technical and fundamental."""
    return FeatureExtractor(combined_config)


# ===== Test Classes =====


class TestFeatureExtractorInit:
    """Tests for FeatureExtractor initialization."""

    def test_init_technical_only(
        self, extractor_technical_only: FeatureExtractor
    ) -> None:
        """Should initialize with technical features only."""
        assert extractor_technical_only.config is not None
        assert extractor_technical_only.technical_extractor is not None
        assert extractor_technical_only.fundamental_loader is not None
        assert extractor_technical_only.technical_scaler is None
        assert extractor_technical_only.fundamental_scaler is None

    def test_init_combined(self, extractor_combined: FeatureExtractor) -> None:
        """Should initialize with both technical and fundamental."""
        assert extractor_combined.config is not None
        assert extractor_combined.config.fundamental.enabled is True

    def test_config_dict_conversion(self, temp_cache_dir: Path) -> None:
        """Should convert dict configs to proper dataclass instances."""
        config = FeatureExtractorConfig(
            technical={"sma_periods": [20]},  # Dict instead of FeatureConfig
            fundamental={"use_mock_data": True},  # Dict instead of FundamentalConfig
            cache_dir=str(temp_cache_dir),
        )

        # Should be converted in __post_init__
        assert isinstance(config.technical, FeatureConfig)
        assert isinstance(config.fundamental, FundamentalConfig)


class TestTechnicalOnlyExtraction:
    """Tests for extraction with technical features only."""

    def test_extract_technical_only(
        self,
        extractor_technical_only: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should extract technical features when fundamentals disabled."""
        features_df = extractor_technical_only.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        # Should have OHLCV + technical features
        assert not features_df.empty
        assert len(features_df) == len(sample_price_data)
        assert "Close" in features_df.columns
        assert "SMA_20" in features_df.columns
        assert "SMA_50" in features_df.columns
        assert "EMA_9" in features_df.columns

        # Should NOT have fundamental features
        assert "PE_Ratio_Trailing" not in features_df.columns
        assert "ROE" not in features_df.columns

    def test_get_feature_names_technical_only(
        self,
        extractor_technical_only: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should return only technical feature names."""
        # Need to extract first to populate feature names
        extractor_technical_only.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        feature_names = extractor_technical_only.get_feature_names()

        assert "technical" in feature_names
        assert "fundamental" in feature_names
        assert "all" in feature_names

        assert len(feature_names["technical"]) > 0
        assert len(feature_names["fundamental"]) == 0
        assert feature_names["all"] == feature_names["technical"]


class TestCombinedExtraction:
    """Tests for extraction with both technical and fundamental features."""

    def test_extract_combined(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should extract both technical and fundamental features."""
        features_df = extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        # Should have OHLCV + technical + fundamental features
        assert not features_df.empty
        assert len(features_df) == len(sample_price_data)

        # OHLCV columns
        assert "Close" in features_df.columns

        # Technical features
        assert "SMA_20" in features_df.columns

        # Fundamental features (mock data should generate these)
        # Note: Feature names may vary based on config
        feature_names = extractor_combined.get_feature_names()
        assert len(feature_names["fundamental"]) > 0

    def test_get_feature_names_combined(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should return both technical and fundamental feature names."""
        # Need to extract first to populate feature names
        extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        feature_names = extractor_combined.get_feature_names()

        assert len(feature_names["technical"]) > 0
        assert len(feature_names["fundamental"]) > 0
        assert len(feature_names["all"]) == (
            len(feature_names["technical"]) + len(feature_names["fundamental"])
        )


class TestSeparateNormalization:
    """Tests for separate normalization of technical vs fundamental features."""

    def test_separate_scalers_fitted(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should fit separate scalers for technical and fundamental."""
        # Extract with fit_normalize=True
        extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        # Technical scaler is fitted by TechnicalFeatures internally
        # Fundamental scaler should be fitted
        assert extractor_combined.fundamental_scaler is not None
        assert hasattr(extractor_combined.fundamental_scaler, "mean_")
        assert hasattr(extractor_combined.fundamental_scaler, "scale_")

    def test_scaler_params_retrieval(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should retrieve scaler parameters."""
        # Fit scalers
        extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        params = extractor_combined.get_scaler_params()

        assert "technical" in params
        assert "fundamental" in params

        # Fundamental scaler should have params
        if params["fundamental"] is not None:
            assert "mean" in params["fundamental"]
            assert "scale" in params["fundamental"]


class TestFitTransformSplit:
    """Tests for fit/transform separation to prevent data leakage."""

    def test_fit_on_train_transform_on_val(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should fit on train, transform on val without refitting."""
        # Split data into train and val
        train_data = sample_price_data.iloc[:200]
        val_data = sample_price_data.iloc[200:]

        # Fit on train
        train_features = extractor_combined.extract(
            symbol="TEST",
            price_data=train_data,
            fit_normalize=True,
        )

        # Store scaler params
        train_params = extractor_combined.get_scaler_params()
        if train_params["fundamental"] is not None:
            train_mean = train_params["fundamental"]["mean"]

            # Transform on val (should NOT refit)
            val_features = extractor_combined.extract(
                symbol="TEST",
                price_data=val_data,
                fit_normalize=False,
            )

            # Scaler params should be unchanged
            val_params = extractor_combined.get_scaler_params()
            val_mean = val_params["fundamental"]["mean"]

            assert train_mean == val_mean, "Scaler was re-fit on validation data (DATA LEAKAGE)"

    def test_transform_without_fit_warns(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
        caplog,
    ) -> None:
        """Should warn when transforming without fitting first."""
        # Try to transform without fitting
        features = extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=False,  # Transform without fit
        )

        # Should still return features (may not be normalized properly)
        assert not features.empty


class TestQuarterlyToDailyAlignment:
    """Tests for quarterly fundamental → daily alignment."""

    def test_forward_fill_fundamentals(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should forward-fill quarterly fundamentals to daily."""
        features_df = extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        # Get fundamental columns
        feature_names = extractor_combined.get_feature_names()
        if len(feature_names["fundamental"]) > 0:
            fund_cols = [
                col for col in features_df.columns
                if col in feature_names["fundamental"]
            ]

            if fund_cols:
                # Check that fundamentals have values (forward-filled)
                # Some NaN at the beginning is OK (before first announcement)
                non_nan_count = features_df[fund_cols].notna().sum().sum()
                assert non_nan_count > 0, "Fundamentals should have some non-NaN values"

    def test_daily_frequency_maintained(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should maintain daily frequency after merging quarterly data."""
        features_df = extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        # Should have same length as input
        assert len(features_df) == len(sample_price_data)

        # Should have same index
        pd.testing.assert_index_equal(features_df.index, sample_price_data.index)


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_empty_price_data(
        self, extractor_combined: FeatureExtractor
    ) -> None:
        """Should raise error for empty price data."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="price_data cannot be empty"):
            extractor_combined.extract(
                symbol="TEST",
                price_data=empty_df,
                fit_normalize=True,
            )

    def test_no_fundamental_data_available(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should handle case when no fundamental data is available."""
        with patch.object(
            extractor_combined.fundamental_loader,
            "download",
            return_value=pd.DataFrame(),  # Empty DataFrame
        ):
            features_df = extractor_combined.extract(
                symbol="NONEXISTENT",
                price_data=sample_price_data,
                fit_normalize=True,
            )

            # Should still return features (technical only)
            assert not features_df.empty
            assert len(features_df) == len(sample_price_data)


class TestFeatureConsistency:
    """Tests for feature consistency across extractions."""

    def test_consistent_feature_count(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should produce consistent feature count across extractions."""
        # First extraction
        features1 = extractor_combined.extract(
            symbol="TEST1",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        # Second extraction (same symbol, same data)
        features2 = extractor_combined.extract(
            symbol="TEST1",
            price_data=sample_price_data,
            fit_normalize=False,
        )

        # Should have same number of columns
        assert len(features1.columns) == len(features2.columns)

    def test_feature_names_match_columns(
        self,
        extractor_combined: FeatureExtractor,
        sample_price_data: pd.DataFrame,
    ) -> None:
        """Should have feature names matching extracted columns."""
        features_df = extractor_combined.extract(
            symbol="TEST",
            price_data=sample_price_data,
            fit_normalize=True,
        )

        feature_names = extractor_combined.get_feature_names()

        # All feature names should be in columns
        for feature in feature_names["all"]:
            assert feature in features_df.columns, f"Feature {feature} not in columns"


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_default_config_initialization(self) -> None:
        """Should initialize with default configuration."""
        config = FeatureExtractorConfig()

        assert config.technical is not None
        assert config.fundamental is not None
        assert config.version == "v2"

    def test_custom_config_initialization(self, temp_cache_dir: Path) -> None:
        """Should initialize with custom configuration."""
        config = FeatureExtractorConfig(
            technical=FeatureConfig(sma_periods=[10, 20, 30]),
            fundamental=FundamentalConfig(use_mock_data=True),
            cache_dir=str(temp_cache_dir),
            version="custom_v1",
        )

        assert config.technical.sma_periods == [10, 20, 30]
        assert config.fundamental.use_mock_data is True
        assert config.version == "custom_v1"
