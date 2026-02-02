"""
Unit tests for scripts/train_intraday.py.

Tests the intraday training script functions including:
- Configuration loading and validation
- Data loading and splitting
- Feature extraction
- Training pipeline
- Command-line argument parsing

Author: TradeBox-RL
Story: STORY-037
"""

import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
import yaml

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from train_intraday import (
    load_config,
    load_intraday_data,
    split_intraday_data,
    extract_intraday_features,
    parse_args,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_intraday_config() -> Dict:
    """
    Create sample intraday training configuration.

    Returns:
        Dictionary with experiment configuration
    """
    return {
        "experiment": {
            "name": "test_intraday",
            "description": "Test intraday configuration",
        },
        "data": {
            "type": "intraday",
            "symbols": ["RELIANCE.NS", "TCS.NS"],
            "interval": "5m",
            "period": "60d",
            "train_days": 40,
            "val_days": 10,
            "test_days": 10,
        },
        "features": {
            "version": "v2_intraday",
            "timeframe": "intraday",
            "technical": {
                "sma_periods": [10, 20, 50],
                "ema_periods": [9, 21],
                "rsi_period": 14,
                "atr_period": 14,
                "vwap_enabled": True,
                "session_high_low": True,
            },
        },
        "env": {
            "type": "intraday",
            "initial_capital": 100000.0,
            "lookback_window": 60,
            "bars_per_session": 75,
            "sessions_per_episode": 10,
            "force_close_eod": True,
        },
        "agent": {
            "algorithm": "PPO",
            "ppo": {
                "learning_rate": 0.0001,
                "n_steps": 4096,
                "batch_size": 128,
                "network_arch": [256, 256],
            },
        },
        "training": {
            "total_timesteps": 3000000,
            "n_envs": 16,
            "eval_freq": 25000,
            "checkpoint_freq": 100000,
            "seed": 42,
        },
    }


@pytest.fixture
def sample_intraday_data() -> pd.DataFrame:
    """
    Create sample 5-minute intraday data.

    Returns:
        DataFrame with OHLCV data for testing (1500 bars)
    """
    np.random.seed(42)
    n_bars = 1500  # ~20 trading days

    # Generate realistic price series
    base_price = 2500.0
    returns = np.random.normal(0.0001, 0.005, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV
    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        "High": prices * (1 + np.abs(np.random.uniform(0, 0.005, n_bars))),
        "Low": prices * (1 - np.abs(np.random.uniform(0, 0.005, n_bars))),
        "Close": prices,
        "Volume": np.random.randint(500000, 2000000, n_bars),
    })

    # Add datetime index (5-minute bars)
    start_date = pd.Timestamp("2024-01-01 09:15:00")
    df.index = pd.date_range(start=start_date, periods=n_bars, freq="5T")

    return df


@pytest.fixture
def temp_config_file(tmp_path, sample_intraday_config):
    """
    Create temporary YAML config file.

    Args:
        tmp_path: pytest temporary directory fixture
        sample_intraday_config: Sample config dictionary

    Returns:
        Path to temporary config file
    """
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_intraday_config, f)
    return config_file


# ============================================================================
# Test Configuration Loading
# ============================================================================


class TestConfigLoading:
    """Test suite for configuration loading functions."""

    def test_load_config_success(self, temp_config_file):
        """Test loading valid config file."""
        config = load_config(str(temp_config_file))

        assert "experiment" in config
        assert "data" in config
        assert "training" in config
        assert config["experiment"]["name"] == "test_intraday"

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises appropriate error."""
        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content:\n  - broken")

        with pytest.raises(yaml.YAMLError):
            load_config(str(invalid_file))

    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty config file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        config = load_config(str(empty_file))
        assert config is None or config == {}


# ============================================================================
# Test Data Loading
# ============================================================================


class TestDataLoading:
    """Test suite for intraday data loading functions."""

    @patch("train_intraday.YahooDataLoader")
    def test_load_intraday_data_single_symbol(self, mock_loader_class, sample_intraday_data):
        """Test loading intraday data for a single symbol."""
        # Setup mock
        mock_loader = MagicMock()
        mock_loader.download_intraday.return_value = sample_intraday_data
        mock_loader_class.return_value = mock_loader

        # Load data
        result = load_intraday_data(
            symbols=["RELIANCE.NS"],
            period="60d",
            interval="5m",
        )

        # Verify
        assert "RELIANCE.NS" in result
        assert len(result["RELIANCE.NS"]) == len(sample_intraday_data)
        mock_loader.download_intraday.assert_called_once_with(
            symbol="RELIANCE.NS",
            period="60d",
            interval="5m",
        )

    @patch("train_intraday.YahooDataLoader")
    def test_load_intraday_data_multiple_symbols(self, mock_loader_class, sample_intraday_data):
        """Test loading intraday data for multiple symbols."""
        mock_loader = MagicMock()
        mock_loader.download_intraday.return_value = sample_intraday_data
        mock_loader_class.return_value = mock_loader

        result = load_intraday_data(
            symbols=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
            period="60d",
            interval="5m",
        )

        assert len(result) == 3
        assert all(symbol in result for symbol in ["RELIANCE.NS", "TCS.NS", "INFY.NS"])
        assert mock_loader.download_intraday.call_count == 3

    @patch("train_intraday.YahooDataLoader")
    def test_load_intraday_data_download_failure(self, mock_loader_class):
        """Test handling of download failures."""
        mock_loader = MagicMock()
        mock_loader.download_intraday.side_effect = Exception("Download failed")
        mock_loader_class.return_value = mock_loader

        # Should raise ValueError when no data loaded
        with pytest.raises(ValueError, match="No data loaded"):
            load_intraday_data(
                symbols=["RELIANCE.NS"],
                period="60d",
                interval="5m",
            )

    @patch("train_intraday.YahooDataLoader")
    def test_load_intraday_data_partial_failure(self, mock_loader_class, sample_intraday_data):
        """Test handling when some symbols fail to download."""
        mock_loader = MagicMock()

        # First call succeeds, second fails
        mock_loader.download_intraday.side_effect = [
            sample_intraday_data,
            Exception("Download failed"),
        ]
        mock_loader_class.return_value = mock_loader

        result = load_intraday_data(
            symbols=["RELIANCE.NS", "TCS.NS"],
            period="60d",
            interval="5m",
        )

        # Should succeed with 1 symbol
        assert len(result) == 1
        assert "RELIANCE.NS" in result


# ============================================================================
# Test Data Splitting
# ============================================================================


class TestDataSplitting:
    """Test suite for intraday data splitting functions."""

    def test_split_intraday_data_standard(self, sample_intraday_data):
        """Test standard train/val/test split."""
        splits = split_intraday_data(
            data=sample_intraday_data,
            train_days=10,
            val_days=5,
            test_days=5,
            bars_per_day=75,
        )

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

        # Verify lengths
        assert len(splits["train"]) == 10 * 75
        assert len(splits["validation"]) == 5 * 75
        assert len(splits["test"]) == 5 * 75

    def test_split_intraday_data_temporal_order(self, sample_intraday_data):
        """Test that splits maintain temporal order."""
        splits = split_intraday_data(
            data=sample_intraday_data,
            train_days=10,
            val_days=5,
            test_days=5,
            bars_per_day=75,
        )

        # Train should come before validation
        assert splits["train"].index[-1] < splits["validation"].index[0]

        # Validation should come before test
        assert splits["validation"].index[-1] < splits["test"].index[0]

    def test_split_intraday_data_insufficient_data(self, sample_intraday_data):
        """Test error when data is too short."""
        with pytest.raises(ValueError, match="Data too short"):
            split_intraday_data(
                data=sample_intraday_data,
                train_days=100,  # Way too many days
                val_days=50,
                test_days=50,
                bars_per_day=75,
            )

    def test_split_intraday_data_custom_bars_per_day(self, sample_intraday_data):
        """Test splitting with custom bars per day."""
        splits = split_intraday_data(
            data=sample_intraday_data,
            train_days=10,
            val_days=5,
            test_days=5,
            bars_per_day=50,  # Custom value
        )

        assert len(splits["train"]) == 10 * 50
        assert len(splits["validation"]) == 5 * 50
        assert len(splits["test"]) == 5 * 50


# ============================================================================
# Test Feature Extraction
# ============================================================================


class TestFeatureExtraction:
    """Test suite for intraday feature extraction."""

    @patch("train_intraday.TechnicalFeatures")
    def test_extract_intraday_features_basic(self, mock_features_class, sample_intraday_data):
        """Test basic feature extraction."""
        # Setup mock
        mock_extractor = MagicMock()
        mock_features = pd.DataFrame({
            "SMA_10": np.random.randn(len(sample_intraday_data)),
            "RSI": np.random.randn(len(sample_intraday_data)),
            "ATR": np.random.randn(len(sample_intraday_data)),
            "VWAP": np.random.randn(len(sample_intraday_data)),
        })
        mock_extractor.extract.return_value = mock_features
        mock_features_class.return_value = mock_extractor

        from tradebox.features.technical import FeatureConfig

        config = FeatureConfig(timeframe="intraday")

        # Extract features
        result = extract_intraday_features(
            data=sample_intraday_data,
            feature_config=config,
            fit_normalize=True,
        )

        # Verify
        assert len(result) == len(sample_intraday_data)
        assert "SMA_10" in result.columns
        assert "VWAP" in result.columns
        mock_extractor.extract.assert_called_once()

    @patch("train_intraday.TechnicalFeatures")
    def test_extract_intraday_features_normalization(self, mock_features_class, sample_intraday_data):
        """Test feature extraction with normalization."""
        mock_extractor = MagicMock()
        mock_features = pd.DataFrame(np.random.randn(len(sample_intraday_data), 5))
        mock_extractor.extract.return_value = mock_features
        mock_features_class.return_value = mock_extractor

        from tradebox.features.technical import FeatureConfig

        config = FeatureConfig(timeframe="intraday", normalize=True)

        # Extract with normalization fit
        extract_intraday_features(
            data=sample_intraday_data,
            feature_config=config,
            fit_normalize=True,
        )

        # Verify extractor was called with fit_normalize=True
        call_args = mock_extractor.extract.call_args
        assert call_args is not None


# ============================================================================
# Test Argument Parsing
# ============================================================================


class TestArgumentParsing:
    """Test suite for command-line argument parsing."""

    def test_parse_args_minimal(self):
        """Test parsing minimal required arguments."""
        test_args = ["--config", "configs/test.yaml"]

        with patch("sys.argv", ["train_intraday.py"] + test_args):
            args = parse_args()

        assert args.config == "configs/test.yaml"
        assert args.verbose is False
        assert args.quick is False

    def test_parse_args_full(self):
        """Test parsing all arguments."""
        test_args = [
            "--config", "configs/test.yaml",
            "--timesteps", "1000000",
            "--n-envs", "8",
            "--seed", "123",
            "--device", "cuda",
            "--symbols", "RELIANCE.NS", "TCS.NS",
            "--period", "90d",
            "--output-dir", "models/test",
            "--log-dir", "logs/test",
            "--verbose",
            "--quick",
        ]

        with patch("sys.argv", ["train_intraday.py"] + test_args):
            args = parse_args()

        assert args.config == "configs/test.yaml"
        assert args.timesteps == 1000000
        assert args.n_envs == 8
        assert args.seed == 123
        assert args.device == "cuda"
        assert args.symbols == ["RELIANCE.NS", "TCS.NS"]
        assert args.period == "90d"
        assert args.output_dir == "models/test"
        assert args.log_dir == "logs/test"
        assert args.verbose is True
        assert args.quick is True

    def test_parse_args_missing_required(self):
        """Test that missing --config raises error."""
        with patch("sys.argv", ["train_intraday.py"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_invalid_device(self):
        """Test that invalid device choice raises error."""
        test_args = ["--config", "test.yaml", "--device", "invalid"]

        with patch("sys.argv", ["train_intraday.py"] + test_args):
            with pytest.raises(SystemExit):
                parse_args()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntradayTrainingIntegration:
    """Integration tests for intraday training pipeline."""

    @patch("train_intraday.YahooDataLoader")
    @patch("train_intraday.TechnicalFeatures")
    def test_full_pipeline_mocked(
        self,
        mock_features_class,
        mock_loader_class,
        sample_intraday_data,
        sample_intraday_config,
    ):
        """Test full training pipeline with mocked components."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader.download_intraday.return_value = sample_intraday_data
        mock_loader_class.return_value = mock_loader

        mock_extractor = MagicMock()
        mock_features = pd.DataFrame(np.random.randn(len(sample_intraday_data), 10))
        mock_extractor.extract.return_value = mock_features
        mock_features_class.return_value = mock_extractor

        # Load data
        data_dict = load_intraday_data(
            symbols=["RELIANCE.NS"],
            period="60d",
            interval="5m",
        )

        assert len(data_dict) == 1

        # Split data
        data = data_dict["RELIANCE.NS"]
        splits = split_intraday_data(
            data=data,
            train_days=10,
            val_days=5,
            test_days=5,
            bars_per_day=75,
        )

        assert len(splits) == 3

        # Extract features
        from tradebox.features.technical import FeatureConfig

        config = FeatureConfig(timeframe="intraday")
        train_features = extract_intraday_features(splits["train"], config, fit_normalize=True)

        assert len(train_features) == len(splits["train"])


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
