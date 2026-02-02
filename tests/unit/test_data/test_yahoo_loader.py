"""Unit tests for Yahoo Finance data loader."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from tradebox.data.loaders.yahoo_loader import YahooDataLoader


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def loader(temp_cache_dir: Path) -> YahooDataLoader:
    """Create a YahooDataLoader instance for testing."""
    return YahooDataLoader(cache_dir=temp_cache_dir, use_cache=True)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [105.0, 106.0, 107.0, 108.0, 109.0],
        "Low": [95.0, 96.0, 97.0, 98.0, 99.0],
        "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
        "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        "Adj Close": [102.0, 103.0, 104.0, 105.0, 106.0],
    })


class TestYahooDataLoaderInit:
    """Tests for YahooDataLoader initialization."""

    def test_init_creates_cache_dir(self, tmp_path: Path) -> None:
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        loader = YahooDataLoader(cache_dir=cache_dir)

        assert cache_dir.exists()
        assert loader.cache_dir == cache_dir
        assert loader.use_cache is True

    def test_init_loads_existing_metadata(self, temp_cache_dir: Path) -> None:
        """Test that initialization loads existing metadata."""
        # Create metadata file
        metadata = {"TEST_2020_2021": {"symbol": "TEST", "rows": 100}}
        metadata_file = temp_cache_dir / ".cache_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        loader = YahooDataLoader(cache_dir=temp_cache_dir)

        assert loader._metadata == metadata

    def test_init_with_use_cache_false(self, temp_cache_dir: Path) -> None:
        """Test initialization with caching disabled."""
        loader = YahooDataLoader(cache_dir=temp_cache_dir, use_cache=False)

        assert loader.use_cache is False


class TestYahooDataLoaderValidation:
    """Tests for input validation."""

    def test_validate_date_range_valid(self) -> None:
        """Test validation with valid date range."""
        # Should not raise
        YahooDataLoader._validate_date_range("2020-01-01", "2020-12-31")

    def test_validate_date_range_invalid_format(self) -> None:
        """Test validation with invalid date format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            YahooDataLoader._validate_date_range("2020/01/01", "2020-12-31")

    def test_validate_date_range_end_before_start(self) -> None:
        """Test validation when end date is before start date."""
        with pytest.raises(ValueError, match="must be after start date"):
            YahooDataLoader._validate_date_range("2020-12-31", "2020-01-01")


class TestYahooDataLoaderDownload:
    """Tests for data download functionality."""

    @patch("tradebox.data.loaders.yahoo_loader.yf.Ticker")
    def test_download_success(
        self,
        mock_ticker: Mock,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test successful data download."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.history.return_value = sample_data.set_index("Date")
        mock_ticker.return_value = mock_instance

        # Download data
        df = loader.download("RELIANCE.NS", "2020-01-01", "2020-01-05")

        # Verify
        assert not df.empty
        assert len(df) == 5
        assert "Date" in df.columns
        assert "Close" in df.columns
        mock_ticker.assert_called_once_with("RELIANCE.NS")

    @patch("tradebox.data.loaders.yahoo_loader.yf.Ticker")
    def test_download_saves_to_cache(
        self,
        mock_ticker: Mock,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test that downloaded data is saved to cache."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.history.return_value = sample_data.set_index("Date")
        mock_ticker.return_value = mock_instance

        # Download data
        df = loader.download("TEST.NS", "2020-01-01", "2020-01-05")

        # Verify cache file exists
        cache_file = loader.cache_dir / "TEST.NS_2020-01-01_2020-01-05.parquet"
        assert cache_file.exists()

        # Verify metadata
        assert "TEST.NS_2020-01-01_2020-01-05" in loader._metadata

    @patch("tradebox.data.loaders.yahoo_loader.yf.Ticker")
    def test_download_loads_from_cache(
        self,
        mock_ticker: Mock,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test that data is loaded from cache on second call."""
        # Setup mock for first call
        mock_instance = Mock()
        mock_instance.history.return_value = sample_data.set_index("Date")
        mock_ticker.return_value = mock_instance

        # First download
        df1 = loader.download("TEST.NS", "2020-01-01", "2020-01-05")

        # Second download (should use cache)
        df2 = loader.download("TEST.NS", "2020-01-01", "2020-01-05")

        # Verify yfinance was only called once
        assert mock_ticker.call_count == 1

        # Verify data is identical
        pd.testing.assert_frame_equal(df1, df2)

    @patch("tradebox.data.loaders.yahoo_loader.yf.Ticker")
    def test_download_with_retry_on_failure(
        self,
        mock_ticker: Mock,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test retry logic on download failure."""
        # Setup mock to fail once then succeed
        mock_instance = Mock()
        mock_instance.history.side_effect = [
            Exception("Network error"),
            sample_data.set_index("Date"),
        ]
        mock_ticker.return_value = mock_instance

        # Download should succeed after retry
        df = loader.download("TEST.NS", "2020-01-01", "2020-01-05")

        assert not df.empty
        assert mock_instance.history.call_count == 2

    @patch("tradebox.data.loaders.yahoo_loader.yf.Ticker")
    def test_download_failure_after_max_retries(
        self,
        mock_ticker: Mock,
        loader: YahooDataLoader,
    ) -> None:
        """Test that RuntimeError is raised after max retries."""
        # Setup mock to always fail
        mock_instance = Mock()
        mock_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_instance

        # Should raise RuntimeError after 3 retries
        with pytest.raises(RuntimeError, match="Failed to download"):
            loader.download("TEST.NS", "2020-01-01", "2020-01-05")

        # Verify 3 attempts were made
        assert mock_instance.history.call_count == 3


class TestYahooDataLoaderBatchDownload:
    """Tests for batch download functionality."""

    @patch("tradebox.data.loaders.yahoo_loader.yf.Ticker")
    @patch("tradebox.data.loaders.yahoo_loader.time.sleep")
    def test_download_batch_success(
        self,
        mock_sleep: Mock,
        mock_ticker: Mock,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test successful batch download."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.history.return_value = sample_data.set_index("Date")
        mock_ticker.return_value = mock_instance

        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        results = loader.download_batch(symbols, "2020-01-01", "2020-01-05")

        # Verify
        assert len(results) == 3
        for symbol in symbols:
            assert symbol in results
            assert not results[symbol].empty
            assert len(results[symbol]) == 5

        # Verify rate limiting delays
        assert mock_sleep.call_count == 3

    @patch("tradebox.data.loaders.yahoo_loader.yf.Ticker")
    @patch("tradebox.data.loaders.yahoo_loader.time.sleep")
    def test_download_batch_handles_failures(
        self,
        mock_sleep: Mock,
        mock_ticker: Mock,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test batch download handles individual symbol failures."""
        # Setup mock to fail for second symbol
        def side_effect(symbol: str):
            mock_instance = Mock()
            if symbol == "TCS.NS":
                mock_instance.history.side_effect = Exception("Failed")
            else:
                mock_instance.history.return_value = sample_data.set_index("Date")
            return mock_instance

        mock_ticker.side_effect = side_effect

        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        results = loader.download_batch(symbols, "2020-01-01", "2020-01-05")

        # Verify
        assert len(results) == 3
        assert not results["RELIANCE.NS"].empty
        assert results["TCS.NS"].empty  # Failed symbol has empty DataFrame
        assert not results["INFY.NS"].empty


class TestYahooDataLoaderCacheManagement:
    """Tests for cache management."""

    def test_clear_cache_specific_symbol(
        self,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test clearing cache for specific symbol."""
        # Save some data to cache
        loader._save_to_cache(sample_data, "TEST1.NS", "2020-01-01", "2020-12-31")
        loader._save_to_cache(sample_data, "TEST2.NS", "2020-01-01", "2020-12-31")

        # Clear cache for TEST1
        loader.clear_cache("TEST1.NS")

        # Verify TEST1 cache is gone but TEST2 remains
        assert not (loader.cache_dir / "TEST1.NS_2020-01-01_2020-12-31.parquet").exists()
        assert (loader.cache_dir / "TEST2.NS_2020-01-01_2020-12-31.parquet").exists()

        assert "TEST1.NS_2020-01-01_2020-12-31" not in loader._metadata
        assert "TEST2.NS_2020-01-01_2020-12-31" in loader._metadata

    def test_clear_cache_all(
        self,
        loader: YahooDataLoader,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test clearing entire cache."""
        # Save some data to cache
        loader._save_to_cache(sample_data, "TEST1.NS", "2020-01-01", "2020-12-31")
        loader._save_to_cache(sample_data, "TEST2.NS", "2020-01-01", "2020-12-31")

        # Clear entire cache
        loader.clear_cache()

        # Verify all cache is gone
        assert len(list(loader.cache_dir.glob("*.parquet"))) == 0
        assert len(loader._metadata) == 0
