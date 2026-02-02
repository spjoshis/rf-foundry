"""Unit tests for data splitter."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from tradebox.data.splitter import DataSplitter, SplitConfig


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data spanning 2010-2024."""
    dates = pd.date_range("2010-01-01", "2024-12-31", freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Close": [100.0 + i * 0.01 for i in range(len(dates))],
        "Volume": [1000000] * len(dates),
    })


@pytest.fixture
def default_config() -> SplitConfig:
    """Create default split configuration."""
    return SplitConfig.default()


@pytest.fixture
def splitter(default_config: SplitConfig) -> DataSplitter:
    """Create DataSplitter with default config."""
    return DataSplitter(default_config)


class TestSplitConfig:
    """Tests for SplitConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SplitConfig.default()

        assert config.train_start == "2010-01-01"
        assert config.train_end == "2018-12-31"
        assert config.val_start == "2019-01-01"
        assert config.val_end == "2021-12-31"
        assert config.test_start == "2022-01-01"
        assert config.test_end == "2024-12-31"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SplitConfig(
            train_start="2015-01-01",
            train_end="2018-12-31",
            val_start="2019-01-01",
            val_end="2020-12-31",
            test_start="2021-01-01",
            test_end="2022-12-31",
        )

        assert config.train_start == "2015-01-01"
        assert config.test_end == "2022-12-31"


class TestDataSplitterInit:
    """Tests for DataSplitter initialization."""

    def test_init_with_valid_config(self, default_config: SplitConfig) -> None:
        """Test initialization with valid configuration."""
        splitter = DataSplitter(default_config)

        assert splitter.config == default_config

    def test_init_with_overlapping_dates(self) -> None:
        """Test initialization fails with overlapping dates."""
        config = SplitConfig(
            train_start="2010-01-01",
            train_end="2020-12-31",
            val_start="2019-01-01",  # Overlaps with train
            val_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2024-12-31",
        )

        with pytest.raises(ValueError, match="must be after train end"):
            DataSplitter(config)

    def test_init_with_invalid_date_range(self) -> None:
        """Test initialization fails when end < start."""
        config = SplitConfig(
            train_start="2018-12-31",
            train_end="2010-01-01",  # End before start
            val_start="2019-01-01",
            val_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2024-12-31",
        )

        with pytest.raises(ValueError, match="train_end must be >= train_start"):
            DataSplitter(config)


class TestDataSplitting:
    """Tests for data splitting functionality."""

    def test_split_single_dataframe(
        self,
        splitter: DataSplitter,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test splitting a single DataFrame."""
        splits = splitter.split(sample_data)

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        assert all(isinstance(df, pd.DataFrame) for df in splits.values())

    def test_split_date_ranges(
        self,
        splitter: DataSplitter,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test that splits respect configured date ranges."""
        splits = splitter.split(sample_data)

        # Check train dates
        train = splits["train"]
        assert train["Date"].min() >= pd.Timestamp("2010-01-01")
        assert train["Date"].max() <= pd.Timestamp("2018-12-31")

        # Check validation dates
        val = splits["validation"]
        assert val["Date"].min() >= pd.Timestamp("2019-01-01")
        assert val["Date"].max() <= pd.Timestamp("2021-12-31")

        # Check test dates
        test = splits["test"]
        assert test["Date"].min() >= pd.Timestamp("2022-01-01")
        assert test["Date"].max() <= pd.Timestamp("2024-12-31")

    def test_split_no_overlap(
        self,
        splitter: DataSplitter,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test that splits have no temporal overlap."""
        splits = splitter.split(sample_data)

        train_max = splits["train"]["Date"].max()
        val_min = splits["validation"]["Date"].min()
        val_max = splits["validation"]["Date"].max()
        test_min = splits["test"]["Date"].min()

        assert train_max < val_min
        assert val_max < test_min

    def test_split_multi_symbol(self, splitter: DataSplitter) -> None:
        """Test splitting multiple symbols."""
        data = {
            "SYMBOL1": pd.DataFrame({
                "Date": pd.date_range("2010-01-01", "2024-12-31", freq="D"),
                "Close": [100.0] * 5479,
            }),
            "SYMBOL2": pd.DataFrame({
                "Date": pd.date_range("2010-01-01", "2024-12-31", freq="D"),
                "Close": [200.0] * 5479,
            }),
        }

        splits = splitter.split(data)

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits
        # Should combine data from both symbols
        assert len(splits["train"]) > 0
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0

    def test_split_with_missing_date_column(self, splitter: DataSplitter) -> None:
        """Test that split fails without Date column."""
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="must have 'Date' column"):
            splitter.split(df)

    def test_split_with_insufficient_data(self, splitter: DataSplitter) -> None:
        """Test splitting with data that doesn't cover all ranges."""
        # Data only from 2020-2022 (missing train period)
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", "2022-12-31", freq="D"),
            "Close": [100.0] * 1096,
        })

        splits = splitter.split(df)

        # Train should be empty (data starts in 2020)
        assert len(splits["train"]) == 0
        # Validation and test should have data
        assert len(splits["validation"]) > 0
        assert len(splits["test"]) > 0


class TestSavingAndLoading:
    """Tests for saving splits to disk."""

    def test_save_splits(
        self,
        splitter: DataSplitter,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test saving splits to Parquet files."""
        splits = splitter.split(sample_data)
        output_dir = tmp_path / "splits"

        splitter.save_splits(splits, output_dir, "TEST")

        # Check files exist
        symbol_dir = output_dir / "TEST"
        assert (symbol_dir / "train.parquet").exists()
        assert (symbol_dir / "validation.parquet").exists()
        assert (symbol_dir / "test.parquet").exists()
        assert (symbol_dir / "metadata.json").exists()

    def test_save_splits_metadata(
        self,
        splitter: DataSplitter,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test metadata is correctly saved."""
        splits = splitter.split(sample_data)
        output_dir = tmp_path / "splits"

        splitter.save_splits(splits, output_dir, "TEST")

        # Read metadata
        metadata_file = output_dir / "TEST" / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        assert metadata["symbol"] == "TEST"
        assert "created_at" in metadata
        assert "split_config" in metadata
        assert "train" in metadata["split_config"]
        assert "validation" in metadata["split_config"]
        assert "test" in metadata["split_config"]
        assert metadata["split_config"]["train"]["n_samples"] > 0

    def test_load_saved_splits(
        self,
        splitter: DataSplitter,
        sample_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test that saved splits can be loaded back."""
        splits = splitter.split(sample_data)
        output_dir = tmp_path / "splits"

        splitter.save_splits(splits, output_dir, "TEST")

        # Load back
        symbol_dir = output_dir / "TEST"
        loaded_train = pd.read_parquet(symbol_dir / "train.parquet")
        loaded_val = pd.read_parquet(symbol_dir / "validation.parquet")
        loaded_test = pd.read_parquet(symbol_dir / "test.parquet")

        # Compare with original (check_index=False because parquet resets index)
        pd.testing.assert_frame_equal(splits["train"].reset_index(drop=True), loaded_train)
        pd.testing.assert_frame_equal(splits["validation"].reset_index(drop=True), loaded_val)
        pd.testing.assert_frame_equal(splits["test"].reset_index(drop=True), loaded_test)


class TestValidation:
    """Tests for split validation."""

    def test_validate_valid_splits(
        self,
        splitter: DataSplitter,
        sample_data: pd.DataFrame,
    ) -> None:
        """Test validation passes for valid splits."""
        splits = splitter.split(sample_data)

        assert splitter.validate_splits(splits) is True

    def test_validate_missing_date_column(self, splitter: DataSplitter) -> None:
        """Test validation fails if Date column missing."""
        splits = {
            "train": pd.DataFrame({"Close": [100, 101]}),
            "validation": pd.DataFrame({"Date": pd.date_range("2019-01-01", periods=2)}),
            "test": pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=2)}),
        }

        with pytest.raises(ValueError, match="missing 'Date' column"):
            splitter.validate_splits(splits)

    def test_validate_overlapping_train_val(self, default_config: SplitConfig) -> None:
        """Test validation fails with overlapping train/validation."""
        splitter = DataSplitter(default_config)

        # Create overlapping splits
        splits = {
            "train": pd.DataFrame({"Date": pd.date_range("2010-01-01", "2020-01-01")}),
            "validation": pd.DataFrame({"Date": pd.date_range("2019-01-01", "2022-01-01")}),
            "test": pd.DataFrame({"Date": pd.date_range("2023-01-01", "2024-01-01")}),
        }

        with pytest.raises(ValueError, match="Train overlaps with validation"):
            splitter.validate_splits(splits)

    def test_validate_overlapping_val_test(self, default_config: SplitConfig) -> None:
        """Test validation fails with overlapping validation/test."""
        splitter = DataSplitter(default_config)

        # Create overlapping splits
        splits = {
            "train": pd.DataFrame({"Date": pd.date_range("2010-01-01", "2018-01-01")}),
            "validation": pd.DataFrame({"Date": pd.date_range("2019-01-01", "2023-01-01")}),
            "test": pd.DataFrame({"Date": pd.date_range("2022-01-01", "2024-01-01")}),
        }

        with pytest.raises(ValueError, match="Validation overlaps with test"):
            splitter.validate_splits(splits)

    def test_validate_empty_splits(self, splitter: DataSplitter) -> None:
        """Test validation warns about empty splits."""
        splits = {
            "train": pd.DataFrame({"Date": []}),
            "validation": pd.DataFrame({"Date": pd.date_range("2019-01-01", periods=10)}),
            "test": pd.DataFrame({"Date": pd.date_range("2022-01-01", periods=10)}),
        }

        # Should pass but log warning
        assert splitter.validate_splits(splits) is True
