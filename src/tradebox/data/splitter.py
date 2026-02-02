"""Data splitting for train/validation/test sets with temporal ordering."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from loguru import logger


@dataclass
class SplitConfig:
    """
    Configuration for data splitting.

    Attributes:
        train_start: Start date for training set (YYYY-MM-DD)
        train_end: End date for training set (YYYY-MM-DD)
        val_start: Start date for validation set (YYYY-MM-DD)
        val_end: End date for validation set (YYYY-MM-DD)
        test_start: Start date for test set (YYYY-MM-DD)
        test_end: End date for test set (YYYY-MM-DD)
    """

    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    @classmethod
    def default(cls) -> "SplitConfig":
        """
        Return default split configuration.

        Returns:
            Default configuration with:
            - Train: 2010-2018 (9 years, ~70%)
            - Validation: 2019-2021 (3 years, ~15%)
            - Test: 2022-2024 (3 years, ~15%)
        """
        return cls(
            train_start="2010-01-01",
            train_end="2018-12-31",
            val_start="2019-01-01",
            val_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2024-12-31",
        )


class DataSplitter:
    """
    Split time-series data into train/validation/test sets.

    This splitter ensures:
    - Strict temporal ordering (no data leakage)
    - Configurable date ranges
    - Metadata tracking for reproducibility
    - Support for single and multi-symbol splitting

    Example:
        >>> config = SplitConfig.default()
        >>> splitter = DataSplitter(config)
        >>> splits = splitter.split(df)
        >>> print(f"Train: {len(splits['train'])} rows")
    """

    def __init__(self, config: SplitConfig) -> None:
        """
        Initialize data splitter.

        Args:
            config: Split configuration with date ranges

        Raises:
            ValueError: If configuration is invalid (overlapping dates, etc.)
        """
        self.config = config
        self._validate_config()
        logger.info(f"DataSplitter initialized with config: {config}")

    def split(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets.

        Args:
            data: Single DataFrame or dict of {symbol: DataFrame}

        Returns:
            Dictionary with keys 'train', 'validation', 'test'

        Raises:
            ValueError: If data doesn't cover required date ranges

        Example:
            >>> df = pd.read_parquet("RELIANCE.parquet")
            >>> splits = splitter.split(df)
            >>> train_df = splits['train']
        """
        if isinstance(data, dict):
            # Multi-symbol: combine all symbols
            logger.info(f"Splitting {len(data)} symbols")
            combined_splits: Dict[str, pd.DataFrame] = {
                "train": pd.DataFrame(),
                "validation": pd.DataFrame(),
                "test": pd.DataFrame(),
            }

            for symbol, df in data.items():
                symbol_splits = self._split_single(df, symbol)
                for split_name in ["train", "validation", "test"]:
                    combined_splits[split_name] = pd.concat(
                        [combined_splits[split_name], symbol_splits[split_name]],
                        ignore_index=True,
                    )

            return combined_splits
        else:
            # Single DataFrame
            return self._split_single(data)

    def _split_single(
        self,
        df: pd.DataFrame,
        symbol: str = "unknown",
    ) -> Dict[str, pd.DataFrame]:
        """
        Split a single DataFrame by date ranges.

        Args:
            df: DataFrame with Date column
            symbol: Symbol name for logging

        Returns:
            Dictionary with train/validation/test splits
        """
        if "Date" not in df.columns:
            raise ValueError("DataFrame must have 'Date' column")

        # Ensure Date is datetime
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        # Split by date ranges
        train = df[
            (df["Date"] >= self.config.train_start) & (df["Date"] <= self.config.train_end)
        ]
        validation = df[
            (df["Date"] >= self.config.val_start) & (df["Date"] <= self.config.val_end)
        ]
        test = df[
            (df["Date"] >= self.config.test_start) & (df["Date"] <= self.config.test_end)
        ]

        logger.info(
            f"Split {symbol}: train={len(train)}, val={len(validation)}, test={len(test)}"
        )

        # Warn if any split is empty
        if len(train) == 0:
            logger.warning(f"Train split is empty for {symbol}")
        if len(validation) == 0:
            logger.warning(f"Validation split is empty for {symbol}")
        if len(test) == 0:
            logger.warning(f"Test split is empty for {symbol}")

        return {
            "train": train,
            "validation": validation,
            "test": test,
        }

    def save_splits(
        self,
        splits: Dict[str, pd.DataFrame],
        output_dir: Path,
        symbol: str,
    ) -> None:
        """
        Save splits to Parquet files with metadata.

        Args:
            splits: Dictionary with train/validation/test DataFrames
            output_dir: Directory to save splits
            symbol: Symbol name for directory structure

        Example:
            >>> splits = splitter.split(df)
            >>> splitter.save_splits(splits, Path("data/splits"), "RELIANCE")
        """
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        for split_name, split_df in splits.items():
            split_file = symbol_dir / f"{split_name}.parquet"
            split_df.to_parquet(split_file, index=False)
            logger.info(f"Saved {split_name} split: {split_file}")

        # Save metadata
        metadata = {
            "symbol": symbol,
            "created_at": datetime.now().isoformat(),
            "split_config": {
                "train": {
                    "start": self.config.train_start,
                    "end": self.config.train_end,
                    "n_samples": len(splits["train"]),
                },
                "validation": {
                    "start": self.config.val_start,
                    "end": self.config.val_end,
                    "n_samples": len(splits["validation"]),
                },
                "test": {
                    "start": self.config.test_start,
                    "end": self.config.test_end,
                    "n_samples": len(splits["test"]),
                },
            },
            "data_version": "1.0",
        }

        metadata_file = symbol_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata: {metadata_file}")

    def validate_splits(self, splits: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that splits have no overlap and proper temporal ordering.

        Args:
            splits: Dictionary with train/validation/test DataFrames

        Returns:
            True if splits are valid

        Raises:
            ValueError: If validation fails
        """
        train = splits["train"]
        validation = splits["validation"]
        test = splits["test"]

        # Check all splits have Date column
        for split_name, split_df in splits.items():
            if "Date" not in split_df.columns:
                raise ValueError(f"{split_name} split missing 'Date' column")
            if len(split_df) == 0:
                logger.warning(f"{split_name} split is empty")

        # Check temporal ordering (train < validation < test)
        if len(train) > 0 and len(validation) > 0:
            train_max = train["Date"].max()
            val_min = validation["Date"].min()
            if train_max >= val_min:
                raise ValueError(
                    f"Train overlaps with validation: "
                    f"train_max={train_max}, val_min={val_min}"
                )

        if len(validation) > 0 and len(test) > 0:
            val_max = validation["Date"].max()
            test_min = test["Date"].min()
            if val_max >= test_min:
                raise ValueError(
                    f"Validation overlaps with test: "
                    f"val_max={val_max}, test_min={test_min}"
                )

        logger.info("Split validation passed")
        return True

    def _validate_config(self) -> None:
        """
        Validate split configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Parse dates
        try:
            train_start = datetime.strptime(self.config.train_start, "%Y-%m-%d")
            train_end = datetime.strptime(self.config.train_end, "%Y-%m-%d")
            val_start = datetime.strptime(self.config.val_start, "%Y-%m-%d")
            val_end = datetime.strptime(self.config.val_end, "%Y-%m-%d")
            test_start = datetime.strptime(self.config.test_start, "%Y-%m-%d")
            test_end = datetime.strptime(self.config.test_end, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format in config: {e}") from e

        # Check ordering within each split
        if train_end < train_start:
            raise ValueError("train_end must be >= train_start")
        if val_end < val_start:
            raise ValueError("val_end must be >= val_start")
        if test_end < test_start:
            raise ValueError("test_end must be >= test_start")

        # Check no overlap between splits
        if val_start <= train_end:
            raise ValueError(
                f"Validation start ({val_start}) must be after train end ({train_end})"
            )
        if test_start <= val_end:
            raise ValueError(
                f"Test start ({test_start}) must be after validation end ({val_end})"
            )

        logger.debug("Configuration validated successfully")
