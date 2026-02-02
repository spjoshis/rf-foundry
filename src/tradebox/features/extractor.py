"""Unified feature extraction pipeline combining technical and fundamental analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from tradebox.data.loaders.fundamental_loader import (
    FundamentalConfig,
    FundamentalDataLoader,
)
from tradebox.features.technical import FeatureConfig, TechnicalFeatures
from tradebox.features.regime import RegimeDetector, RegimeConfig


@dataclass
class FeatureExtractorConfig:
    """
    Combined configuration for technical + fundamental feature extraction.

    Attributes:
        technical: Configuration for technical indicators
        fundamental: Configuration for fundamental metrics
        regime: Configuration for regime detection
        cache_dir: Directory for caching fundamental data
        version: Feature pipeline version for cache invalidation
    """

    technical: FeatureConfig = field(default_factory=FeatureConfig)
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    cache_dir: str = "data/processed"
    version: str = "v2"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure technical and fundamental configs are proper instances
        if isinstance(self.technical, dict):
            self.technical = FeatureConfig(**self.technical)
        if isinstance(self.fundamental, dict):
            self.fundamental = FundamentalConfig(**self.fundamental)
        if isinstance(self.regime, dict):
            self.regime = RegimeConfig(**self.regime)


class FeatureExtractor:
    """
    Unified feature extraction pipeline for technical + fundamental features.

    Combines:
    - Technical indicators (daily data, windowed in observations)
    - Fundamental metrics (quarterly data, static in observations)
    - Separate normalization for each feature type
    - Point-in-time correctness for fundamentals

    The key architectural decision is that fundamentals are treated as STATIC
    features in the observation space (current quarter's values only), while
    technical indicators are windowed (e.g., 60-day history). This is efficient
    because fundamentals change quarterly, not daily.

    Example:
        >>> from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
        >>> from tradebox.features.technical import FeatureConfig
        >>> from tradebox.data.loaders.fundamental_loader import FundamentalConfig
        >>>
        >>> config = FeatureExtractorConfig(
        ...     technical=FeatureConfig(sma_periods=[20, 50]),
        ...     fundamental=FundamentalConfig(enabled=True)
        ... )
        >>> extractor = FeatureExtractor(config)
        >>>
        >>> # Extract features with normalization
        >>> features_df = extractor.extract(
        ...     symbol="RELIANCE",
        ...     price_data=ohlcv_df,
        ...     fit_normalize=True  # True for train, False for val/test
        ... )
        >>>
        >>> # Get feature names
        >>> feature_names = extractor.get_feature_names()
        >>> print(f"Technical: {len(feature_names['technical'])}")
        >>> print(f"Fundamental: {len(feature_names['fundamental'])}")
    """

    def __init__(self, config: FeatureExtractorConfig) -> None:
        """
        Initialize the unified feature extractor.

        Args:
            config: Combined configuration for technical and fundamental features

        Example:
            >>> config = FeatureExtractorConfig(
            ...     technical=FeatureConfig(),
            ...     fundamental=FundamentalConfig(enabled=True)
            ... )
            >>> extractor = FeatureExtractor(config)
        """
        self.config = config

        # Initialize technical feature extractor
        self.technical_extractor = TechnicalFeatures(config.technical)

        # Initialize fundamental data loader
        self.fundamental_loader = FundamentalDataLoader(
            config=config.fundamental,
            cache_dir=Path(config.cache_dir) / "fundamentals",
            use_cache=True,
        )

        # Initialize regime detector
        self.regime_detector = RegimeDetector(config.regime)

        # Separate scalers for technical and fundamental features
        self.technical_scaler: Optional[StandardScaler] = None
        self.fundamental_scaler: Optional[StandardScaler] = None

        logger.info(
            f"FeatureExtractor initialized (version={config.version}, "
            f"regime_type={config.regime.regime_type}, "
            f"fundamental_enabled={config.fundamental.enabled if hasattr(config.fundamental, 'enabled') else True})"
        )

    def extract(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        fit_normalize: bool = False,
    ) -> pd.DataFrame:
        """
        Extract combined technical + fundamental features.

        This method:
        1. Extracts technical indicators from daily price data
        2. Downloads/loads quarterly fundamental data
        3. Applies point-in-time adjustment to fundamentals (45-day delay)
        4. Aligns quarterly fundamentals to daily dates (forward-fill)
        5. Normalizes features using separate scalers

        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS.NS")
            price_data: OHLCV DataFrame with DatetimeIndex
            fit_normalize: If True, fit scalers on this data (use for training).
                          If False, use previously fitted scalers (use for val/test).
                          This separation prevents data leakage.

        Returns:
            DataFrame with columns:
            - Original OHLCV columns (Open, High, Low, Close, Volume)
            - Technical features (SMA_20, RSI, MACD, etc.)
            - Fundamental features (PE_Ratio_Trailing, ROE, etc.) if enabled
            Index: DatetimeIndex matching price_data

        Raises:
            ValueError: If price_data is empty or missing required columns

        Example:
            >>> # Training: fit scalers
            >>> train_features = extractor.extract(
            ...     "RELIANCE", train_data, fit_normalize=True
            ... )
            >>>
            >>> # Validation: use fitted scalers (no refitting)
            >>> val_features = extractor.extract(
            ...     "RELIANCE", val_data, fit_normalize=False
            ... )
        """
        if price_data.empty:
            raise ValueError("price_data cannot be empty")

        # Step 1: Extract technical features (includes normalization if enabled)
        logger.debug(f"Extracting technical features for {symbol}")
        technical_df = self.technical_extractor.extract(
            price_data, fit_normalize=fit_normalize
        )

        # Step 1.5: Detect regime from technical indicators
        logger.debug(f"Detecting market regime for {symbol}")
        try:
            regime_df = self.regime_detector.detect(technical_df)
            # Merge regime features into technical DataFrame
            technical_df = technical_df.join(regime_df, how="left")
            logger.debug(f"Added {len(regime_df.columns)} regime features")
        except Exception as e:
            logger.warning(f"Failed to detect regime: {e}. Continuing without regime features.")

        # Step 2: Load and align fundamental features (if enabled)
        fundamental_enabled = getattr(self.config.fundamental, 'enabled', True)
        if fundamental_enabled:
            logger.debug(f"Extracting fundamental features for {symbol}")
            fundamental_df = self._extract_fundamentals(
                symbol=symbol,
                price_data=price_data,
                fit_normalize=fit_normalize,
            )

            # Merge technical + fundamental
            combined_df = technical_df.join(fundamental_df, how="left")

            # Forward-fill fundamentals to handle daily alignment
            # (quarterly data → daily data)
            fund_cols = fundamental_df.columns.tolist()
            combined_df[fund_cols] = combined_df[fund_cols].ffill()

            logger.info(
                f"Extracted {len(self.technical_extractor.get_feature_names())} technical + "
                f"{len(fund_cols)} fundamental features for {symbol}"
            )
        else:
            combined_df = technical_df
            logger.info(
                f"Extracted {len(self.technical_extractor.get_feature_names())} "
                f"technical features for {symbol} (fundamentals disabled)"
            )

        return combined_df

    def _extract_fundamentals(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        fit_normalize: bool,
    ) -> pd.DataFrame:
        """
        Load fundamental data, apply PIT delay, align to daily, and normalize.

        Process:
        1. Download quarterly fundamentals from yfinance
        2. Point-in-time adjustment already applied by FundamentalDataLoader
           (45-day announcement delay)
        3. Align quarterly data to daily dates using announcement dates
        4. Normalize using separate scaler (prevents mixing with technical features)

        Args:
            symbol: Stock symbol
            price_data: OHLCV DataFrame with DatetimeIndex
            fit_normalize: Whether to fit the scaler

        Returns:
            DataFrame with fundamental features, DatetimeIndex matching price_data

        Example:
            For Q1 2023 (quarter ending Mar 31):
            - Quarter end: 2023-03-31
            - Announcement date: 2023-05-15 (Mar 31 + 45 days)
            - Data available from: 2023-05-15 onwards
            - Forward-filled until next quarter's announcement
        """
        # Download quarterly fundamentals
        start_date = price_data.index.min().strftime("%Y-%m-%d")
        end_date = price_data.index.max().strftime("%Y-%m-%d")

        quarterly_df = self.fundamental_loader.download(
            symbol=symbol,
            start=start_date,
            end=end_date,
        )

        if quarterly_df.empty:
            logger.warning(f"No fundamental data available for {symbol}")
            # Return empty DataFrame with same index as price_data
            return pd.DataFrame(index=price_data.index)

        # Extract feature columns (exclude metadata columns)
        metadata_cols = ["Quarter_End", "Announcement_Date"]
        feature_cols = [col for col in quarterly_df.columns if col not in metadata_cols]

        if not feature_cols:
            logger.warning(f"No fundamental features found for {symbol}")
            return pd.DataFrame(index=price_data.index)

        # Use Announcement_Date as index for point-in-time correctness
        if "Announcement_Date" in quarterly_df.columns:
            fundamental_data = quarterly_df.set_index("Announcement_Date")[feature_cols]
        else:
            # Fallback to using original index
            fundamental_data = quarterly_df[feature_cols]

        # Align quarterly → daily (reindex to price_data dates with forward-fill)
        daily_fundamental = fundamental_data.reindex(
            price_data.index,
            method="ffill",  # Forward-fill quarterly values across days
        )

        # Handle any remaining NaN (before first announcement date)
        # These should remain NaN to maintain point-in-time correctness
        logger.debug(
            f"Fundamental data: {len(fundamental_data)} quarters → "
            f"{len(daily_fundamental)} daily observations"
        )

        # Normalize with separate scaler
        if fit_normalize:
            # Fit scaler on this data (training set)
            self.fundamental_scaler = StandardScaler()

            # Only fit on non-NaN values
            non_nan_mask = daily_fundamental.notna().all(axis=1)
            if non_nan_mask.sum() > 0:
                self.fundamental_scaler.fit(daily_fundamental[non_nan_mask])
                normalized_values = self.fundamental_scaler.transform(
                    daily_fundamental.fillna(0)
                )
            else:
                logger.warning("No non-NaN fundamental data to fit scaler")
                normalized_values = daily_fundamental.fillna(0).values

        elif self.fundamental_scaler is not None:
            # Transform using fitted scaler (validation/test set)
            normalized_values = self.fundamental_scaler.transform(
                daily_fundamental.fillna(0)
            )
        else:
            # No scaler fitted yet, return raw values
            logger.warning(
                "Fundamental scaler not fitted. Use fit_normalize=True on training data first."
            )
            normalized_values = daily_fundamental.values

        # Create normalized DataFrame
        normalized_df = pd.DataFrame(
            normalized_values,
            index=daily_fundamental.index,
            columns=daily_fundamental.columns,
        )

        return normalized_df

    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Return feature names organized by type.

        Returns:
            Dictionary with keys:
            - "technical": List of technical feature names
            - "regime": List of regime feature names
            - "fundamental": List of fundamental feature names
            - "all": Combined list of all features

        Example:
            >>> feature_names = extractor.get_feature_names()
            >>> print(f"Technical features: {feature_names['technical']}")
            >>> print(f"Regime features: {feature_names['regime']}")
            >>> print(f"Fundamental features: {feature_names['fundamental']}")
            >>> print(f"Total features: {len(feature_names['all'])}")
        """
        technical_features = self.technical_extractor.get_feature_names()

        # Regime features
        regime_features = [
            "regime_state",
            "regime_strength",
            "trend_bias",
            "regime_persistence"
        ]

        fundamental_enabled = getattr(self.config.fundamental, 'enabled', True)
        if fundamental_enabled:
            fundamental_features = self.config.fundamental.get_enabled_features()
        else:
            fundamental_features = []

        return {
            "technical": technical_features,
            "regime": regime_features,
            "fundamental": fundamental_features,
            "all": technical_features + regime_features + fundamental_features,
        }

    def get_scaler_params(self) -> Dict[str, Optional[Dict]]:
        """
        Return normalization parameters for both scalers.

        Useful for:
        - Verifying scalers were fitted correctly
        - Debugging normalization issues
        - Saving scaler state for later use

        Returns:
            Dictionary with keys:
            - "technical": Technical scaler parameters (mean, std) or None
            - "fundamental": Fundamental scaler parameters (mean, std) or None

        Example:
            >>> params = extractor.get_scaler_params()
            >>> if params["technical"]:
            ...     print(f"Technical scaler mean: {params['technical']['mean']}")
        """
        technical_params = None
        fundamental_params = None

        if self.technical_scaler is not None:
            # Technical features use the TechnicalFeatures internal scaler
            # Access through _normalization_params
            technical_params = getattr(
                self.technical_extractor, "_normalization_params", None
            )

        if self.fundamental_scaler is not None:
            fundamental_params = {
                "mean": self.fundamental_scaler.mean_.tolist(),
                "scale": self.fundamental_scaler.scale_.tolist(),
                "var": self.fundamental_scaler.var_.tolist(),
            }

        return {
            "technical": technical_params,
            "fundamental": fundamental_params,
        }
