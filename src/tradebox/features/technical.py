"""Technical indicator feature extraction using ta library."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


@dataclass
class FeatureConfig:
    """
    Configuration for technical feature extraction.

    Supports both EOD (daily) and intraday (bar-based) feature extraction.
    For intraday trading, period values represent bars (e.g., 14 bars ~= 70 minutes for 5-min data).
    For EOD trading, period values represent days (e.g., 14 days).

    Attributes:
        version: Feature version for cache invalidation
        normalize: Whether to apply z-score normalization
        lookback_window: Number of periods (days or bars) for rolling features
        timeframe: "eod" or "intraday" - determines period interpretation
        trend_enabled: Enable trend indicators (SMA, EMA)
        momentum_enabled: Enable momentum indicators (RSI, Stochastic, ROC)
        volatility_enabled: Enable volatility indicators (ATR, Bollinger Bands)
        volume_enabled: Enable volume indicators (Volume MA, OBV)

        # Trend parameters
        sma_periods: Periods for SMA indicators
                    EOD: [20, 50, 200] days
                    Intraday: [10, 20, 50] bars (~50min, 100min, 4hours for 5-min)
        ema_periods: Periods for EMA indicators
                    EOD: [9, 21] days
                    Intraday: [9, 21] bars

        # Momentum parameters
        rsi_period: Period for RSI (14 days/bars)
        stochastic_period: Period for Stochastic Oscillator (14 days/bars)
        roc_period: Period for Rate of Change (10 days/bars)

        # Volatility parameters
        atr_period: Period for Average True Range (14 days/bars)
        bollinger_period: Period for Bollinger Bands (20 days/bars)
        bollinger_std: Standard deviation multiplier for Bollinger Bands
        rolling_std_period: Period for rolling standard deviation (20 days/bars)

        # Volume parameters
        volume_ma_period: Period for volume moving average (20 days/bars)
        obv: Enable On-Balance Volume indicator

        # Intraday-specific features
        vwap_enabled: Enable session VWAP calculation (intraday only)
        session_high_low: Track intraday session high/low (intraday only)
        intraday_returns: Enable multi-bar return features (1, 3, 10 bars)
    """

    version: str = "1.0"
    normalize: bool = True
    lookback_window: int = 60  # 60 days for EOD, 60 bars for intraday
    timeframe: str = "eod"  # "eod" or "intraday"

    # Feature groups
    trend_enabled: bool = True
    momentum_enabled: bool = True
    volatility_enabled: bool = True
    volume_enabled: bool = True

    # Trend parameters (interpreted as days for EOD, bars for intraday)
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [9, 21])

    # Momentum parameters
    rsi_period: int = 14
    stochastic_period: int = 14
    roc_period: int = 10

    # Volatility parameters
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: int = 2
    rolling_std_period: int = 20

    # Volume parameters
    volume_ma_period: int = 20
    obv: bool = True

    # Directional indicators for regime detection
    adx_enabled: bool = True
    adx_period: int = 14

    # Intraday-specific features
    vwap_enabled: bool = False  # Session VWAP (reset daily)
    session_high_low: bool = False  # Track session extremes
    intraday_returns: bool = False  # Multi-bar returns (1, 3, 10)

    def __post_init__(self):
        """Adjust default parameters for intraday timeframe."""
        if self.timeframe == "intraday":
            # Override default periods for intraday
            if self.sma_periods == [20, 50, 200]:  # Check if using defaults
                self.sma_periods = [10, 20, 50]  # ~50min, 100min, 4hours for 5-min

            # Adjust ADX period for faster response in intraday
            if self.adx_enabled and self.adx_period == 14:
                self.adx_period = 7  # 7 bars = 35 minutes for 5-min data
                logger.info(f"Adjusted ADX period to {self.adx_period} for intraday")

            # Enable intraday-specific features by default
            self.vwap_enabled = True
            self.session_high_low = True
            self.intraday_returns = True

            logger.info(
                f"FeatureConfig initialized for intraday trading: "
                f"SMA periods={self.sma_periods}, VWAP={self.vwap_enabled}"
            )


class TechnicalFeatures:
    """
    Extract technical indicators from OHLCV data.

    This class provides:
    - 20-25 technical indicators across 4 categories
    - Configurable indicator selection
    - Z-score normalization
    - Feature metadata tracking

    Example:
        >>> config = FeatureConfig()
        >>> extractor = TechnicalFeatures(config)
        >>> features = extractor.extract(df)
        >>> print(f"Extracted {len(extractor.get_feature_names())} features")
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """
        Initialize technical feature extractor.

        Args:
            config: Feature configuration. If None, uses defaults.
        """
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
        self._normalization_params: Dict[str, Dict[str, float]] = {}
        logger.info(f"TechnicalFeatures initialized with version {self.config.version}")

    def extract(self, df: pd.DataFrame, fit_normalize: bool = True) -> pd.DataFrame:
        """
        Extract all configured technical indicators.

        Args:
            df: DataFrame with OHLCV columns
            fit_normalize: If True, fit normalization on this data.
                          If False, use previously fitted parameters.

        Returns:
            DataFrame with original columns plus technical features

        Raises:
            ValueError: If required columns are missing

        Example:
            >>> df = pd.read_parquet("RELIANCE.parquet")
            >>> features = extractor.extract(df)
        """
        self._validate_input(df)

        features = df.copy()
        self.feature_names = []

        # Extract each category
        if self.config.trend_enabled:
            features = self._add_trend_features(features)

        # Add directional features (ADX, +DI, -DI) for regime detection
        if self.config.adx_enabled:
            features = self._add_directional_features(features)

        if self.config.momentum_enabled:
            features = self._add_momentum_features(features)

        if self.config.volatility_enabled:
            features = self._add_volatility_features(features)

        if self.config.volume_enabled:
            features = self._add_volume_features(features)

        # Add intraday-specific features if enabled
        if self.config.timeframe == "intraday":
            features = self._add_intraday_features(features)

        # Handle NaN values (from indicators with warm-up periods)
        # Fill NaN with 0 for feature columns only
        feature_cols = [col for col in features.columns if col in self.feature_names]
        features[feature_cols] = features[feature_cols].fillna(0.0)

        # Normalize features
        if self.config.normalize:
            features = self._normalize_features(features, fit=fit_normalize)

        logger.info(f"Extracted {len(self.feature_names)} technical features")
        return features

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators (SMA, EMA, MACD)."""
        close = df["Close"]

        # Simple Moving Averages
        for period in self.config.sma_periods:
            sma = SMAIndicator(close=close, window=period)
            col_name = f"SMA_{period}"
            df[col_name] = sma.sma_indicator()
            self.feature_names.append(col_name)

            # Price/SMA ratio
            ratio_name = f"Close_SMA{period}_Ratio"
            df[ratio_name] = close / df[col_name]
            self.feature_names.append(ratio_name)

        # Exponential Moving Averages
        for period in self.config.ema_periods:
            ema = EMAIndicator(close=close, window=period)
            col_name = f"EMA_{period}"
            df[col_name] = ema.ema_indicator()
            self.feature_names.append(col_name)

        # MACD
        macd = MACD(close=close)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Diff"] = macd.macd_diff()
        self.feature_names.extend(["MACD", "MACD_Signal", "MACD_Diff"])

        logger.debug(f"Added {len(self.config.sma_periods) * 2 + len(self.config.ema_periods) + 3} trend features")
        return df

    def _add_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add directional indicators (ADX, +DI, -DI) for regime detection.

        ADX (Average Directional Index) measures trend strength:
        - ADX > 25: Strong trend (trending market)
        - ADX < 20: Weak trend (ranging market)
        - Between 20-25: Transition zone

        +DI and -DI measure trend direction:
        - +DI > -DI: Uptrend
        - -DI > +DI: Downtrend

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with added directional features
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # Calculate ADX and directional indicators using ta library
        try:
            adx_indicator = ADXIndicator(
                high=high,
                low=low,
                close=close,
                window=self.config.adx_period
            )

            df["ADX"] = adx_indicator.adx()
            df["Plus_DI"] = adx_indicator.adx_pos()
            df["Minus_DI"] = adx_indicator.adx_neg()

            self.feature_names.extend(["ADX", "Plus_DI", "Minus_DI"])

            logger.debug(
                f"Added 3 directional features (ADX period={self.config.adx_period})"
            )
        except Exception as e:
            logger.warning(f"Failed to calculate ADX indicators: {e}")
            # Add zero-filled columns as fallback
            df["ADX"] = 0.0
            df["Plus_DI"] = 0.0
            df["Minus_DI"] = 0.0
            self.feature_names.extend(["ADX", "Plus_DI", "Minus_DI"])

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators (RSI, Stochastic, ROC)."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # RSI
        rsi = RSIIndicator(close=close, window=self.config.rsi_period)
        df["RSI"] = rsi.rsi()
        self.feature_names.append("RSI")

        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=self.config.stochastic_period,
        )
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()
        self.feature_names.extend(["Stoch_K", "Stoch_D"])

        # Rate of Change
        roc = ROCIndicator(close=close, window=self.config.roc_period)
        df["ROC"] = roc.roc()
        self.feature_names.append("ROC")

        logger.debug("Added 4 momentum features")
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators (ATR, Bollinger Bands, Std Dev)."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # Average True Range
        atr = AverageTrueRange(high=high, low=low, close=close, window=self.config.atr_period)
        df["ATR"] = atr.average_true_range()
        # Normalized ATR (% of price)
        df["ATR_Pct"] = (df["ATR"] / close) * 100
        self.feature_names.extend(["ATR", "ATR_Pct"])

        # Bollinger Bands
        bb = BollingerBands(
            close=close,
            window=self.config.bollinger_period,
            window_dev=self.config.bollinger_std,
        )
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()
        df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]
        df["BB_Position"] = (close - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"])
        self.feature_names.extend(["BB_High", "BB_Low", "BB_Mid", "BB_Width", "BB_Position"])

        # Rolling Standard Deviation of returns
        returns = close.pct_change()
        df["Returns_Std"] = returns.rolling(self.config.rolling_std_period).std()
        self.feature_names.append("Returns_Std")

        logger.debug("Added 8 volatility features")
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators (Volume MA, OBV)."""
        volume = df["Volume"]
        close = df["Close"]

        # Volume Moving Average
        volume_ma = volume.rolling(self.config.volume_ma_period).mean()
        df["Volume_MA"] = volume_ma
        df["Volume_Ratio"] = volume / volume_ma
        self.feature_names.extend(["Volume_MA", "Volume_Ratio"])

        # On-Balance Volume
        if self.config.obv:
            obv = OnBalanceVolumeIndicator(close=close, volume=volume)
            df["OBV"] = obv.on_balance_volume()
            # Normalized OBV (divide by volume to make it scale-invariant)
            df["OBV_Norm"] = df["OBV"] / (volume.rolling(20).mean() + 1)
            self.feature_names.extend(["OBV", "OBV_Norm"])

        logger.debug(f"Added {4 if self.config.obv else 2} volume features")
        return df

    def _add_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intraday-specific features (VWAP, session high/low, multi-bar returns).

        These features are only relevant for intraday trading with 5-minute or similar bars.

        Args:
            df: DataFrame with OHLCV columns and datetime index

        Returns:
            DataFrame with added intraday features
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Ensure we have a datetime index or Date column
        if "Date" in df.columns:
            df["_date"] = pd.to_datetime(df["Date"]).dt.date
        elif isinstance(df.index, pd.DatetimeIndex):
            df["_date"] = df.index.date
        else:
            logger.warning("Cannot compute session-based features without datetime information")
            return df

        feature_count = 0

        # 1. Session VWAP (Volume-Weighted Average Price)
        if self.config.vwap_enabled:
            # Calculate cumulative sum within each session (day)
            df["_cum_volume"] = df.groupby("_date")["Volume"].cumsum()
            df["_cum_vol_price"] = df.groupby("_date").apply(
                lambda x: (x["Close"] * x["Volume"]).cumsum()
            ).reset_index(level=0, drop=True)

            # VWAP = cumulative(price × volume) / cumulative(volume)
            df["VWAP"] = df["_cum_vol_price"] / (df["_cum_volume"] + 1e-8)

            # VWAP deviation (how far current price is from VWAP)
            df["VWAP_Deviation"] = (close - df["VWAP"]) / (df["VWAP"] + 1e-8)

            self.feature_names.extend(["VWAP", "VWAP_Deviation"])
            feature_count += 2

            # Clean up temporary columns
            df = df.drop(columns=["_cum_volume", "_cum_vol_price"])

        # 2. Session High/Low tracking
        if self.config.session_high_low:
            # Cumulative max/min within each session
            df["Session_High"] = df.groupby("_date")["High"].cummax()
            df["Session_Low"] = df.groupby("_date")["Low"].cummin()

            # Current position within session range
            session_range = df["Session_High"] - df["Session_Low"]
            df["Session_Position"] = (
                (close - df["Session_Low"]) / (session_range + 1e-8)
            )  # 0 to 1

            self.feature_names.extend(["Session_High", "Session_Low", "Session_Position"])
            feature_count += 3

        # 3. Multi-bar returns (faster mean reversion signals)
        if self.config.intraday_returns:
            # 1-bar return (5 minutes)
            df["Return_1bar"] = close.pct_change(1)

            # 3-bar return (15 minutes)
            df["Return_3bar"] = close.pct_change(3)

            # 10-bar return (~50 minutes)
            df["Return_10bar"] = close.pct_change(10)

            # High-Low range as percentage of close
            df["HL_Range_Pct"] = (high - low) / (close + 1e-8)

            # Close-Open range (bar direction)
            df["CO_Range_Pct"] = (close - df["Open"]) / (df["Open"] + 1e-8)

            self.feature_names.extend([
                "Return_1bar",
                "Return_3bar",
                "Return_10bar",
                "HL_Range_Pct",
                "CO_Range_Pct",
            ])
            feature_count += 5

        # Clean up temporary date column
        df = df.drop(columns=["_date"])

        logger.debug(f"Added {feature_count} intraday-specific features")
        return df

    def _normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply z-score normalization to features.

        Args:
            df: DataFrame with features
            fit: If True, compute mean/std from data. If False, use stored params.

        Returns:
            DataFrame with normalized features
        """
        for feature in self.feature_names:
            if feature not in df.columns:
                continue

            if fit:
                # Compute and store normalization parameters
                mean = df[feature].mean()
                std = df[feature].std()
                self._normalization_params[feature] = {"mean": mean, "std": std}
            else:
                # Use stored parameters
                if feature not in self._normalization_params:
                    logger.warning(f"No normalization params for {feature}, skipping")
                    continue
                mean = self._normalization_params[feature]["mean"]
                std = self._normalization_params[feature]["std"]

            # Apply z-score normalization
            if std > 0:
                df[feature] = (df[feature] - mean) / std
            else:
                logger.warning(f"Zero std for {feature}, setting to 0")
                df[feature] = 0.0

        return df

    def get_feature_names(self) -> List[str]:
        """
        Return list of all extracted feature names.

        Returns:
            List of feature names
        """
        return self.feature_names.copy()

    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about feature extraction.

        Returns:
            Dictionary with version, indicators, config
        """
        return {
            "version": self.config.version,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "normalization": self.config.normalize,
            "lookback_window": self.config.lookback_window,
            "config": {
                "trend_enabled": self.config.trend_enabled,
                "momentum_enabled": self.config.momentum_enabled,
                "volatility_enabled": self.config.volatility_enabled,
                "volume_enabled": self.config.volume_enabled,
            },
        }

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """
        Validate input DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df) < 200:
            logger.warning(
                f"DataFrame has only {len(df)} rows. "
                f"Some indicators (e.g., SMA 200) may have many NaN values."
            )
