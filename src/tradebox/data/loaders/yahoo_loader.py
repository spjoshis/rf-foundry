"""Yahoo Finance data loader with caching and corporate action handling."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger


class YahooDataLoader:
    """
    Download and cache historical stock data from Yahoo Finance.

    This loader handles:
    - OHLCV data downloads for Indian stocks (NSE)
    - Automatic corporate action adjustments (splits, dividends)
    - Intelligent caching using Parquet format
    - Rate limit handling and retry logic
    - Batch downloads with progress tracking

    Attributes:
        cache_dir: Directory for storing cached data
        use_cache: Whether to use cached data when available
        metadata_file: Path to cache metadata JSON file
    """

    def __init__(self, cache_dir: Path, use_cache: bool = True) -> None:
        """
        Initialize the Yahoo Finance data loader.

        Args:
            cache_dir: Directory path for caching downloaded data
            use_cache: If True, load from cache when available

        Raises:
            ValueError: If cache_dir is not a valid directory path
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.metadata_file = self.cache_dir / ".cache_metadata.json"

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self._metadata = self._load_metadata()

        logger.info(f"YahooDataLoader initialized with cache_dir={self.cache_dir}")

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """
        Download OHLCV data for a single symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS" for NSE)
            start: Start date in "YYYY-MM-DD" format
            end: End date in "YYYY-MM-DD" format
            auto_adjust: If True, adjust for splits and dividends

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume, Adj Close
            Index is DatetimeIndex

        Raises:
            ValueError: If date range is invalid
            RuntimeError: If download fails after retries

        Example:
            >>> loader = YahooDataLoader(Path("data/raw"))
            >>> df = loader.download("RELIANCE.NS", "2020-01-01", "2024-12-31")
            >>> print(df.head())
        """
        # Validate inputs
        self._validate_date_range(start, end)

        # Check cache first
        if self.use_cache:
            cached_data = self._load_from_cache(symbol, start, end)
            if cached_data is not None:
                logger.info(f"Loaded {symbol} from cache ({len(cached_data)} rows)")
                return cached_data

        # Download from Yahoo Finance
        logger.info(f"Downloading {symbol} from {start} to {end}")
        df = self._download_with_retry(symbol, start, end, auto_adjust)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df

        # Save to cache
        if self.use_cache:
            self._save_to_cache(df, symbol, start, end)

        logger.info(f"Downloaded {symbol}: {len(df)} rows")
        return df

    def download_batch(
        self,
        symbols: List[str],
        start: str,
        end: str,
        auto_adjust: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols with progress tracking.

        Args:
            symbols: List of stock symbols
            start: Start date in "YYYY-MM-DD" format
            end: End date in "YYYY-MM-DD" format
            auto_adjust: If True, adjust for splits and dividends

        Returns:
            Dictionary mapping symbol to DataFrame

        Example:
            >>> symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
            >>> data = loader.download_batch(symbols, "2020-01-01", "2024-12-31")
            >>> print(f"Downloaded {len(data)} stocks")
        """
        logger.info(f"Batch downloading {len(symbols)} symbols")
        results: Dict[str, pd.DataFrame] = {}

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Progress: {i}/{len(symbols)} - {symbol}")
            try:
                df = self.download(symbol, start, end, auto_adjust)
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame on failure

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        successful = sum(1 for df in results.values() if not df.empty)
        logger.info(f"Batch download complete: {successful}/{len(symbols)} successful")

        return results

    def download_intraday(
        self,
        symbol: str,
        period: str = "5d",
        interval: str = "5m",
        auto_adjust: bool = True,
        filter_market_hours: bool = True,
    ) -> pd.DataFrame:
        """
        Download intraday data from Yahoo Finance with market hours filtering.

        Args:
            symbol: Stock ticker with .NS suffix (e.g., "RELIANCE.NS")
            period: Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"
            interval: Bar interval - "1m", "5m", "15m", "30m", "1h"
            auto_adjust: If True, adjust for splits and dividends
            filter_market_hours: If True, filter to Indian market hours (9:15-15:30 IST)

        Returns:
            DataFrame with intraday OHLCV data, DatetimeIndex

        Raises:
            ValueError: If period/interval combination is invalid
            RuntimeError: If download fails after retries

        Note:
            Yahoo Finance limits:
            - 1m interval: Max 7 days
            - 5m, 15m, 30m interval: Max 60 days
            - 1h interval: Max 730 days

            For longer history with 5m data, use download_intraday_range()

        Example:
            >>> loader = YahooDataLoader(Path("data/intraday"))
            >>> df = loader.download_intraday("RELIANCE.NS", period="60d", interval="5m")
            >>> print(f"Downloaded {len(df)} bars")
            >>> print(df.head())
        """
        # Validate interval and period combination
        self._validate_intraday_params(period, interval)

        # Check cache first
        if self.use_cache:
            cached_data = self._load_intraday_from_cache(symbol, period, interval)
            if cached_data is not None:
                logger.info(
                    f"Loaded intraday {symbol} ({interval}) from cache ({len(cached_data)} rows)"
                )
                return cached_data

        # Download from Yahoo Finance
        logger.info(f"Downloading intraday {symbol} - period={period}, interval={interval}")
        df = self._download_intraday_with_retry(symbol, period, interval, auto_adjust)

        if df.empty:
            logger.warning(f"No intraday data returned for {symbol}")
            return df

        # Filter to market hours (9:15 AM - 3:30 PM IST)
        if filter_market_hours:
            df = self._filter_market_hours(df)
            logger.info(f"Filtered to market hours: {len(df)} bars remaining")

        # Save to cache
        if self.use_cache:
            self._save_intraday_to_cache(df, symbol, period, interval)

        logger.info(f"Downloaded intraday {symbol}: {len(df)} bars")
        return df

    def download_intraday_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "5m",
        auto_adjust: bool = True,
        filter_market_hours: bool = True,
    ) -> pd.DataFrame:
        """
        Download intraday data for a specific date range by chunking requests.

        Use this for longer historical periods that exceed Yahoo Finance limits.
        For 5-minute data, downloads in 60-day chunks and concatenates.

        Args:
            symbol: Stock ticker with .NS suffix (e.g., "RELIANCE.NS")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            interval: Bar interval - "5m", "15m", "30m", "1h"
            auto_adjust: If True, adjust for splits and dividends
            filter_market_hours: If True, filter to Indian market hours

        Returns:
            DataFrame with concatenated intraday data

        Example:
            >>> loader = YahooDataLoader(Path("data/intraday"))
            >>> # Download 3 months of 5-minute data (exceeds 60-day limit)
            >>> df = loader.download_intraday_range(
            ...     "RELIANCE.NS",
            ...     "2024-07-01",
            ...     "2024-10-01",
            ...     interval="5m"
            ... )
            >>> print(f"Downloaded {len(df)} bars across {df.index.date.nunique()} days")
        """
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        if end_dt < start_dt:
            raise ValueError(f"End date ({end_date}) must be after start date ({start_date})")

        # Determine chunk size based on interval
        chunk_days = self._get_chunk_size(interval)

        # Download in chunks
        all_chunks = []
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)

            logger.info(
                f"Downloading chunk: {current_start.date()} to {current_end.date()}"
            )

            # Convert to period string (e.g., "60d")
            days_diff = (current_end - current_start).days
            period_str = f"{days_diff}d"

            try:
                # Download chunk
                chunk_df = self._download_intraday_with_retry(
                    symbol=symbol,
                    period=period_str,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    start=current_start.strftime("%Y-%m-%d"),
                    end=current_end.strftime("%Y-%m-%d"),
                )

                if not chunk_df.empty:
                    all_chunks.append(chunk_df)

            except Exception as e:
                logger.error(f"Failed to download chunk: {e}")

            # Move to next chunk
            current_start = current_end
            time.sleep(1)  # Rate limiting

        if not all_chunks:
            logger.warning(f"No data downloaded for {symbol} in range {start_date} to {end_date}")
            return pd.DataFrame()

        # Concatenate all chunks
        df = pd.concat(all_chunks, ignore_index=True)

        # Remove duplicates (overlapping chunks)
        if "Datetime" in df.columns:
            df = df.drop_duplicates(subset=["Datetime"], keep="first")
            df = df.sort_values("Datetime").reset_index(drop=True)

        # Filter to market hours
        if filter_market_hours:
            df = self._filter_market_hours(df)

        logger.info(
            f"Downloaded {len(df)} bars for {symbol} from {start_date} to {end_date}"
        )

        return df

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            symbol: If provided, clear cache for specific symbol only.
                   If None, clear entire cache.

        Example:
            >>> loader.clear_cache("RELIANCE.NS")  # Clear specific symbol
            >>> loader.clear_cache()  # Clear all cache
        """
        if symbol:
            # Clear specific symbol cache
            pattern = f"{symbol}_*.parquet"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"Deleted cache file: {cache_file.name}")

            # Remove from metadata
            self._metadata = {
                k: v for k, v in self._metadata.items() if not k.startswith(symbol)
            }
        else:
            # Clear entire cache
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            self._metadata = {}
            logger.info("Cleared entire cache")

        self._save_metadata()

    def _download_intraday_with_retry(
        self,
        symbol: str,
        period: str,
        interval: str,
        auto_adjust: bool,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Download intraday data with exponential backoff retry logic.

        Args:
            symbol: Stock symbol
            period: Period string (e.g., "5d", "60d")
            interval: Interval string (e.g., "5m", "1h")
            auto_adjust: Adjustment flag
            start: Optional start date (YYYY-MM-DD)
            end: Optional end date (YYYY-MM-DD)
            max_retries: Maximum number of retry attempts

        Returns:
            Downloaded DataFrame with Datetime index

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(1, max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)

                # Download with period or start/end
                if start and end:
                    df = ticker.history(
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=auto_adjust,
                    )
                else:
                    df = ticker.history(
                        period=period,
                        interval=interval,
                        auto_adjust=auto_adjust,
                    )

                # Reset index to make Datetime a column
                df = df.reset_index()

                # Rename 'Datetime' column if present (intraday data uses Datetime instead of Date)
                if "Datetime" in df.columns:
                    df = df.rename(columns={"Datetime": "Date"})

                return df

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed for intraday {symbol}: {e}"
                )

                if attempt < max_retries:
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Failed to download intraday {symbol} after {max_retries} attempts"
                    ) from e

        return pd.DataFrame()

    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to Indian market hours (9:15 AM - 3:30 PM IST).

        Args:
            df: DataFrame with 'Date' column (datetime)

        Returns:
            Filtered DataFrame containing only market hours bars
        """
        if df.empty or "Date" not in df.columns:
            return df

        # Ensure Date column is datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Extract time component
        df["_time"] = df["Date"].dt.time

        # Market hours: 9:15 AM - 3:30 PM (inclusive)
        market_open = pd.to_datetime("09:15:00").time()
        market_close = pd.to_datetime("15:30:00").time()

        # Filter to market hours
        mask = (df["_time"] >= market_open) & (df["_time"] <= market_close)
        filtered_df = df[mask].copy()

        # Drop temporary time column
        filtered_df = filtered_df.drop(columns=["_time"])

        return filtered_df

    def _load_intraday_from_cache(
        self,
        symbol: str,
        period: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """
        Load intraday data from cache.

        Args:
            symbol: Stock symbol
            period: Period string
            interval: Interval string

        Returns:
            Cached DataFrame or None if not found/invalid
        """
        cache_key = f"{symbol}_{period}_{interval}"
        cache_file = self.cache_dir / f"intraday_{cache_key}.parquet"

        if not cache_file.exists():
            return None

        if cache_key in self._metadata:
            try:
                df = pd.read_parquet(cache_file)
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                return df
            except Exception as e:
                logger.warning(f"Failed to load intraday cache for {symbol}: {e}")
                return None

        return None

    def _save_intraday_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        period: str,
        interval: str,
    ) -> None:
        """
        Save intraday DataFrame to cache.

        Args:
            df: DataFrame to cache
            symbol: Stock symbol
            period: Period string
            interval: Interval string
        """
        cache_key = f"{symbol}_{period}_{interval}"
        cache_file = self.cache_dir / f"intraday_{cache_key}.parquet"

        try:
            df.to_parquet(cache_file, index=False)

            # Update metadata
            self._metadata[cache_key] = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "type": "intraday",
                "cached_at": datetime.now().isoformat(),
                "rows": len(df),
            }
            self._save_metadata()

            logger.debug(f"Saved intraday {symbol} to cache: {cache_file.name}")

        except Exception as e:
            logger.error(f"Failed to save intraday cache for {symbol}: {e}")

    @staticmethod
    def _validate_intraday_params(period: str, interval: str) -> None:
        """
        Validate intraday period and interval parameters.

        Args:
            period: Period string
            interval: Interval string

        Raises:
            ValueError: If combination exceeds Yahoo Finance limits
        """
        # Define limits
        limits = {
            "1m": 7,  # Max 7 days for 1-minute data
            "5m": 60,  # Max 60 days for 5-minute data
            "15m": 60,  # Max 60 days
            "30m": 60,  # Max 60 days
            "1h": 730,  # Max 730 days (2 years)
        }

        if interval not in limits:
            raise ValueError(
                f"Invalid interval '{interval}'. Must be one of: {list(limits.keys())}"
            )

        # Parse period to days
        period_days = YahooDataLoader._parse_period_to_days(period)

        max_days = limits[interval]
        if period_days > max_days:
            raise ValueError(
                f"Period '{period}' ({period_days} days) exceeds Yahoo Finance limit "
                f"for interval '{interval}' (max {max_days} days). "
                f"Use download_intraday_range() for longer periods."
            )

    @staticmethod
    def _parse_period_to_days(period: str) -> int:
        """
        Parse period string to approximate number of days.

        Args:
            period: Period string (e.g., "5d", "3mo", "1y")

        Returns:
            Approximate number of days
        """
        period = period.lower()

        if period.endswith("d"):
            return int(period[:-1])
        elif period.endswith("mo"):
            return int(period[:-2]) * 30  # Approximate
        elif period.endswith("y"):
            return int(period[:-1]) * 365  # Approximate
        else:
            raise ValueError(f"Invalid period format: {period}")

    @staticmethod
    def _get_chunk_size(interval: str) -> int:
        """
        Get appropriate chunk size in days for the given interval.

        Args:
            interval: Interval string

        Returns:
            Chunk size in days
        """
        chunk_sizes = {
            "1m": 5,  # 5-day chunks for 1-minute data
            "5m": 55,  # 55-day chunks (below 60-day limit)
            "15m": 55,
            "30m": 55,
            "1h": 700,  # 700-day chunks (below 730-day limit)
        }

        return chunk_sizes.get(interval, 30)  # Default 30 days

    def _download_with_retry(
        self,
        symbol: str,
        start: str,
        end: str,
        auto_adjust: bool,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Download data with exponential backoff retry logic.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            auto_adjust: Adjustment flag
            max_retries: Maximum number of retry attempts

        Returns:
            Downloaded DataFrame

        Raises:
            RuntimeError: If all retries fail
        """
        for attempt in range(1, max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                )

                # Reset index to make Date a column
                df = df.reset_index()

                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed for {symbol}: {e}")

                if attempt < max_retries:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = 2**attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Failed to download {symbol} after {max_retries} attempts"
                    ) from e

        # This should never be reached, but for type safety
        return pd.DataFrame()

    def _load_from_cache(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and valid.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date

        Returns:
            Cached DataFrame or None if not found/invalid
        """
        cache_key = f"{symbol}_{start}_{end}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if not cache_file.exists():
            return None

        # Check metadata
        if cache_key in self._metadata:
            try:
                df = pd.read_parquet(cache_file)
                # Convert Date column to datetime if needed
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
                return None

        return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
    ) -> None:
        """
        Save DataFrame to cache.

        Args:
            df: DataFrame to cache
            symbol: Stock symbol
            start: Start date
            end: End date
        """
        cache_key = f"{symbol}_{start}_{end}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        try:
            df.to_parquet(cache_file, index=False)

            # Update metadata
            self._metadata[cache_key] = {
                "symbol": symbol,
                "start": start,
                "end": end,
                "cached_at": datetime.now().isoformat(),
                "rows": len(df),
            }
            self._save_metadata()

            logger.debug(f"Saved {symbol} to cache: {cache_file.name}")

        except Exception as e:
            logger.error(f"Failed to save cache for {symbol}: {e}")

    def _load_metadata(self) -> Dict:
        """Load cache metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to JSON file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    @staticmethod
    def _validate_date_range(start: str, end: str) -> None:
        """
        Validate date range inputs.

        Args:
            start: Start date string
            end: End date string

        Raises:
            ValueError: If dates are invalid or end < start
        """
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")

            if end_dt < start_dt:
                raise ValueError(f"End date ({end}) must be after start date ({start})")

        except ValueError as e:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD: {e}") from e
