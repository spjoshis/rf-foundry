"""Fundamental data loader with point-in-time correctness."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger


# yfinance Ticker.info field mapping to our feature names
INFO_FIELD_MAPPING = {
    # Valuation ratios
    "PE_Ratio_Trailing": "trailingPE",
    "PE_Ratio_Forward": "forwardPE",
    "PB_Ratio": "priceToBook",
    "PS_Ratio": "priceToSalesTrailing12Months",
    # Profitability metrics
    "ROE": "returnOnEquity",
    "ROA": "returnOnAssets",
    "Profit_Margin": "profitMargins",
    "Operating_Margin": "operatingMargins",
    "Gross_Margin": "grossMargins",
    # Leverage ratios
    "Debt_to_Equity": "debtToEquity",
    "Current_Ratio": "currentRatio",
    "Interest_Coverage": "interestCoverage",
    # Growth metrics
    "Revenue_Growth": "revenueGrowth",
    "Earnings_Growth": "earningsGrowth",
}


@dataclass
class FundamentalConfig:
    """
    Configuration for fundamental data loading and extraction.

    Attributes:
        version: Feature version for cache invalidation
        enabled: If False, disables all fundamental feature extraction
        use_mock_data: If True, generate mock data instead of API calls
        announcement_delay_days: Days after quarter-end to assume data availability (PIT)
        valuation_enabled: Extract valuation ratios (PE, PB, PS)
        profitability_enabled: Extract profitability metrics (ROE, ROA, margins)
        leverage_enabled: Extract leverage metrics (D/E, current ratio)
        growth_enabled: Extract growth metrics (revenue/EPS growth)
        pe_ratio: Include trailing and forward PE
        pb_ratio: Include price-to-book
        ps_ratio: Include price-to-sales
        roe: Include return on equity
        roa: Include return on assets
        profit_margin: Include net profit margin
        operating_margin: Include operating margin
        gross_margin: Include gross margin
        debt_to_equity: Include debt-to-equity ratio
        current_ratio: Include current ratio
        interest_coverage: Include interest coverage ratio
        revenue_growth: Include YoY revenue growth
        eps_growth: Include YoY EPS growth
        book_value_growth: Include book value growth (calculated if possible)
        asset_growth: Include total asset growth (calculated if possible)
    """

    version: str = "1.0"
    enabled: bool = True
    use_mock_data: bool = False
    announcement_delay_days: int = 45

    # Feature groups
    valuation_enabled: bool = True
    profitability_enabled: bool = True
    leverage_enabled: bool = True
    growth_enabled: bool = True

    # Valuation ratios (3 features)
    pe_ratio: bool = True
    pb_ratio: bool = True
    ps_ratio: bool = True

    # Profitability metrics (5 features)
    roe: bool = True
    roa: bool = True
    profit_margin: bool = True
    operating_margin: bool = True
    gross_margin: bool = True

    # Leverage ratios (3 features)
    debt_to_equity: bool = True
    current_ratio: bool = True
    interest_coverage: bool = True

    # Growth metrics (4 features)
    revenue_growth: bool = True
    eps_growth: bool = True
    book_value_growth: bool = True
    asset_growth: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.announcement_delay_days < 0:
            raise ValueError(
                f"announcement_delay_days must be >= 0, got {self.announcement_delay_days}"
            )

        if self.announcement_delay_days > 180:
            logger.warning(
                f"announcement_delay_days is very large ({self.announcement_delay_days}). "
                "Typical values are 30-60 days."
            )

    def get_enabled_features(self) -> List[str]:
        """
        Return list of enabled feature names.

        Returns:
            List of feature column names that are enabled
        """
        # If fundamentals globally disabled, return empty list
        if not self.enabled:
            return []

        features = []

        if self.valuation_enabled:
            if self.pe_ratio:
                features.extend(["PE_Ratio_Trailing", "PE_Ratio_Forward"])
            if self.pb_ratio:
                features.append("PB_Ratio")
            if self.ps_ratio:
                features.append("PS_Ratio")

        if self.profitability_enabled:
            if self.roe:
                features.append("ROE")
            if self.roa:
                features.append("ROA")
            if self.profit_margin:
                features.append("Profit_Margin")
            if self.operating_margin:
                features.append("Operating_Margin")
            if self.gross_margin:
                features.append("Gross_Margin")

        if self.leverage_enabled:
            if self.debt_to_equity:
                features.append("Debt_to_Equity")
            if self.current_ratio:
                features.append("Current_Ratio")
            if self.interest_coverage:
                features.append("Interest_Coverage")

        if self.growth_enabled:
            if self.revenue_growth:
                features.append("Revenue_Growth")
            if self.eps_growth:
                features.append("EPS_Growth")
            if self.book_value_growth:
                features.append("Book_Value_Growth")
            if self.asset_growth:
                features.append("Asset_Growth")

        return features


class FundamentalDataLoader:
    """
    Download and cache fundamental data from Yahoo Finance with point-in-time correctness.

    This loader handles:
    - Quarterly financial data extraction from yfinance
    - Current snapshot metrics via Ticker.info
    - Point-in-time correctness with configurable announcement delay
    - Mock data support for testing
    - Intelligent caching using Parquet format
    - Retry logic and error handling

    IMPORTANT: yfinance only provides ~4 quarters of historical fundamental data.
    For longer backtests (10+ years), use mock data or integrate Financial Modeling Prep API.

    Attributes:
        config: Configuration for fundamental extraction
        cache_dir: Directory for storing cached data
        use_cache: Whether to use cached data when available
        metadata_file: Path to cache metadata JSON file

    Example:
        >>> config = FundamentalConfig(use_mock_data=True)
        >>> loader = FundamentalDataLoader(config, Path("data/fundamental_cache"))
        >>> df = loader.download("RELIANCE.NS", "2020-01-01", "2024-12-31")
        >>> print(f"Loaded {len(df)} quarters with {len(df.columns)} features")
    """

    def __init__(
        self,
        config: FundamentalConfig,
        cache_dir: Path,
        use_cache: bool = True,
    ) -> None:
        """
        Initialize the fundamental data loader.

        Args:
            config: Configuration for fundamental extraction
            cache_dir: Directory path for caching downloaded data
            use_cache: If True, load from cache when available

        Raises:
            ValueError: If cache_dir is not a valid directory path
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.metadata_file = self.cache_dir / ".cache_metadata.json"

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize metadata
        self._metadata = self._load_metadata()

        logger.info(
            f"FundamentalDataLoader initialized (version={config.version}, "
            f"use_mock_data={config.use_mock_data}, cache_dir={self.cache_dir})"
        )

    def download(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Download fundamental data for a single symbol.

        Returns quarterly fundamental data with point-in-time adjustment.
        If use_mock_data=True, generates realistic mock data instead of API calls.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE.NS" for NSE)
            start: Start date in "YYYY-MM-DD" format
            end: End date in "YYYY-MM-DD" format

        Returns:
            DataFrame with columns:
            - Date: Quarter-end date
            - PE_Ratio_Trailing: Trailing P/E ratio
            - PE_Ratio_Forward: Forward P/E ratio
            - PB_Ratio: Price-to-Book ratio
            - PS_Ratio: Price-to-Sales ratio
            - ROE: Return on Equity (as decimal, e.g., 0.15 = 15%)
            - ROA: Return on Assets
            - Profit_Margin: Net profit margin
            - Operating_Margin: Operating margin
            - Gross_Margin: Gross margin
            - Debt_to_Equity: Debt-to-Equity ratio
            - Current_Ratio: Current ratio
            - Interest_Coverage: Interest coverage ratio
            - Revenue_Growth: YoY revenue growth (as decimal)
            - EPS_Growth: YoY EPS growth
            - Book_Value_Growth: YoY book value growth
            - Asset_Growth: YoY total asset growth
            - Quarter_End: Original quarter-end date
            - Announcement_Date: Estimated announcement date (quarter_end + delay)

        Raises:
            ValueError: If date range is invalid
            RuntimeError: If download fails after retries

        Example:
            >>> loader = FundamentalDataLoader(config, Path("data/fundamental_cache"))
            >>> df = loader.download("RELIANCE.NS", "2020-01-01", "2024-12-31")
            >>> print(df.head())
        """
        # Validate inputs
        self._validate_date_range(start, end)

        # Check cache first
        if self.use_cache:
            cached_data = self._load_from_cache(symbol, start, end)
            if cached_data is not None:
                logger.info(f"Loaded {symbol} from cache ({len(cached_data)} quarters)")
                return cached_data

        # Generate mock data or download from yfinance
        if self.config.use_mock_data:
            logger.info(f"Generating mock fundamental data for {symbol}")
            df = self._generate_mock_data(symbol, start, end)
        else:
            logger.info(f"Downloading fundamental data for {symbol} from {start} to {end}")
            df = self._download_with_retry(symbol, start, end)

        if df.empty:
            logger.warning(f"No fundamental data returned for {symbol}")
            return df

        # Apply point-in-time adjustment (STORY-015)
        df = self._apply_point_in_time_adjustment(df, self.config.announcement_delay_days)

        # Save to cache
        if self.use_cache and not df.empty:
            self._save_to_cache(df, symbol, start, end)

        logger.info(
            f"Downloaded {len(df)} quarters for {symbol} "
            f"({len(self.config.get_enabled_features())} features)"
        )

        return df

    def download_batch(
        self,
        symbols: List[str],
        start: str,
        end: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start: Start date in "YYYY-MM-DD" format
            end: End date in "YYYY-MM-DD" format

        Returns:
            Dictionary mapping symbol to DataFrame

        Example:
            >>> symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
            >>> data_dict = loader.download_batch(symbols, "2020-01-01", "2024-12-31")
            >>> print(f"Downloaded {len(data_dict)} symbols")
        """
        results = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"Downloading {symbol} ({i+1}/{len(symbols)})")

            try:
                df = self.download(symbol, start, end)
                if not df.empty:
                    results[symbol] = df
                else:
                    logger.warning(f"No data for {symbol}, skipping")

            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                continue

            # Rate limiting to avoid overwhelming yfinance API
            if i < len(symbols) - 1 and not self.config.use_mock_data:
                time.sleep(0.5)  # 0.5 second delay between requests

        logger.info(f"Successfully downloaded {len(results)}/{len(symbols)} symbols")
        return results

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached fundamental data.

        Args:
            symbol: If provided, clear only this symbol's cache.
                   If None, clear all cached data.
        """
        if symbol is None:
            # Clear all cache
            for file in self.cache_dir.glob("*.parquet"):
                file.unlink()
            self._metadata = {}
            self._save_metadata()
            logger.info("Cleared all fundamental data cache")
        else:
            # Clear specific symbol
            pattern = f"{symbol}_*.parquet"
            for file in self.cache_dir.glob(pattern):
                file.unlink()
            # Remove from metadata
            self._metadata = {k: v for k, v in self._metadata.items() if not k.startswith(symbol)}
            self._save_metadata()
            logger.info(f"Cleared cache for {symbol}")

    def _generate_mock_data(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Generate realistic mock fundamental data for testing.

        Creates quarterly data with:
        - Realistic ranges (PE: 5-50, ROE: 5-35%, etc.)
        - Temporal consistency (no sudden jumps)
        - All configured features

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date

        Returns:
            DataFrame with mock fundamental data
        """
        # Generate quarterly dates
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        # Create quarter-end dates (Mar 31, Jun 30, Sep 30, Dec 31)
        dates = pd.date_range(start=start_date, end=end_date, freq="QE")

        if len(dates) == 0:
            return pd.DataFrame()

        n_quarters = len(dates)
        logger.debug(f"Generating mock data for {n_quarters} quarters")

        # Generate base values with small random walk for temporal consistency
        np.random.seed(hash(symbol) % (2**32))  # Reproducible per symbol

        # Initialize data dictionary
        data = {"Date": dates}

        enabled_features = self.config.get_enabled_features()

        # Generate mock values for each feature with realistic ranges
        for feature in enabled_features:
            if feature in ["PE_Ratio_Trailing", "PE_Ratio_Forward"]:
                # PE ratio: 10-40 typical range
                base = 20.0 + np.random.randn() * 5.0
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.5)
                values = np.clip(values, 5.0, 100.0)  # Clip to reasonable range

            elif feature == "PB_Ratio":
                # Price-to-Book: 1.5-5.0 typical range
                base = 2.5 + np.random.randn() * 0.5
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.1)
                values = np.clip(values, 0.5, 10.0)

            elif feature == "PS_Ratio":
                # Price-to-Sales: 0.5-3.0 typical range
                base = 1.5 + np.random.randn() * 0.3
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.05)
                values = np.clip(values, 0.2, 5.0)

            elif feature in ["ROE", "ROA"]:
                # Returns: 5-30% typical range (stored as decimal)
                base = 0.15 + np.random.randn() * 0.03 if feature == "ROE" else 0.08 + np.random.randn() * 0.02
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.005)
                values = np.clip(values, -0.10, 0.50)  # -10% to 50%

            elif feature in ["Profit_Margin", "Operating_Margin", "Gross_Margin"]:
                # Margins: typically positive, increasing order
                if feature == "Gross_Margin":
                    base = 0.40 + np.random.randn() * 0.05
                elif feature == "Operating_Margin":
                    base = 0.20 + np.random.randn() * 0.03
                else:  # Profit margin (lowest)
                    base = 0.12 + np.random.randn() * 0.02

                values = base + np.cumsum(np.random.randn(n_quarters) * 0.005)
                values = np.clip(values, -0.05, 0.80)

            elif feature == "Debt_to_Equity":
                # Debt-to-Equity: 0.2-2.0 typical range
                base = 0.8 + np.random.randn() * 0.2
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.05)
                values = np.clip(values, 0.0, 5.0)

            elif feature == "Current_Ratio":
                # Current ratio: 1.0-3.0 typical range
                base = 1.5 + np.random.randn() * 0.2
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.05)
                values = np.clip(values, 0.5, 5.0)

            elif feature == "Interest_Coverage":
                # Interest coverage: 2-10 typical range
                base = 5.0 + np.random.randn() * 1.0
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.2)
                values = np.clip(values, 0.5, 20.0)

            elif feature in ["Revenue_Growth", "EPS_Growth", "Book_Value_Growth", "Asset_Growth"]:
                # Growth rates: -5% to 25% typical range (stored as decimal)
                base = 0.10 + np.random.randn() * 0.03
                values = base + np.cumsum(np.random.randn(n_quarters) * 0.02)
                values = np.clip(values, -0.20, 0.50)  # -20% to 50%

            else:
                # Fallback: generate random values
                logger.warning(f"Unknown feature {feature}, generating random values")
                values = np.random.randn(n_quarters)

            data[feature] = values

        df = pd.DataFrame(data)
        df = df.set_index("Date")

        logger.debug(f"Generated mock data with {len(df)} quarters and {len(enabled_features)} features")

        return df

    def _download_with_retry(
        self,
        symbol: str,
        start: str,
        end: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Download fundamental data with exponential backoff retry logic.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date
            max_retries: Maximum number of retry attempts

        Returns:
            DataFrame with fundamental data

        Raises:
            RuntimeError: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)

                # Extract from Ticker.info (current snapshot)
                info_data = self._extract_from_info(ticker)

                # Extract from financial statements (quarterly historical)
                financials_data = self._extract_from_financials(ticker)

                # Combine both sources
                if financials_data.empty:
                    # Only current snapshot available - create single-row DataFrame
                    if info_data:
                        df = pd.DataFrame([info_data])
                        df["Date"] = pd.Timestamp.now().normalize()
                        df = df.set_index("Date")
                    else:
                        df = pd.DataFrame()
                else:
                    df = financials_data

                return df

            except Exception as e:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {symbol}")
                    raise RuntimeError(
                        f"Failed to download fundamental data for {symbol} after {max_retries} attempts"
                    ) from e

        return pd.DataFrame()

    def _extract_from_info(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Extract fundamental ratios from Ticker.info.

        Handles:
        - Missing fields (return None)
        - Data type conversion
        - Error handling for API changes

        Args:
            ticker: yfinance Ticker object

        Returns:
            Dictionary with feature names as keys
        """
        extracted = {}

        try:
            info = ticker.info

            # Map yfinance fields to our feature names
            for our_name, yf_field in INFO_FIELD_MAPPING.items():
                # Check if feature is enabled in config
                if our_name not in self.config.get_enabled_features():
                    continue

                # Extract value, handle missing fields
                value = info.get(yf_field)

                # Convert percentage values to decimals if needed
                if value is not None and our_name in [
                    "ROE", "ROA", "Profit_Margin", "Operating_Margin", "Gross_Margin",
                    "Revenue_Growth", "Earnings_Growth"
                ]:
                    # yfinance returns these as decimals already (0.15 = 15%)
                    pass

                extracted[our_name] = value

            logger.debug(f"Extracted {len(extracted)} features from Ticker.info")

        except Exception as e:
            logger.error(f"Error extracting from Ticker.info: {e}")

        return extracted

    def _extract_from_financials(self, ticker: yf.Ticker) -> pd.DataFrame:
        """
        Extract quarterly fundamental data from financial statements.

        Combines:
        - ticker.quarterly_financials (income statement)
        - ticker.quarterly_balance_sheet (balance sheet)
        - ticker.quarterly_cashflow (cash flow)

        IMPORTANT: yfinance only provides ~4 quarters of data.

        Args:
            ticker: yfinance Ticker object

        Returns:
            DataFrame with quarterly data, index is Date
        """
        try:
            # Access financial statements
            qtr_financials = ticker.quarterly_financials
            qtr_balance_sheet = ticker.quarterly_balance_sheet
            qtr_cashflow = ticker.quarterly_cashflow

            if qtr_financials.empty:
                logger.warning("No quarterly financial data available")
                return pd.DataFrame()

            # Transpose to have dates as rows
            qtr_financials = qtr_financials.T
            qtr_balance_sheet = qtr_balance_sheet.T if not qtr_balance_sheet.empty else pd.DataFrame()
            qtr_cashflow = qtr_cashflow.T if not qtr_cashflow.empty else pd.DataFrame()

            # Calculate derived ratios
            df = self._calculate_derived_ratios(qtr_financials, qtr_balance_sheet, qtr_cashflow)

            logger.debug(f"Extracted {len(df)} quarters from financial statements")

            return df

        except Exception as e:
            logger.error(f"Error extracting from financial statements: {e}")
            return pd.DataFrame()

    def _calculate_derived_ratios(
        self,
        financials: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate fundamental ratios from raw financial statements.

        Examples:
        - ROE = Net Income / Shareholder Equity
        - ROA = Net Income / Total Assets
        - Debt/Equity = Total Debt / Shareholder Equity
        - Current Ratio = Current Assets / Current Liabilities
        - Profit Margin = Net Income / Revenue

        Args:
            financials: Income statement (transposed)
            balance_sheet: Balance sheet (transposed)
            cashflow: Cash flow statement (transposed)

        Returns:
            DataFrame with calculated ratios
        """
        if financials.empty:
            return pd.DataFrame()

        result = pd.DataFrame(index=financials.index)
        enabled_features = self.config.get_enabled_features()

        try:
            # Extract common fields from financial statements
            # Note: yfinance uses various field names, need to handle multiple possibilities

            # Income Statement fields
            net_income = self._get_field(financials, ["Net Income", "Net Income Common Stockholders"])
            total_revenue = self._get_field(financials, ["Total Revenue", "Revenue"])
            operating_income = self._get_field(financials, ["Operating Income", "EBIT"])
            gross_profit = self._get_field(financials, ["Gross Profit"])

            # Balance Sheet fields
            total_assets = self._get_field(balance_sheet, ["Total Assets"])
            shareholder_equity = self._get_field(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
            total_debt = self._get_field(balance_sheet, ["Total Debt", "Long Term Debt"])
            current_assets = self._get_field(balance_sheet, ["Current Assets"])
            current_liabilities = self._get_field(balance_sheet, ["Current Liabilities"])

            # Calculate ratios based on available data

            # Profitability Ratios
            if "ROE" in enabled_features and net_income is not None and shareholder_equity is not None:
                # ROE = Net Income / Shareholder Equity
                result["ROE"] = self._safe_divide(net_income, shareholder_equity)

            if "ROA" in enabled_features and net_income is not None and total_assets is not None:
                # ROA = Net Income / Total Assets
                result["ROA"] = self._safe_divide(net_income, total_assets)

            if "Profit_Margin" in enabled_features and net_income is not None and total_revenue is not None:
                # Profit Margin = Net Income / Revenue
                result["Profit_Margin"] = self._safe_divide(net_income, total_revenue)

            if "Operating_Margin" in enabled_features and operating_income is not None and total_revenue is not None:
                # Operating Margin = Operating Income / Revenue
                result["Operating_Margin"] = self._safe_divide(operating_income, total_revenue)

            if "Gross_Margin" in enabled_features and gross_profit is not None and total_revenue is not None:
                # Gross Margin = Gross Profit / Revenue
                result["Gross_Margin"] = self._safe_divide(gross_profit, total_revenue)

            # Leverage Ratios
            if "Debt_to_Equity" in enabled_features and total_debt is not None and shareholder_equity is not None:
                # Debt-to-Equity = Total Debt / Shareholder Equity
                result["Debt_to_Equity"] = self._safe_divide(total_debt, shareholder_equity) * 100  # As percentage

            if "Current_Ratio" in enabled_features and current_assets is not None and current_liabilities is not None:
                # Current Ratio = Current Assets / Current Liabilities
                result["Current_Ratio"] = self._safe_divide(current_assets, current_liabilities)

            # Growth Metrics (YoY growth)
            if "Revenue_Growth" in enabled_features and total_revenue is not None:
                result["Revenue_Growth"] = total_revenue.pct_change(periods=-1)  # YoY growth

            if "EPS_Growth" in enabled_features and net_income is not None:
                # Simplified: using net income growth as proxy
                result["EPS_Growth"] = net_income.pct_change(periods=-1)

            if "Book_Value_Growth" in enabled_features and shareholder_equity is not None:
                result["Book_Value_Growth"] = shareholder_equity.pct_change(periods=-1)

            if "Asset_Growth" in enabled_features and total_assets is not None:
                result["Asset_Growth"] = total_assets.pct_change(periods=-1)

            # Note: PE, PB, PS ratios require stock price, so they come from Ticker.info
            # Interest Coverage requires EBIT and interest expense - not always available

            calculated = [col for col in result.columns if result[col].notna().any()]
            logger.debug(f"Calculated {len(calculated)} ratios from financial statements: {calculated}")

        except Exception as e:
            logger.error(f"Error calculating derived ratios: {e}")

        return result

    def _get_field(self, df: pd.DataFrame, field_names: List[str]) -> Optional[pd.Series]:
        """
        Get a field from DataFrame, trying multiple possible names.

        Args:
            df: DataFrame to search
            field_names: List of possible field names to try

        Returns:
            Series if found, None otherwise
        """
        if df.empty:
            return None

        for name in field_names:
            if name in df.columns:
                return df[name]

        return None

    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """
        Safely divide two series, handling division by zero.

        Args:
            numerator: Numerator series
            denominator: Denominator series

        Returns:
            Series with division result, NaN where denominator is zero
        """
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    def _apply_point_in_time_adjustment(
        self,
        df: pd.DataFrame,
        announcement_delay_days: int = 45,
    ) -> pd.DataFrame:
        """
        Apply point-in-time correction to fundamental data.

        CRITICAL for STORY-015: This prevents look-ahead bias by ensuring
        fundamentals are only available AFTER they would have been announced.

        Algorithm:
        1. Extract quarter-end date from financial statement timestamp
        2. Add announcement_delay_days (default: 45)
        3. Set this as 'Announcement_Date'
        4. Only make data available from this date forward

        Example:
            Quarter ending 2024-03-31
            + 45 days delay
            = Available from 2024-05-15 onwards

        This prevents using Q1 2024 fundamentals in March 2024 trading decisions.

        Args:
            df: DataFrame with quarterly fundamentals, index is Date
            announcement_delay_days: Days after quarter-end to assume availability

        Returns:
            DataFrame with added 'Quarter_End' and 'Announcement_Date' columns
        """
        if df.empty:
            return df

        df = df.copy()

        # Store original quarter-end date
        df["Quarter_End"] = df.index

        # Calculate announcement date
        df["Announcement_Date"] = df["Quarter_End"] + pd.Timedelta(days=announcement_delay_days)

        logger.debug(
            f"Applied PIT adjustment: {announcement_delay_days} days delay. "
            f"Latest quarter {df['Quarter_End'].max()} available from {df['Announcement_Date'].max()}"
        )

        return df

    def _validate_date_range(self, start: str, end: str) -> None:
        """
        Validate that date range is valid.

        Args:
            start: Start date string
            end: End date string

        Raises:
            ValueError: If date range is invalid
        """
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        if start_date >= end_date:
            raise ValueError(f"Start date {start} must be before end date {end}")

    def _load_from_cache(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """
        Load fundamental data from cache.

        Args:
            symbol: Stock symbol
            start: Start date
            end: End date

        Returns:
            Cached DataFrame if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(symbol, start, end)
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if not cache_file.exists():
            return None

        try:
            df = pd.read_parquet(cache_file)

            # Validate cache
            if df.empty:
                logger.warning(f"Cache file empty for {symbol}")
                return None

            # Check if cache is for correct version
            cached_metadata = self._metadata.get(cache_key, {})
            if cached_metadata.get("version") != self.config.version:
                logger.info(f"Cache version mismatch for {symbol}, will re-download")
                return None

            return df

        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
    ) -> None:
        """
        Save fundamental data to cache.

        Args:
            df: DataFrame to cache
            symbol: Stock symbol
            start: Start date
            end: End date
        """
        cache_key = self._get_cache_key(symbol, start, end)
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        try:
            # Save data as Parquet
            df.to_parquet(cache_file)

            # Update metadata
            self._metadata[cache_key] = {
                "symbol": symbol,
                "start": start,
                "end": end,
                "cached_at": datetime.now().isoformat(),
                "version": self.config.version,
                "n_quarters": len(df),
            }
            self._save_metadata()

            logger.debug(f"Saved {symbol} to cache ({len(df)} quarters)")

        except Exception as e:
            logger.error(f"Failed to save cache for {symbol}: {e}")

    def _get_cache_key(self, symbol: str, start: str, end: str) -> str:
        """Generate cache key for a symbol and date range."""
        return f"{symbol}_{start}_{end}"

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to JSON file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
