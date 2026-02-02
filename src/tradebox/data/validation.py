"""Data validation pipeline for quality assurance."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class Severity(str, Enum):
    """Validation issue severity levels."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class Category(str, Enum):
    """Validation issue categories."""

    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ANOMALY = "anomaly"
    TEMPORAL = "temporal"


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.

    Attributes:
        severity: Issue severity (ERROR, WARNING, INFO)
        category: Issue category (completeness, consistency, anomaly, temporal)
        message: Human-readable description
        date: Date associated with issue (if applicable)
        details: Additional context and metadata
    """

    severity: Severity
    category: Category
    message: str
    date: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return string representation."""
        date_str = f" on {self.date.date()}" if self.date else ""
        return f"[{self.severity.value}] {self.category.value}: {self.message}{date_str}"


@dataclass
class ValidationReport:
    """
    Comprehensive validation report.

    Attributes:
        symbol: Stock symbol validated
        validated_at: Timestamp of validation
        issues: List of all validation issues found
        is_valid: True if no ERROR-level issues
        summary: Count of issues by severity
    """

    symbol: str
    validated_at: datetime
    issues: List[ValidationIssue]
    is_valid: bool
    summary: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "symbol": self.symbol,
            "validated_at": self.validated_at.isoformat(),
            "is_valid": self.is_valid,
            "total_issues": len(self.issues),
            "summary": self.summary,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category.value,
                    "message": issue.message,
                    "date": issue.date.isoformat() if issue.date else None,
                    "details": issue.details,
                }
                for issue in self.issues
            ],
        }


@dataclass
class ValidationConfig:
    """
    Configuration for validation rules.

    Attributes:
        price_jump_threshold: Percentage threshold for suspicious price jumps
        volume_spike_threshold: Multiplier for volume spike detection
        zero_volume_days_threshold: Max consecutive days of zero volume
        stale_price_days_threshold: Max days with identical price
        max_gap_days: Maximum allowed gap between trading days
    """

    price_jump_threshold: float = 0.20  # 20%
    volume_spike_threshold: float = 10.0  # 10x average
    zero_volume_days_threshold: int = 5
    stale_price_days_threshold: int = 10
    max_gap_days: int = 7


class DataValidator:
    """
    Validates stock market data for quality issues.

    This validator performs comprehensive checks on OHLCV data:
    - Completeness: Missing values, gaps in data
    - Consistency: OHLC relationships, valid ranges
    - Anomalies: Suspicious price jumps, volume spikes
    - Temporal: Date continuity, chronological order

    Example:
        >>> validator = DataValidator()
        >>> report = validator.validate(df, "RELIANCE.NS")
        >>> if report.is_valid:
        >>>     print("Data passed validation")
        >>> else:
        >>>     print(f"Found {len(report.issues)} issues")
    """

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        """
        Initialize data validator.

        Args:
            config: Validation configuration. If None, uses defaults.
        """
        self.config = config or ValidationConfig()
        logger.info("DataValidator initialized with config: {}", self.config)

    def validate(self, df: pd.DataFrame, symbol: str) -> ValidationReport:
        """
        Perform comprehensive validation on DataFrame.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol being validated

        Returns:
            ValidationReport with all issues found

        Example:
            >>> df = pd.read_parquet("RELIANCE.parquet")
            >>> report = validator.validate(df, "RELIANCE.NS")
            >>> print(f"Valid: {report.is_valid}")
        """
        logger.info(f"Validating {symbol} - {len(df)} rows")
        issues: List[ValidationIssue] = []

        # Run all validation checks
        issues.extend(self.check_completeness(df))
        issues.extend(self.check_consistency(df))
        issues.extend(self.check_anomalies(df))
        issues.extend(self.check_temporal(df))

        # Calculate summary
        summary = {
            "ERROR": sum(1 for i in issues if i.severity == Severity.ERROR),
            "WARNING": sum(1 for i in issues if i.severity == Severity.WARNING),
            "INFO": sum(1 for i in issues if i.severity == Severity.INFO),
        }

        is_valid = summary["ERROR"] == 0

        report = ValidationReport(
            symbol=symbol,
            validated_at=datetime.now(),
            issues=issues,
            is_valid=is_valid,
            summary=summary,
        )

        logger.info(
            f"Validation complete for {symbol}: "
            f"{summary['ERROR']} errors, {summary['WARNING']} warnings, "
            f"{summary['INFO']} info"
        )

        return report

    def check_completeness(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Check for missing values and data completeness.

        Args:
            df: DataFrame to check

        Returns:
            List of completeness issues
        """
        issues: List[ValidationIssue] = []

        # Required columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.COMPLETENESS,
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing_columns": missing_cols},
                )
            )
            return issues  # Can't continue without required columns

        # Check for missing values in OHLCV columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category=Category.COMPLETENESS,
                        message=f"Missing values in {col}: {missing_count} ({missing_pct:.2f}%)",
                        details={"column": col, "missing_count": missing_count},
                    )
                )

        # Check for extended periods of zero volume
        if "Volume" in df.columns:
            zero_volume = df["Volume"] == 0
            if zero_volume.any():
                zero_count = zero_volume.sum()
                zero_pct = (zero_count / len(df)) * 100
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category=Category.COMPLETENESS,
                        message=f"Zero volume on {zero_count} days ({zero_pct:.2f}%)",
                        details={"zero_volume_days": zero_count},
                    )
                )

        return issues

    def check_consistency(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Check OHLC relationships and value ranges.

        Args:
            df: DataFrame to check

        Returns:
            List of consistency issues
        """
        issues: List[ValidationIssue] = []

        # Check High >= Open, Close, Low
        high_low_violations = df[df["High"] < df["Low"]]
        if not high_low_violations.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.CONSISTENCY,
                    message=f"High < Low on {len(high_low_violations)} days",
                    details={"violation_count": len(high_low_violations)},
                )
            )

        high_open_violations = df[df["High"] < df["Open"]]
        if not high_open_violations.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.CONSISTENCY,
                    message=f"High < Open on {len(high_open_violations)} days",
                    details={"violation_count": len(high_open_violations)},
                )
            )

        high_close_violations = df[df["High"] < df["Close"]]
        if not high_close_violations.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.CONSISTENCY,
                    message=f"High < Close on {len(high_close_violations)} days",
                    details={"violation_count": len(high_close_violations)},
                )
            )

        # Check Low <= Open, Close, High
        low_open_violations = df[df["Low"] > df["Open"]]
        if not low_open_violations.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.CONSISTENCY,
                    message=f"Low > Open on {len(low_open_violations)} days",
                    details={"violation_count": len(low_open_violations)},
                )
            )

        low_close_violations = df[df["Low"] > df["Close"]]
        if not low_close_violations.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.CONSISTENCY,
                    message=f"Low > Close on {len(low_close_violations)} days",
                    details={"violation_count": len(low_close_violations)},
                )
            )

        # Check for negative prices
        for col in ["Open", "High", "Low", "Close"]:
            negative = df[df[col] <= 0]
            if not negative.empty:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category=Category.CONSISTENCY,
                        message=f"Non-positive {col} prices on {len(negative)} days",
                        details={"column": col, "violation_count": len(negative)},
                    )
                )

        # Check for negative volume
        negative_volume = df[df["Volume"] < 0]
        if not negative_volume.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.CONSISTENCY,
                    message=f"Negative volume on {len(negative_volume)} days",
                    details={"violation_count": len(negative_volume)},
                )
            )

        return issues

    def check_anomalies(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Detect anomalies like price jumps and volume spikes.

        Args:
            df: DataFrame to check

        Returns:
            List of anomaly issues
        """
        issues: List[ValidationIssue] = []

        if len(df) < 2:
            return issues

        # Calculate daily returns
        df_copy = df.copy()
        df_copy["Return"] = df_copy["Close"].pct_change()

        # Check for large price jumps
        large_jumps = df_copy[df_copy["Return"].abs() > self.config.price_jump_threshold]
        for idx, row in large_jumps.iterrows():
            if pd.notna(row["Return"]):
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category=Category.ANOMALY,
                        message=f"Large price jump: {row['Return']*100:.2f}%",
                        date=pd.to_datetime(row["Date"]) if "Date" in row else None,
                        details={"return_pct": float(row["Return"] * 100)},
                    )
                )

        # Check for volume spikes
        if "Volume" in df.columns and len(df) >= 20:
            df_copy["Volume_MA20"] = df_copy["Volume"].rolling(20, min_periods=1).mean()
            df_copy["Volume_Ratio"] = df_copy["Volume"] / df_copy["Volume_MA20"]

            volume_spikes = df_copy[
                df_copy["Volume_Ratio"] > self.config.volume_spike_threshold
            ]
            if not volume_spikes.empty:
                issues.append(
                    ValidationIssue(
                        severity=Severity.INFO,
                        category=Category.ANOMALY,
                        message=f"Volume spikes detected on {len(volume_spikes)} days",
                        details={"spike_count": len(volume_spikes)},
                    )
                )

        # Check for stale prices (same price for many days)
        df_copy["Price_Change"] = df_copy["Close"].diff().abs()
        for i in range(len(df_copy) - self.config.stale_price_days_threshold):
            window = df_copy.iloc[i : i + self.config.stale_price_days_threshold]
            if (window["Price_Change"] == 0).all():
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category=Category.ANOMALY,
                        message=f"Stale price: {self.config.stale_price_days_threshold} days with no change",
                        date=pd.to_datetime(window.iloc[0]["Date"])
                        if "Date" in window.columns
                        else None,
                        details={"stale_days": self.config.stale_price_days_threshold},
                    )
                )
                break  # Only report once

        # Check for consecutive zero volume days
        if "Volume" in df.columns:
            zero_volume_mask = df_copy["Volume"] == 0
            consecutive_zeros = 0
            for is_zero in zero_volume_mask:
                if is_zero:
                    consecutive_zeros += 1
                    if consecutive_zeros >= self.config.zero_volume_days_threshold:
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                category=Category.ANOMALY,
                                message=f"Consecutive zero volume: {consecutive_zeros} days",
                                details={"consecutive_days": consecutive_zeros},
                            )
                        )
                        consecutive_zeros = 0  # Reset to avoid duplicate reports
                else:
                    consecutive_zeros = 0

        return issues

    def check_temporal(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Check temporal consistency and date continuity.

        Args:
            df: DataFrame to check

        Returns:
            List of temporal issues
        """
        issues: List[ValidationIssue] = []

        if "Date" not in df.columns or len(df) < 2:
            return issues

        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"])

        # Check for duplicate dates
        duplicates = df_copy[df_copy["Date"].duplicated()]
        if not duplicates.empty:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.TEMPORAL,
                    message=f"Duplicate dates found: {len(duplicates)} occurrences",
                    details={"duplicate_count": len(duplicates)},
                )
            )

        # Check chronological order
        if not df_copy["Date"].is_monotonic_increasing:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category=Category.TEMPORAL,
                    message="Dates are not in chronological order",
                )
            )

        # Check for large gaps
        df_copy = df_copy.sort_values("Date")
        df_copy["Date_Diff"] = df_copy["Date"].diff().dt.days

        large_gaps = df_copy[df_copy["Date_Diff"] > self.config.max_gap_days]
        for idx, row in large_gaps.iterrows():
            if pd.notna(row["Date_Diff"]):
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category=Category.TEMPORAL,
                        message=f"Large gap in data: {int(row['Date_Diff'])} days",
                        date=pd.to_datetime(row["Date"]),
                        details={"gap_days": int(row["Date_Diff"])},
                    )
                )

        return issues
