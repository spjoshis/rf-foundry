"""Unit tests for data validation pipeline."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from tradebox.data.validation import (
    Category,
    DataValidator,
    Severity,
    ValidationConfig,
    ValidationIssue,
)


@pytest.fixture
def clean_data() -> pd.DataFrame:
    """Create clean, valid OHLCV data for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")  # Business days
    return pd.DataFrame({
        "Date": dates,
        "Open": [100.0 + i * 0.5 for i in range(100)],
        "High": [105.0 + i * 0.5 for i in range(100)],
        "Low": [95.0 + i * 0.5 for i in range(100)],
        "Close": [102.0 + i * 0.5 for i in range(100)],
        "Volume": [1000000 + i * 1000 for i in range(100)],
    })


@pytest.fixture
def validator() -> DataValidator:
    """Create DataValidator with default config."""
    return DataValidator()


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_validation_issue_creation(self) -> None:
        """Test creating a validation issue."""
        issue = ValidationIssue(
            severity=Severity.ERROR,
            category=Category.COMPLETENESS,
            message="Missing values",
            date=datetime(2020, 1, 1),
            details={"count": 5},
        )

        assert issue.severity == Severity.ERROR
        assert issue.category == Category.COMPLETENESS
        assert issue.message == "Missing values"
        assert issue.date == datetime(2020, 1, 1)
        assert issue.details["count"] == 5

    def test_validation_issue_string_representation(self) -> None:
        """Test string representation of issue."""
        issue = ValidationIssue(
            severity=Severity.WARNING,
            category=Category.ANOMALY,
            message="Price jump",
            date=datetime(2020, 1, 1),
        )

        assert "[WARNING]" in str(issue)
        assert "anomaly" in str(issue)
        assert "Price jump" in str(issue)


class TestDataValidatorInit:
    """Tests for DataValidator initialization."""

    def test_init_with_default_config(self) -> None:
        """Test initialization with default configuration."""
        validator = DataValidator()

        assert validator.config is not None
        assert validator.config.price_jump_threshold == 0.20
        assert validator.config.volume_spike_threshold == 10.0

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = ValidationConfig(price_jump_threshold=0.15)
        validator = DataValidator(config=config)

        assert validator.config.price_jump_threshold == 0.15


class TestCompletenessChecks:
    """Tests for completeness validation."""

    def test_clean_data_passes_completeness(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test that clean data passes completeness checks."""
        issues = validator.check_completeness(clean_data)

        assert len(issues) == 0

    def test_missing_required_columns(self, validator: DataValidator) -> None:
        """Test detection of missing required columns."""
        df = pd.DataFrame({"Date": [1, 2, 3], "Open": [100, 101, 102]})

        issues = validator.check_completeness(df)

        assert len(issues) > 0
        assert any("Missing required columns" in issue.message for issue in issues)
        assert any(issue.severity == Severity.ERROR for issue in issues)

    def test_missing_values_in_ohlc(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of missing values in OHLC columns."""
        df = clean_data.copy()
        df.loc[5:10, "Close"] = None

        issues = validator.check_completeness(df)

        assert len(issues) > 0
        assert any("Missing values in Close" in issue.message for issue in issues)

    def test_zero_volume_detection(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of zero volume days."""
        df = clean_data.copy()
        df.loc[5:10, "Volume"] = 0

        issues = validator.check_completeness(df)

        assert len(issues) > 0
        assert any("Zero volume" in issue.message for issue in issues)
        assert any(issue.severity == Severity.WARNING for issue in issues)


class TestConsistencyChecks:
    """Tests for consistency validation."""

    def test_clean_data_passes_consistency(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test that clean data passes consistency checks."""
        issues = validator.check_consistency(clean_data)

        assert len(issues) == 0

    def test_high_less_than_low(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection when High < Low."""
        df = clean_data.copy()
        df.loc[5, "High"] = 90.0  # Less than Low (95.0)
        df.loc[5, "Low"] = 95.0

        issues = validator.check_consistency(df)

        assert len(issues) > 0
        assert any("High < Low" in issue.message for issue in issues)
        assert any(issue.severity == Severity.ERROR for issue in issues)

    def test_high_less_than_open(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection when High < Open."""
        df = clean_data.copy()
        df.loc[5, "High"] = 95.0
        df.loc[5, "Open"] = 100.0

        issues = validator.check_consistency(df)

        assert len(issues) > 0
        assert any("High < Open" in issue.message for issue in issues)

    def test_low_greater_than_open(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection when Low > Open."""
        df = clean_data.copy()
        df.loc[5, "Low"] = 110.0
        df.loc[5, "Open"] = 100.0

        issues = validator.check_consistency(df)

        assert len(issues) > 0
        assert any("Low > Open" in issue.message for issue in issues)

    def test_negative_prices(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of negative prices."""
        df = clean_data.copy()
        df.loc[5, "Close"] = -10.0

        issues = validator.check_consistency(df)

        assert len(issues) > 0
        assert any("Non-positive Close prices" in issue.message for issue in issues)
        assert any(issue.severity == Severity.ERROR for issue in issues)

    def test_negative_volume(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of negative volume."""
        df = clean_data.copy()
        df.loc[5, "Volume"] = -1000

        issues = validator.check_consistency(df)

        assert len(issues) > 0
        assert any("Negative volume" in issue.message for issue in issues)


class TestAnomalyChecks:
    """Tests for anomaly detection."""

    def test_large_price_jump_detection(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of large price jumps."""
        df = clean_data.copy()
        # Create a 25% price jump
        df.loc[50, "Close"] = df.loc[49, "Close"] * 1.25

        issues = validator.check_anomalies(df)

        assert len(issues) > 0
        assert any("Large price jump" in issue.message for issue in issues)
        assert any(issue.severity == Severity.WARNING for issue in issues)

    def test_volume_spike_detection(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of volume spikes."""
        df = clean_data.copy()
        # Create a much larger volume spike (200x) to ensure it exceeds 10x the MA
        base_volume = df.loc[49, "Volume"]
        df.loc[50, "Volume"] = base_volume * 200

        issues = validator.check_anomalies(df)

        assert len(issues) > 0
        assert any("Volume spike" in issue.message for issue in issues)

    def test_stale_price_detection(self, validator: DataValidator) -> None:
        """Test detection of stale prices (no change for many days)."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0] * 50,
            "High": [105.0] * 50,
            "Low": [95.0] * 50,
            "Close": [100.0] * 50,  # Same price for all days
            "Volume": [1000000] * 50,
        })

        issues = validator.check_anomalies(df)

        assert len(issues) > 0
        assert any("Stale price" in issue.message for issue in issues)

    def test_consecutive_zero_volume(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of consecutive zero volume days."""
        df = clean_data.copy()
        # Set 6 consecutive days to zero volume
        df.loc[10:15, "Volume"] = 0

        issues = validator.check_anomalies(df)

        assert len(issues) > 0
        assert any("Consecutive zero volume" in issue.message for issue in issues)


class TestTemporalChecks:
    """Tests for temporal validation."""

    def test_clean_data_passes_temporal(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test that clean data passes temporal checks."""
        issues = validator.check_temporal(clean_data)

        # May have weekend gaps, but no errors
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_duplicate_dates_detection(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of duplicate dates."""
        df = clean_data.copy()
        # Duplicate a date
        df.loc[50, "Date"] = df.loc[49, "Date"]

        issues = validator.check_temporal(df)

        assert len(issues) > 0
        assert any("Duplicate dates" in issue.message for issue in issues)
        assert any(issue.severity == Severity.ERROR for issue in issues)

    def test_non_chronological_order(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test detection of non-chronological dates."""
        df = clean_data.copy()
        # Swap two dates
        df.loc[50, "Date"], df.loc[51, "Date"] = df.loc[51, "Date"], df.loc[50, "Date"]

        issues = validator.check_temporal(df)

        assert len(issues) > 0
        assert any("not in chronological order" in issue.message for issue in issues)
        assert any(issue.severity == Severity.ERROR for issue in issues)

    def test_large_gap_detection(self, validator: DataValidator) -> None:
        """Test detection of large gaps in data."""
        dates = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 15),  # 13-day gap
        ]
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [102, 103, 104],
            "Volume": [1000000, 1100000, 1200000],
        })

        issues = validator.check_temporal(df)

        assert len(issues) > 0
        assert any("Large gap in data" in issue.message for issue in issues)


class TestValidationReport:
    """Tests for overall validation and reporting."""

    def test_validate_clean_data(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test validation of clean data."""
        report = validator.validate(clean_data, "TEST.NS")

        assert report.is_valid is True
        assert report.symbol == "TEST.NS"
        assert report.summary["ERROR"] == 0

    def test_validate_with_errors(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test validation with errors."""
        df = clean_data.copy()
        df.loc[5, "Close"] = -10.0  # Error: negative price

        report = validator.validate(df, "TEST.NS")

        assert report.is_valid is False
        assert report.summary["ERROR"] > 0
        assert len(report.issues) > 0

    def test_validate_with_warnings_only(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test validation with warnings but no errors."""
        df = clean_data.copy()
        # Create a price jump (warning, not error) and adjust High to maintain consistency
        df.loc[50, "Close"] = df.loc[49, "Close"] * 1.25
        df.loc[50, "High"] = max(df.loc[50, "Close"], df.loc[50, "High"])  # Ensure High >= Close
        df.loc[50, "Low"] = min(df.loc[50, "Low"], df.loc[50, "Close"])  # Ensure Low <= Close

        report = validator.validate(df, "TEST.NS")

        assert report.is_valid is True  # Warnings don't invalidate
        assert report.summary["WARNING"] > 0
        assert report.summary["ERROR"] == 0

    def test_validation_report_to_dict(
        self,
        validator: DataValidator,
        clean_data: pd.DataFrame,
    ) -> None:
        """Test conversion of validation report to dictionary."""
        report = validator.validate(clean_data, "TEST.NS")

        report_dict = report.to_dict()

        assert "symbol" in report_dict
        assert "validated_at" in report_dict
        assert "is_valid" in report_dict
        assert "summary" in report_dict
        assert "issues" in report_dict
        assert report_dict["symbol"] == "TEST.NS"
