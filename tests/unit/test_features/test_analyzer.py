"""Unit tests for feature analyzer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tradebox.features.analyzer import FeatureAnalyzer


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    # Create features with known correlations
    data = {
        "feature_a": np.random.randn(n_samples),
        "feature_b": np.random.randn(n_samples),
        "feature_c": np.random.randn(n_samples),
    }

    # Make feature_b highly correlated with feature_a
    data["feature_b"] = data["feature_a"] * 0.98 + np.random.randn(n_samples) * 0.02

    # Make feature_c independent
    data["feature_c"] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def sample_target(sample_features):
    """Create sample target variable."""
    # Target is mostly related to feature_a
    return (
        sample_features["feature_a"] * 2
        + sample_features["feature_c"] * 0.5
        + np.random.randn(len(sample_features)) * 0.1
    )


@pytest.fixture
def analyzer():
    """Create FeatureAnalyzer instance."""
    return FeatureAnalyzer()


def test_analyzer_initialization(analyzer):
    """Test FeatureAnalyzer initialization."""
    assert analyzer is not None
    assert isinstance(analyzer, FeatureAnalyzer)


def test_compute_correlations(analyzer, sample_features):
    """Test correlation matrix computation."""
    corr_matrix = analyzer.compute_correlations(sample_features)

    # Check shape and type
    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (3, 3)

    # Check diagonal is 1.0 (self-correlation)
    assert np.allclose(np.diag(corr_matrix.values), 1.0)

    # Check symmetry
    assert np.allclose(corr_matrix.values, corr_matrix.values.T)

    # Check feature_a and feature_b are highly correlated
    assert corr_matrix.loc["feature_a", "feature_b"] > 0.9


def test_find_high_correlations_with_threshold(analyzer, sample_features):
    """Test finding high correlations with custom threshold."""
    corr_matrix = analyzer.compute_correlations(sample_features)
    high_corr = analyzer.find_high_correlations(corr_matrix, threshold=0.9)

    # Should find feature_a <-> feature_b correlation
    assert len(high_corr) >= 1

    # Check structure
    assert all(len(pair) == 3 for pair in high_corr)  # (f1, f2, corr_val)

    # Check sorting (highest correlation first)
    if len(high_corr) > 1:
        abs_corrs = [abs(pair[2]) for pair in high_corr]
        assert abs_corrs == sorted(abs_corrs, reverse=True)


def test_find_high_correlations_no_matches(analyzer):
    """Test find_high_correlations with uncorrelated features."""
    # Create uncorrelated features
    np.random.seed(42)
    features = pd.DataFrame(
        {
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "f3": np.random.randn(100),
        }
    )

    corr_matrix = analyzer.compute_correlations(features)
    high_corr = analyzer.find_high_correlations(corr_matrix, threshold=0.99)

    # Should find no high correlations
    assert len(high_corr) == 0


def test_compute_feature_importance(analyzer, sample_features, sample_target):
    """Test feature importance computation."""
    importance = analyzer.compute_feature_importance(
        sample_features, sample_target, n_estimators=50
    )

    # Check type and shape
    assert isinstance(importance, pd.Series)
    assert len(importance) == len(sample_features.columns)

    # Check values are between 0 and 1
    assert all(0 <= val <= 1 for val in importance.values)

    # Check sum is approximately 1
    assert 0.99 <= importance.sum() <= 1.01

    # Check sorting (descending)
    assert list(importance.values) == sorted(importance.values, reverse=True)

    # feature_a should be most important (strongest relationship with target)
    assert importance.idxmax() == "feature_a"


def test_compute_feature_importance_with_nans(analyzer):
    """Test feature importance handles NaN values correctly."""
    np.random.seed(42)
    features = pd.DataFrame(
        {
            "f1": [1, 2, np.nan, 4, 5],
            "f2": [1, np.nan, 3, 4, 5],
            "f3": [1, 2, 3, 4, 5],
        }
    )
    target = pd.Series([1, 2, 3, 4, 5])

    importance = analyzer.compute_feature_importance(features, target)

    # Should succeed by dropping NaN rows
    assert isinstance(importance, pd.Series)
    assert len(importance) == 3


def test_compute_feature_importance_length_mismatch(analyzer, sample_features):
    """Test feature importance raises error on length mismatch."""
    target = pd.Series([1, 2, 3])  # Wrong length

    with pytest.raises(ValueError, match="different lengths"):
        analyzer.compute_feature_importance(sample_features, target)


def test_compute_feature_importance_all_nans(analyzer):
    """Test feature importance raises error when all samples are NaN."""
    features = pd.DataFrame({"f1": [np.nan, np.nan], "f2": [np.nan, np.nan]})
    target = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError, match="No valid samples"):
        analyzer.compute_feature_importance(features, target)


def test_plot_correlation_matrix_no_output(analyzer, sample_features):
    """Test correlation matrix plotting without saving."""
    corr_matrix = analyzer.compute_correlations(sample_features)

    # Should not raise error
    analyzer.plot_correlation_matrix(corr_matrix)


def test_plot_correlation_matrix_with_output(analyzer, sample_features, tmp_path):
    """Test correlation matrix plotting with file output."""
    corr_matrix = analyzer.compute_correlations(sample_features)
    output_path = tmp_path / "corr_matrix.png"

    analyzer.plot_correlation_matrix(corr_matrix, output_path=output_path)

    # Check file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_feature_importance_no_output(analyzer, sample_features, sample_target):
    """Test feature importance plotting without saving."""
    importance = analyzer.compute_feature_importance(
        sample_features, sample_target, n_estimators=50
    )

    # Should not raise error
    analyzer.plot_feature_importance(importance, top_n=3)


def test_plot_feature_importance_with_output(
    analyzer, sample_features, sample_target, tmp_path
):
    """Test feature importance plotting with file output."""
    importance = analyzer.compute_feature_importance(
        sample_features, sample_target, n_estimators=50
    )
    output_path = tmp_path / "importance.png"

    analyzer.plot_feature_importance(importance, top_n=3, output_path=output_path)

    # Check file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_report(analyzer, sample_features, sample_target, tmp_path):
    """Test comprehensive report generation."""
    output_dir = tmp_path / "report"

    results = analyzer.generate_report(
        sample_features, sample_target, output_dir, corr_threshold=0.9
    )

    # Check output directory was created
    assert output_dir.exists()

    # Check all expected files were created
    assert (output_dir / "correlation_matrix.csv").exists()
    assert (output_dir / "high_correlations.csv").exists()
    assert (output_dir / "feature_importance.csv").exists()
    assert (output_dir / "correlation_heatmap.png").exists()
    assert (output_dir / "feature_importance.png").exists()

    # Check results dictionary
    assert isinstance(results, dict)
    assert "n_features" in results
    assert "n_samples" in results
    assert "n_high_correlations" in results
    assert "mean_abs_correlation" in results
    assert "top_5_features" in results

    # Check values
    assert results["n_features"] == 3
    assert results["n_samples"] == 100
    assert results["n_high_correlations"] >= 1  # feature_a <-> feature_b


def test_generate_report_creates_directory(
    analyzer, sample_features, sample_target, tmp_path
):
    """Test that generate_report creates output directory if it doesn't exist."""
    output_dir = tmp_path / "nested" / "report"

    # Directory doesn't exist yet
    assert not output_dir.exists()

    analyzer.generate_report(sample_features, sample_target, output_dir)

    # Directory should be created
    assert output_dir.exists()


def test_correlation_matrix_values(analyzer):
    """Test correlation matrix has expected values."""
    # Create perfectly correlated features
    features = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [2, 4, 6, 8, 10]})

    corr_matrix = analyzer.compute_correlations(features)

    # f1 and f2 should be perfectly correlated (f2 = 2*f1)
    assert corr_matrix.loc["f1", "f2"] == pytest.approx(1.0, abs=1e-10)
    assert corr_matrix.loc["f2", "f1"] == pytest.approx(1.0, abs=1e-10)


def test_high_correlations_no_duplicates(analyzer):
    """Test that find_high_correlations doesn't return duplicate pairs."""
    # Create features where all are highly correlated
    np.random.seed(42)
    base = np.random.randn(100)
    features = pd.DataFrame(
        {
            "f1": base,
            "f2": base + np.random.randn(100) * 0.01,
            "f3": base + np.random.randn(100) * 0.01,
        }
    )

    corr_matrix = analyzer.compute_correlations(features)
    high_corr = analyzer.find_high_correlations(corr_matrix, threshold=0.9)

    # Check no duplicate pairs (e.g., both (f1, f2) and (f2, f1))
    pairs_set = set()
    for f1, f2, _ in high_corr:
        pair = tuple(sorted([f1, f2]))
        assert pair not in pairs_set, f"Duplicate pair: {pair}"
        pairs_set.add(pair)


def test_feature_importance_custom_parameters(analyzer, sample_features, sample_target):
    """Test feature importance with custom n_estimators and random_state."""
    importance1 = analyzer.compute_feature_importance(
        sample_features, sample_target, n_estimators=10, random_state=42
    )
    importance2 = analyzer.compute_feature_importance(
        sample_features, sample_target, n_estimators=10, random_state=42
    )

    # Same random state should give same results
    pd.testing.assert_series_equal(importance1, importance2)


def test_high_correlations_threshold_edge_case(analyzer):
    """Test high correlations with threshold exactly at correlation value."""
    features = pd.DataFrame({"f1": [1, 2, 3], "f2": [1, 2, 3]})  # Perfect correlation

    corr_matrix = analyzer.compute_correlations(features)

    # With threshold=1.0, should not find the perfect correlation (need >1.0)
    high_corr = analyzer.find_high_correlations(corr_matrix, threshold=1.0)
    assert len(high_corr) == 0

    # With threshold=0.99, should find it
    high_corr = analyzer.find_high_correlations(corr_matrix, threshold=0.99)
    assert len(high_corr) == 1
