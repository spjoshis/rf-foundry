"""Feature correlation and importance analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestRegressor


class FeatureAnalyzer:
    """
    Analyze feature correlations and importance for trading indicators.

    Provides tools to:
    - Compute correlation matrices
    - Identify highly correlated features (multicollinearity)
    - Calculate feature importance using Random Forest
    - Generate visualizations and reports

    Example:
        >>> analyzer = FeatureAnalyzer()
        >>> corr_matrix = analyzer.compute_correlations(features_df)
        >>> high_corr = analyzer.find_high_correlations(corr_matrix, threshold=0.95)
        >>> importance = analyzer.compute_feature_importance(features_df, target_series)
    """

    def __init__(self) -> None:
        """Initialize feature analyzer."""
        logger.info("FeatureAnalyzer initialized")

    def compute_correlations(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix for all features.

        Args:
            features: DataFrame with features as columns

        Returns:
            Correlation matrix (DataFrame) with Pearson correlation coefficients

        Example:
            >>> corr = analyzer.compute_correlations(features)
            >>> corr.loc['RSI', 'Stochastic']  # Correlation between RSI and Stochastic
        """
        logger.info(f"Computing correlations for {len(features.columns)} features")
        corr_matrix = features.corr()
        logger.debug(
            f"Correlation matrix shape: {corr_matrix.shape}, "
            f"mean abs correlation: {corr_matrix.abs().mean().mean():.3f}"
        )
        return corr_matrix

    def find_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.95
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of features with correlation above threshold.

        Useful for detecting multicollinearity (features that are redundant).

        Args:
            corr_matrix: Correlation matrix from compute_correlations()
            threshold: Minimum absolute correlation to report (default: 0.95)

        Returns:
            List of tuples: (feature1, feature2, correlation_value)
            Sorted by absolute correlation (highest first)

        Example:
            >>> high_corr = analyzer.find_high_correlations(corr, threshold=0.9)
            >>> for f1, f2, corr_val in high_corr:
            ...     print(f"{f1} <-> {f2}: {corr_val:.3f}")
        """
        logger.info(f"Finding correlations above {threshold}")
        high_corr = []

        # Iterate upper triangle only (avoid duplicates and self-correlation)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                    )

        # Sort by absolute correlation (highest first)
        high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

        logger.info(f"Found {len(high_corr)} feature pairs with |corr| > {threshold}")
        return high_corr

    def compute_feature_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> pd.Series:
        """
        Compute feature importance using Random Forest.

        Uses RandomForestRegressor to measure which features are most predictive
        of the target variable (e.g., next-day returns).

        Args:
            features: DataFrame with features as columns
            target: Series with target variable (same length as features)
            n_estimators: Number of trees in random forest (default: 100)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            Series with feature importance scores (sorted descending)
            Higher scores = more important features

        Raises:
            ValueError: If features and target have different lengths or contain NaNs

        Example:
            >>> target = df['Close'].pct_change().shift(-1)  # Next-day return
            >>> importance = analyzer.compute_feature_importance(features, target)
            >>> print(importance.head(10))  # Top 10 features
        """
        logger.info(
            f"Computing feature importance for {len(features.columns)} features "
            f"using {n_estimators} trees"
        )

        # Validation
        if len(features) != len(target):
            raise ValueError(
                f"Features ({len(features)}) and target ({len(target)}) "
                "have different lengths"
            )

        # Drop rows with NaN in either features or target
        mask = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features[mask]
        target_clean = target[mask]

        if len(features_clean) == 0:
            raise ValueError("No valid samples after dropping NaN values")

        logger.debug(
            f"Training on {len(features_clean)} samples "
            f"(dropped {len(features) - len(features_clean)} NaN rows)"
        )

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        rf.fit(features_clean, target_clean)

        # Extract and sort importances
        importance = pd.Series(rf.feature_importances_, index=features.columns)
        importance = importance.sort_values(ascending=False)

        logger.info(
            f"Top 3 features: "
            f"{importance.head(3).to_dict()}"
        )

        return importance

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """
        Generate heatmap visualization of correlation matrix.

        Args:
            corr_matrix: Correlation matrix from compute_correlations()
            output_path: If provided, save figure to this path
            figsize: Figure size (width, height) in inches

        Example:
            >>> analyzer.plot_correlation_matrix(corr, output_path=Path("corr.png"))
        """
        logger.info("Plotting correlation matrix heatmap")

        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=False,  # Don't annotate cells (too many features)
            cmap="RdBu_r",  # Red-Blue diverging colormap
            center=0,  # Center colormap at 0
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
        )

        plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved correlation heatmap to {output_path}")

        plt.close()

    def plot_feature_importance(
        self,
        importance: pd.Series,
        top_n: int = 20,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        Generate bar plot of top N most important features.

        Args:
            importance: Feature importance from compute_feature_importance()
            top_n: Number of top features to plot (default: 20)
            output_path: If provided, save figure to this path
            figsize: Figure size (width, height) in inches

        Example:
            >>> analyzer.plot_feature_importance(importance, top_n=15,
            ...                                   output_path=Path("importance.png"))
        """
        logger.info(f"Plotting top {top_n} feature importances")

        top_features = importance.head(top_n)

        plt.figure(figsize=figsize)
        top_features.sort_values().plot(kind="barh", color="steelblue")

        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title(
            f"Top {top_n} Feature Importances", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {output_path}")

        plt.close()

    def generate_report(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        output_dir: Path,
        corr_threshold: float = 0.95,
    ) -> Dict[str, any]:
        """
        Generate comprehensive feature analysis report.

        Creates:
        - Correlation matrix CSV
        - High correlation pairs CSV
        - Feature importance CSV
        - Correlation heatmap PNG
        - Feature importance plot PNG
        - Summary statistics JSON

        Args:
            features: DataFrame with features
            target: Target variable (e.g., next-day returns)
            output_dir: Directory to save all outputs
            corr_threshold: Threshold for high correlation detection

        Returns:
            Dictionary with analysis results

        Example:
            >>> results = analyzer.generate_report(
            ...     features, target, Path("analysis_output"), corr_threshold=0.9
            ... )
            >>> print(f"Found {results['n_high_correlations']} redundant pairs")
        """
        logger.info(f"Generating feature analysis report in {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute correlation
        corr_matrix = self.compute_correlations(features)
        high_corr = self.find_high_correlations(corr_matrix, threshold=corr_threshold)

        # Compute importance
        importance = self.compute_feature_importance(features, target)

        # Save data
        corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
        pd.DataFrame(
            high_corr, columns=["Feature1", "Feature2", "Correlation"]
        ).to_csv(output_dir / "high_correlations.csv", index=False)
        importance.to_csv(output_dir / "feature_importance.csv", header=True)

        # Save plots
        self.plot_correlation_matrix(corr_matrix, output_dir / "correlation_heatmap.png")
        self.plot_feature_importance(
            importance, top_n=20, output_path=output_dir / "feature_importance.png"
        )

        # Summary statistics
        results = {
            "n_features": len(features.columns),
            "n_samples": len(features),
            "n_high_correlations": len(high_corr),
            "mean_abs_correlation": float(corr_matrix.abs().mean().mean()),
            "top_5_features": importance.head(5).to_dict(),
        }

        logger.info(f"Report generated: {results['n_features']} features analyzed")
        return results
