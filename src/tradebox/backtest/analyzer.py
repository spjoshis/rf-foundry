"""Analysis and visualization for backtest results."""

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from tradebox.backtest.engine import BacktestResult


class BacktestAnalyzer:
    """
    Analyze and visualize backtest results.

    Creates various plots for performance analysis including equity curve,
    drawdown chart, returns distribution, and trade analysis.

    Example:
        >>> analyzer = BacktestAnalyzer()
        >>> analyzer.plot_equity_curve(result)
        >>> analyzer.plot_drawdown(result)
        >>> analyzer.create_dashboard(result, output_path="reports/backtest.png")
    """

    def __init__(self, figsize: tuple = (12, 8)) -> None:
        """
        Initialize analyzer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-darkgrid")

    def plot_equity_curve(
        self,
        result: BacktestResult,
        title: str = "Portfolio Equity Curve",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot portfolio equity curve over time.

        Args:
            result: BacktestResult to visualize
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        equity = result.equity_curve
        ax.plot(equity.index, equity.values, linewidth=2, label="Portfolio Value")
        ax.axhline(
            y=result.config.initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value (₹)", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Add stats text
        initial = result.config.initial_capital
        final = equity.iloc[-1]
        total_return = (final - initial) / initial * 100

        stats_text = f"Initial: ₹{initial:,.0f}\nFinal: ₹{final:,.0f}\nReturn: {total_return:.2f}%"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved equity curve to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_drawdown(
        self,
        result: BacktestResult,
        title: str = "Drawdown Chart",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot drawdown chart over time.

        Args:
            result: BacktestResult to visualize
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        equity = result.equity_curve
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100  # Convert to percentage

        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        ax.plot(drawdown.index, drawdown.values, linewidth=1, color="darkred")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add max drawdown line
        max_dd = drawdown.min()
        ax.axhline(
            y=max_dd,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Max Drawdown: {max_dd:.2f}%",
        )
        ax.legend(loc="lower left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved drawdown chart to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_returns_distribution(
        self,
        result: BacktestResult,
        title: str = "Daily Returns Distribution",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot histogram of daily returns.

        Args:
            result: BacktestResult to visualize
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        returns = result.daily_returns * 100  # Convert to percentage

        ax.hist(returns, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax.axvline(x=returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.3f}%")
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.5)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Daily Return (%)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Add stats
        stats_text = (
            f"Mean: {returns.mean():.3f}%\n"
            f"Std: {returns.std():.3f}%\n"
            f"Skew: {returns.skew():.3f}\n"
            f"Kurt: {returns.kurtosis():.3f}"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved returns distribution to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_monthly_returns_heatmap(
        self,
        result: BacktestResult,
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot heatmap of monthly returns.

        Args:
            result: BacktestResult to visualize
            title: Plot title
            save_path: Optional path to save figure
        """
        # Calculate monthly returns
        equity = result.equity_curve
        monthly_returns = equity.resample("M").last().pct_change().dropna() * 100

        # Create year-month pivot table
        monthly_returns_df = pd.DataFrame({
            "Year": monthly_returns.index.year,
            "Month": monthly_returns.index.month,
            "Return": monthly_returns.values,
        })

        pivot = monthly_returns_df.pivot(index="Year", columns="Month", values="Return")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

        # Set ticks
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ])
        ax.set_yticklabels(pivot.index)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Return (%)", rotation=270, labelpad=20)

        # Add values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if not pd.isna(pivot.values[i, j]):
                    text = ax.text(
                        j, i, f"{pivot.values[i, j]:.1f}",
                        ha="center", va="center", color="black", fontsize=8
                    )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Year", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved monthly returns heatmap to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_trade_analysis(
        self,
        result: BacktestResult,
        title: str = "Trade Analysis",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Plot trade P&L distribution.

        Args:
            result: BacktestResult to visualize
            title: Plot title
            save_path: Optional path to save figure
        """
        trades = [t for t in result.trades if t.pnl is not None]

        if not trades:
            logger.warning("No completed trades to analyze")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # P&L histogram
        pnls = [t.pnl for t in trades]
        ax1.hist(pnls, bins=30, alpha=0.7, color="blue", edgecolor="black")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax1.set_title("Trade P&L Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("P&L (₹)", fontsize=10)
        ax1.set_ylabel("Frequency", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Win/Loss pie chart
        winning = len([t for t in trades if t.pnl > 0])
        losing = len([t for t in trades if t.pnl < 0])
        breakeven = len([t for t in trades if t.pnl == 0])

        ax2.pie(
            [winning, losing, breakeven],
            labels=["Wins", "Losses", "Breakeven"],
            colors=["green", "red", "gray"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("Win/Loss Distribution", fontsize=12, fontweight="bold")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved trade analysis to {save_path}")
        else:
            plt.show()

        plt.close()

    def create_dashboard(
        self,
        result: BacktestResult,
        output_path: Union[str, Path],
        metrics: Optional[Dict] = None,
    ) -> None:
        """
        Create comprehensive dashboard with all plots.

        Args:
            result: BacktestResult to visualize
            output_path: Path to save dashboard PNG
            metrics: Optional metrics dictionary to display
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Equity curve (top, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        equity = result.equity_curve
        ax1.plot(equity.index, equity.values, linewidth=2)
        ax1.set_title("Portfolio Equity Curve", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Value (₹)", fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Drawdown (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Returns distribution (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        returns = result.daily_returns * 100
        ax3.hist(returns, bins=30, alpha=0.7, color="blue", edgecolor="black")
        ax3.set_title("Daily Returns", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Return (%)", fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Trade P&L (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        trades = [t for t in result.trades if t.pnl is not None]
        if trades:
            pnls = [t.pnl for t in trades]
            ax4.hist(pnls, bins=20, alpha=0.7, color="green", edgecolor="black")
            ax4.axvline(x=0, color="red", linestyle="--")
        ax4.set_title("Trade P&L", fontsize=12, fontweight="bold")
        ax4.set_xlabel("P&L (₹)", fontsize=10)
        ax4.grid(True, alpha=0.3)

        # Metrics table (top right)
        ax5 = fig.add_subplot(gs[0, 2])
        ax5.axis("off")
        if metrics:
            metrics_text = "Performance Metrics\n" + "=" * 30 + "\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    if "rate" in key or "ratio" in key or "return" in key:
                        metrics_text += f"{key}: {value:.3f}\n"
                    else:
                        metrics_text += f"{key}: {value:.2f}\n"
                else:
                    metrics_text += f"{key}: {value}\n"
            ax5.text(
                0.1, 0.9, metrics_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment="top", family="monospace"
            )

        # Rolling Sharpe (bottom left and center)
        ax6 = fig.add_subplot(gs[2, :2])
        rolling_sharpe = (
            returns.rolling(window=60).mean() / returns.rolling(window=60).std()
        ) * np.sqrt(252)
        ax6.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
        ax6.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax6.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1")
        ax6.set_title("Rolling 60-Day Sharpe Ratio", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Sharpe Ratio", fontsize=10)
        ax6.legend(loc="best")
        ax6.grid(True, alpha=0.3)

        plt.suptitle(
            "Backtest Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved dashboard to {output_path}")
        plt.close()
