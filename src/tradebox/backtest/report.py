"""Report generation for backtest results."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger

from tradebox.backtest.engine import BacktestResult


class BacktestReport:
    """
    Generate reports from backtest results.

    Creates JSON and text summary reports.

    Example:
        >>> report = BacktestReport()
        >>> report.save_json(result, metrics, "reports/backtest.json")
        >>> report.save_text_summary(result, metrics, "reports/summary.txt")
    """

    def save_json(
        self,
        result: BacktestResult,
        metrics: Dict,
        output_path: Union[str, Path],
    ) -> None:
        """
        Save backtest results as JSON.

        Args:
            result: BacktestResult to save
            metrics: Metrics dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = result.to_dict()
        data["metrics"] = metrics

        # Convert equity curve to list
        data["equity_curve"] = {
            "dates": result.equity_curve.index.strftime("%Y-%m-%d").tolist(),
            "values": result.equity_curve.values.tolist(),
        }

        # Save
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved backtest JSON to {output_path}")

    def save_text_summary(
        self,
        result: BacktestResult,
        metrics: Dict,
        output_path: Union[str, Path],
    ) -> None:
        """
        Save text summary of backtest results.

        Args:
            result: BacktestResult to summarize
            metrics: Metrics dictionary
            output_path: Path to save text file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build summary
        summary = []
        summary.append("=" * 60)
        summary.append("BACKTEST RESULTS SUMMARY")
        summary.append("=" * 60)
        summary.append("")

        # Configuration
        summary.append("Configuration:")
        summary.append(f"  Initial Capital: ₹{result.config.initial_capital:,.0f}")
        summary.append(f"  Commission: {result.config.commission_pct:.4f}")
        summary.append(f"  Slippage: {result.config.slippage_pct:.4f}")
        summary.append("")

        # Period
        if not result.equity_curve.empty:
            start_date = result.equity_curve.index[0].strftime("%Y-%m-%d")
            end_date = result.equity_curve.index[-1].strftime("%Y-%m-%d")
            days = len(result.equity_curve)
            summary.append(f"Period: {start_date} to {end_date} ({days} days)")
            summary.append("")

        # Returns
        summary.append("Returns:")
        summary.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        summary.append(f"  CAGR: {metrics.get('cagr', 0):.2%}")
        summary.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        summary.append("")

        # Risk-Adjusted Metrics
        summary.append("Risk-Adjusted Metrics:")
        summary.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        summary.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        summary.append(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        summary.append("")

        # Risk Metrics
        summary.append("Risk Metrics:")
        summary.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        summary.append(f"  Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}")
        summary.append(f"  VaR 95%: {metrics.get('var_95', 0):.4f}")
        summary.append(f"  CVaR 95%: {metrics.get('cvar_95', 0):.4f}")
        summary.append("")

        # Trading Metrics
        summary.append("Trading Metrics:")
        summary.append(f"  Total Trades: {metrics.get('total_trades', 0)}")
        summary.append(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        summary.append(f"  Avg Win: ₹{metrics.get('avg_win', 0):,.2f}")
        summary.append(f"  Avg Loss: ₹{metrics.get('avg_loss', 0):,.2f}")
        summary.append(f"  Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}")
        summary.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        summary.append(f"  Avg Trade Duration: {metrics.get('avg_trade_duration_days', 0):.1f} days")
        summary.append("")

        # Portfolio
        if not result.equity_curve.empty:
            final_value = result.equity_curve.iloc[-1]
            pnl = final_value - result.config.initial_capital
            summary.append("Final Portfolio:")
            summary.append(f"  Value: ₹{final_value:,.0f}")
            summary.append(f"  P&L: ₹{pnl:,.0f}")
            summary.append("")

        summary.append("=" * 60)

        # Write file
        with open(output_path, "w") as f:
            f.write("\n".join(summary))

        logger.info(f"Saved text summary to {output_path}")

    def print_summary(self, result: BacktestResult, metrics: Dict) -> None:
        """
        Print summary to console.

        Args:
            result: BacktestResult to summarize
            metrics: Metrics dictionary
        """
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        if not result.equity_curve.empty:
            final_value = result.equity_curve.iloc[-1]
            total_return = metrics.get("total_return", 0)
            print(f"\nFinal Portfolio Value: ₹{final_value:,.0f}")
            print(f"Total Return: {total_return:.2%}")

        print(f"\nSharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")

        print("=" * 60 + "\n")
