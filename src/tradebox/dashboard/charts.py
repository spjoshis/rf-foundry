"""Chart builder for creating consistent Plotly charts."""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


class ChartBuilder:
    """
    Helper class for creating consistent Plotly charts.

    Provides reusable chart templates with consistent styling
    for the dashboard.

    Example:
        >>> df = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 15]})
        >>> fig = ChartBuilder.line_chart(df, x='x', y='y', title='Test')
    """

    # Color scheme
    COLORS = {
        "primary": "#1f77b4",
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ff7f0e",
        "info": "#17becf",
    }

    @staticmethod
    def portfolio_value_chart(df: pd.DataFrame, title: str = "Portfolio Value") -> go.Figure:
        """
        Create portfolio value line chart.

        Args:
            df: DataFrame with 'timestamp' and 'total_value' columns
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> fig = ChartBuilder.portfolio_value_chart(portfolio_df)
        """
        if df.empty:
            return ChartBuilder._empty_chart("No portfolio data")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["total_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color=ChartBuilder.COLORS["primary"], width=2),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.1)",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            hovermode="x unified",
            template="plotly_white",
            height=400,
        )

        return fig

    @staticmethod
    def pnl_distribution_chart(df: pd.DataFrame, title: str = "P&L Distribution") -> go.Figure:
        """
        Create P&L distribution histogram.

        Args:
            df: DataFrame with 'pnl' or returns column
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> fig = ChartBuilder.pnl_distribution_chart(trades_df)
        """
        if df.empty or "pnl" not in df.columns:
            return ChartBuilder._empty_chart("No P&L data")

        # Remove outliers for better visualization
        pnl = df["pnl"]
        q1, q3 = pnl.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        pnl_filtered = pnl[(pnl >= lower) & (pnl <= upper)]

        fig = px.histogram(
            pnl_filtered,
            nbins=30,
            title=title,
            labels={"value": "P&L (₹)", "count": "Frequency"},
        )

        fig.update_traces(marker_color=ChartBuilder.COLORS["info"])

        fig.update_layout(
            xaxis_title="P&L (₹)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=350,
        )

        return fig

    @staticmethod
    def slippage_timeline_chart(df: pd.DataFrame, title: str = "Slippage Over Time") -> go.Figure:
        """
        Create slippage timeline scatter plot.

        Args:
            df: DataFrame with 'timestamp' and 'slippage_pct' columns
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> fig = ChartBuilder.slippage_timeline_chart(trades_df)
        """
        if df.empty:
            return ChartBuilder._empty_chart("No trade data")

        fig = px.scatter(
            df,
            x="timestamp",
            y="slippage_pct",
            color="symbol" if "symbol" in df.columns else None,
            title=title,
            labels={"slippage_pct": "Slippage (%)", "timestamp": "Date"},
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Slippage (%)",
            hovermode="x unified",
            template="plotly_white",
            height=350,
        )

        return fig

    @staticmethod
    def action_distribution_chart(action_counts: dict, title: str = "Action Distribution") -> go.Figure:
        """
        Create action distribution pie chart.

        Args:
            action_counts: Dict mapping action names to counts
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> counts = {0: 50, 1: 30, 2: 20}  # hold, buy, sell
            >>> fig = ChartBuilder.action_distribution_chart(counts)
        """
        if not action_counts:
            return ChartBuilder._empty_chart("No action data")

        # Map action codes to names
        action_names = {0: "Hold", 1: "Buy", 2: "Sell"}
        labels = [action_names.get(k, f"Action {k}") for k in action_counts.keys()]
        values = list(action_counts.values())

        fig = px.pie(
            names=labels,
            values=values,
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )

        fig.update_layout(
            template="plotly_white",
            height=350,
        )

        return fig

    @staticmethod
    def confidence_distribution_chart(
        df: pd.DataFrame, title: str = "Confidence Distribution"
    ) -> go.Figure:
        """
        Create confidence distribution histogram.

        Args:
            df: DataFrame with 'confidence' column
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> fig = ChartBuilder.confidence_distribution_chart(model_df)
        """
        if df.empty or "confidence" not in df.columns:
            return ChartBuilder._empty_chart("No confidence data")

        # Filter out None values
        df_filtered = df[df["confidence"].notna()]

        if df_filtered.empty:
            return ChartBuilder._empty_chart("No confidence data available")

        fig = px.histogram(
            df_filtered,
            x="confidence",
            nbins=20,
            title=title,
            labels={"confidence": "Confidence Score"},
        )

        fig.update_traces(marker_color=ChartBuilder.COLORS["success"])

        fig.update_layout(
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            template="plotly_white",
            height=350,
        )

        return fig

    @staticmethod
    def error_timeline_chart(df: pd.DataFrame, title: str = "Error Timeline") -> go.Figure:
        """
        Create error timeline bar chart.

        Args:
            df: DataFrame with 'date' and 'count' columns
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> fig = ChartBuilder.error_timeline_chart(errors_df)
        """
        if df.empty:
            return ChartBuilder._empty_chart("No error data")

        fig = px.bar(
            df,
            x="date",
            y="count",
            title=title,
            labels={"count": "Error Count", "date": "Date"},
        )

        fig.update_traces(marker_color=ChartBuilder.COLORS["danger"])

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Error Count",
            template="plotly_white",
            height=300,
        )

        return fig

    @staticmethod
    def trade_distribution_by_symbol_chart(
        df: pd.DataFrame, title: str = "Trade Distribution by Symbol"
    ) -> go.Figure:
        """
        Create trade distribution pie chart by symbol.

        Args:
            df: DataFrame with 'symbol' column
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> fig = ChartBuilder.trade_distribution_by_symbol_chart(trades_df)
        """
        if df.empty or "symbol" not in df.columns:
            return ChartBuilder._empty_chart("No trade data")

        symbol_counts = df["symbol"].value_counts()

        fig = px.pie(
            names=symbol_counts.index,
            values=symbol_counts.values,
            title=title,
        )

        fig.update_layout(
            template="plotly_white",
            height=350,
        )

        return fig

    @staticmethod
    def returns_over_time_chart(df: pd.DataFrame, title: str = "Returns Over Time") -> go.Figure:
        """
        Create returns line chart.

        Args:
            df: DataFrame with 'timestamp' and 'daily_return_pct' columns
            title: Chart title

        Returns:
            Plotly Figure

        Example:
            >>> fig = ChartBuilder.returns_over_time_chart(portfolio_df)
        """
        if df.empty or "daily_return_pct" not in df.columns:
            return ChartBuilder._empty_chart("No returns data")

        fig = go.Figure()

        # Color returns based on positive/negative
        colors = ["green" if x > 0 else "red" for x in df["daily_return_pct"]]

        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["daily_return_pct"],
                name="Daily Returns",
                marker_color=colors,
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode="x unified",
            template="plotly_white",
            height=350,
        )

        return fig

    @staticmethod
    def _empty_chart(message: str = "No data available") -> go.Figure:
        """
        Create empty placeholder chart.

        Args:
            message: Message to display

        Returns:
            Plotly Figure with message
        """
        fig = go.Figure()

        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )

        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            template="plotly_white",
            height=300,
        )

        return fig
