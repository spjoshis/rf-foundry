"""Overview dashboard page."""

import streamlit as st
from tradebox.monitoring import MetricsQuery
from tradebox.dashboard.charts import ChartBuilder
from tradebox.dashboard.utils import format_currency, format_percentage, calculate_color


def render(metrics: MetricsQuery) -> None:
    """
    Render overview dashboard page.

    Args:
        metrics: MetricsQuery instance

    Displays:
        - Key portfolio metrics
        - Portfolio value chart
        - Active positions table
        - Recent trades
    """
    st.title("📈 Portfolio Overview")

    # Fetch latest portfolio metrics
    portfolio = metrics.get_latest_portfolio()

    if portfolio is None:
        st.warning("⚠️ No portfolio data available. Run the orchestrator to collect metrics.")
        st.info(
            "Run: `poetry run python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --once`"
        )
        return

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            format_currency(portfolio["total_value"]),
            delta=format_percentage(portfolio.get("daily_return_pct", 0.0))
            if portfolio.get("daily_return_pct")
            else None,
        )

    with col2:
        st.metric("Cash", format_currency(portfolio["cash"]))

    with col3:
        pnl = portfolio["unrealized_pnl"]
        st.metric(
            "Unrealized P&L",
            format_currency(pnl),
            delta=format_currency(pnl) if pnl != 0 else None,
            delta_color="normal" if pnl >= 0 else "inverse",
        )

    with col4:
        sharpe = portfolio.get("sharpe_ratio")
        if sharpe is not None:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        else:
            st.metric("Sharpe Ratio", "N/A")

    st.markdown("---")

    # Portfolio value chart
    st.subheader("Portfolio Value Over Time")

    history_df = metrics.get_portfolio_history(days=30)

    if not history_df.empty:
        fig = ChartBuilder.portfolio_value_chart(history_df)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)

        with col1:
            start_value = history_df["total_value"].iloc[0]
            end_value = history_df["total_value"].iloc[-1]
            total_return = ((end_value - start_value) / start_value) * 100
            st.metric("30-Day Return", format_percentage(total_return))

        with col2:
            max_value = history_df["total_value"].max()
            st.metric("Peak Value", format_currency(max_value))

        with col3:
            min_value = history_df["total_value"].min()
            st.metric("Trough Value", format_currency(min_value))
    else:
        st.info("No portfolio history available yet.")

    st.markdown("---")

    # Active positions
    st.subheader("Active Positions")

    positions_df = metrics.get_active_positions()

    if not positions_df.empty:
        # Format DataFrame
        positions_df["unrealized_pnl_formatted"] = positions_df["unrealized_pnl"].apply(
            lambda x: format_currency(x)
        )

        st.dataframe(
            positions_df[
                ["symbol", "quantity", "avg_price", "current_price", "unrealized_pnl_formatted"]
            ].rename(
                columns={
                    "symbol": "Symbol",
                    "quantity": "Quantity",
                    "avg_price": "Avg Price (₹)",
                    "current_price": "Current Price (₹)",
                    "unrealized_pnl_formatted": "Unrealized P&L",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No active positions.")

    st.markdown("---")

    # Recent trades
    st.subheader("Recent Trades")

    trades_df = metrics.get_recent_trades(n=10)

    if not trades_df.empty:
        # Format columns
        trades_df["filled_price_formatted"] = trades_df["filled_price"].apply(
            lambda x: f"₹{x:.2f}"
        )
        trades_df["slippage_formatted"] = trades_df["slippage_pct"].apply(
            lambda x: f"{x:+.2f}%"
        )
        trades_df["commission_formatted"] = trades_df["commission"].apply(
            lambda x: f"₹{x:.2f}"
        )

        st.dataframe(
            trades_df[
                [
                    "timestamp",
                    "symbol",
                    "side",
                    "quantity",
                    "filled_price_formatted",
                    "slippage_formatted",
                    "commission_formatted",
                    "order_status",
                ]
            ].rename(
                columns={
                    "timestamp": "Time",
                    "symbol": "Symbol",
                    "side": "Side",
                    "quantity": "Qty",
                    "filled_price_formatted": "Price",
                    "slippage_formatted": "Slippage",
                    "commission_formatted": "Commission",
                    "order_status": "Status",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No trades executed yet.")

    # Performance summary
    st.markdown("---")
    st.subheader("30-Day Performance Summary")

    perf = metrics.get_performance_summary(days=30)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")

    with col2:
        st.metric("Sortino Ratio", f"{perf['sortino_ratio']:.2f}")

    with col3:
        st.metric("Max Drawdown", format_percentage(perf["max_drawdown_pct"]))

    with col4:
        st.metric("Win Rate", format_percentage(perf["win_rate"] * 100))
