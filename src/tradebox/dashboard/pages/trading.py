"""Trading page for trade execution analytics."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from tradebox.monitoring import MetricsQuery
from tradebox.dashboard.charts import ChartBuilder
from tradebox.dashboard.utils import format_currency, format_percentage


def render(metrics: MetricsQuery) -> None:
    """
    Render trading analytics page.

    Args:
        metrics: MetricsQuery instance

    Displays:
        - Fill quality metrics (slippage, latency, success rate)
        - Trade history with filtering
        - Trade distribution by symbol
        - Slippage timeline
        - Commission analysis
    """
    st.title("💰 Trading Analytics")

    # Date range filter
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            max_value=datetime.now().date(),
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
        )

    # Fetch trades for date range
    trades_df = metrics.store.query_trades_by_date(
        start_date=start_date, end_date=end_date
    )

    if trades_df.empty:
        st.warning("⚠️ No trades found for selected date range.")
        st.info(
            "Execute trades using the orchestrator to see analytics: "
            "`poetry run python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --once`"
        )
        return

    # Fill Quality Metrics
    st.subheader("📊 Fill Quality Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_slippage = trades_df["slippage_pct"].mean()
        st.metric(
            "Avg Slippage",
            format_percentage(avg_slippage),
            delta=format_percentage(avg_slippage) if avg_slippage != 0 else None,
            delta_color="inverse",  # Lower slippage is better
        )

    with col2:
        avg_latency = trades_df["latency_ms"].mean()
        st.metric("Avg Latency", f"{avg_latency:.0f} ms")

    with col3:
        success_rate = (
            trades_df["order_status"] == "COMPLETE"
        ).sum() / len(trades_df) * 100
        st.metric("Fill Success Rate", format_percentage(success_rate))

    with col4:
        total_commission = trades_df["commission"].sum()
        st.metric("Total Commission", format_currency(total_commission))

    st.markdown("---")

    # Trade Distribution by Symbol
    st.subheader("Trade Distribution by Symbol")

    col1, col2 = st.columns(2)

    with col1:
        fig_symbol = ChartBuilder.trade_distribution_by_symbol_chart(trades_df)
        st.plotly_chart(fig_symbol, use_container_width=True)

    with col2:
        # Trade side distribution
        side_counts = trades_df["side"].value_counts()
        st.markdown("**Trade Side Distribution**")

        for side, count in side_counts.items():
            pct = (count / len(trades_df)) * 100
            st.metric(f"{side} Trades", f"{count} ({pct:.1f}%)")

    st.markdown("---")

    # Slippage Analysis
    st.subheader("Slippage Over Time")

    fig_slippage = ChartBuilder.slippage_timeline_chart(trades_df)
    st.plotly_chart(fig_slippage, use_container_width=True)

    # Slippage statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        min_slippage = trades_df["slippage_pct"].min()
        st.metric("Best Fill", format_percentage(min_slippage))

    with col2:
        median_slippage = trades_df["slippage_pct"].median()
        st.metric("Median Slippage", format_percentage(median_slippage))

    with col3:
        max_slippage = trades_df["slippage_pct"].max()
        st.metric(
            "Worst Fill",
            format_percentage(max_slippage),
            delta_color="inverse",
        )

    st.markdown("---")

    # Commission Analysis
    st.subheader("Commission Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Commission by symbol
        commission_by_symbol = (
            trades_df.groupby("symbol")["commission"].sum().sort_values(ascending=False)
        )

        st.markdown("**Commission by Symbol**")
        for symbol, commission in commission_by_symbol.items():
            st.write(f"- {symbol}: {format_currency(commission)}")

    with col2:
        # Commission metrics
        avg_commission = trades_df["commission"].sum() / len(trades_df)
        total_trade_value = (trades_df["filled_price"] * trades_df["quantity"]).sum()
        commission_pct = (
            (trades_df["commission"].sum() / total_trade_value) * 100
            if total_trade_value > 0
            else 0
        )

        st.metric("Avg Commission/Trade", format_currency(avg_commission))
        st.metric("Total Trade Value", format_currency(total_trade_value))
        st.metric("Commission %", format_percentage(commission_pct))

    st.markdown("---")

    # Trade History Table
    st.subheader("Trade History")

    # Symbol filter
    symbols = trades_df["symbol"].unique().tolist()
    selected_symbols = st.multiselect(
        "Filter by Symbol",
        options=symbols,
        default=symbols,
    )

    # Side filter
    sides = trades_df["side"].unique().tolist()
    selected_sides = st.multiselect(
        "Filter by Side",
        options=sides,
        default=sides,
    )

    # Apply filters
    filtered_df = trades_df[
        (trades_df["symbol"].isin(selected_symbols))
        & (trades_df["side"].isin(selected_sides))
    ]

    # Format for display
    display_df = filtered_df.copy()
    display_df["filled_price_formatted"] = display_df["filled_price"].apply(
        lambda x: f"₹{x:.2f}"
    )
    display_df["slippage_formatted"] = display_df["slippage_pct"].apply(
        lambda x: f"{x:+.2f}%"
    )
    display_df["commission_formatted"] = display_df["commission"].apply(
        lambda x: f"₹{x:.2f}"
    )
    display_df["latency_formatted"] = display_df["latency_ms"].apply(
        lambda x: f"{x:.0f} ms"
    )

    # Calculate P&L for each trade (approximate)
    # For buy: negative cost; for sell: positive revenue
    display_df["trade_value"] = (
        display_df["filled_price"] * display_df["quantity"]
    )
    display_df["trade_value_formatted"] = display_df["trade_value"].apply(
        format_currency
    )

    st.dataframe(
        display_df[
            [
                "timestamp",
                "symbol",
                "side",
                "quantity",
                "filled_price_formatted",
                "trade_value_formatted",
                "slippage_formatted",
                "commission_formatted",
                "latency_formatted",
                "order_status",
            ]
        ].rename(
            columns={
                "timestamp": "Time",
                "symbol": "Symbol",
                "side": "Side",
                "quantity": "Qty",
                "filled_price_formatted": "Price",
                "trade_value_formatted": "Value",
                "slippage_formatted": "Slippage",
                "commission_formatted": "Commission",
                "latency_formatted": "Latency",
                "order_status": "Status",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Export functionality
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"trades_{start_date}_{end_date}.csv",
            mime="text/csv",
        )

    with col2:
        st.caption(f"Showing {len(filtered_df)} of {len(trades_df)} trades")
