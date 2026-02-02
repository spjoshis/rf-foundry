"""Model page for RL agent performance analytics."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from tradebox.monitoring import MetricsQuery
from tradebox.dashboard.charts import ChartBuilder
from tradebox.dashboard.utils import format_percentage


def render(metrics: MetricsQuery) -> None:
    """
    Render model performance page.

    Args:
        metrics: MetricsQuery instance

    Displays:
        - Latest model predictions
        - Action distribution (pie chart)
        - Confidence distribution (histogram)
        - Action timeline
        - Model decision analysis
    """
    st.title("🤖 Model Performance")

    # Date range filter
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            max_value=datetime.now().date(),
            key="model_start",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            key="model_end",
        )

    # Fetch model metrics
    model_df = metrics.store.query_model_metrics_by_date(
        start_date=start_date, end_date=end_date
    )

    if model_df.empty:
        st.warning("⚠️ No model predictions found for selected date range.")
        st.info(
            "Run the orchestrator to generate predictions: "
            "`poetry run python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --once`"
        )
        return

    # Action mapping
    ACTION_NAMES = {0: "Hold", 1: "Buy", 2: "Sell"}

    # Map action codes to names
    if "action" in model_df.columns:
        model_df["action_name"] = model_df["action"].map(ACTION_NAMES)

    # Key Model Metrics
    st.subheader("📊 Model Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_predictions = len(model_df)
        st.metric("Total Predictions", f"{total_predictions:,}")

    with col2:
        unique_symbols = model_df["symbol"].nunique()
        st.metric("Symbols Tracked", unique_symbols)

    with col3:
        avg_confidence = model_df["confidence"].mean() if "confidence" in model_df.columns else None
        if avg_confidence is not None:
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        else:
            st.metric("Avg Confidence", "N/A")

    with col4:
        # Calculate action diversity (entropy)
        action_counts = model_df["action"].value_counts(normalize=True)
        if len(action_counts) > 1:
            import numpy as np
            entropy = -sum(action_counts * np.log(action_counts))
            max_entropy = np.log(3)  # For 3 actions
            diversity = (entropy / max_entropy) * 100
            st.metric("Action Diversity", f"{diversity:.0f}%")
        else:
            st.metric("Action Diversity", "Low")

    st.markdown("---")

    # Action Distribution
    st.subheader("Action Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart
        action_counts = model_df["action"].value_counts().to_dict()
        fig_actions = ChartBuilder.action_distribution_chart(action_counts)
        st.plotly_chart(fig_actions, use_container_width=True)

    with col2:
        # Action breakdown table
        st.markdown("**Action Breakdown**")

        for action_code, count in model_df["action"].value_counts().items():
            action_name = ACTION_NAMES.get(action_code, f"Action {action_code}")
            pct = (count / len(model_df)) * 100

            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(f"**{action_name}**")
            with col_b:
                st.write(f"{count} ({pct:.1f}%)")

        # Expected values if shown
        if "expected_value" in model_df.columns:
            st.markdown("---")
            st.markdown("**Average Expected Value by Action**")
            avg_ev = model_df.groupby("action")["expected_value"].mean()
            for action_code, ev in avg_ev.items():
                action_name = ACTION_NAMES.get(action_code, f"Action {action_code}")
                st.write(f"- {action_name}: {ev:.4f}")

    st.markdown("---")

    # Confidence Distribution
    st.subheader("Confidence Distribution")

    if "confidence" in model_df.columns and model_df["confidence"].notna().any():
        fig_confidence = ChartBuilder.confidence_distribution_chart(model_df)
        st.plotly_chart(fig_confidence, use_container_width=True)

        # Confidence statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_conf = model_df["confidence"].min()
            st.metric("Min Confidence", f"{min_conf:.2%}")

        with col2:
            median_conf = model_df["confidence"].median()
            st.metric("Median Confidence", f"{median_conf:.2%}")

        with col3:
            max_conf = model_df["confidence"].max()
            st.metric("Max Confidence", f"{max_conf:.2%}")

        with col4:
            std_conf = model_df["confidence"].std()
            st.metric("Std Dev", f"{std_conf:.2%}")
    else:
        st.info("Confidence scores not available in model predictions.")

    st.markdown("---")

    # Action Timeline
    st.subheader("Actions Over Time")

    # Resample by date for cleaner visualization
    model_df["date"] = pd.to_datetime(model_df["timestamp"]).dt.date
    daily_actions = model_df.groupby(["date", "action_name"]).size().unstack(fill_value=0)

    if not daily_actions.empty:
        # Stacked bar chart
        import plotly.graph_objects as go

        fig = go.Figure()

        colors = {
            "Hold": "#FFA500",  # Orange
            "Buy": "#2ca02c",   # Green
            "Sell": "#d62728",  # Red
        }

        for action_name in ["Hold", "Buy", "Sell"]:
            if action_name in daily_actions.columns:
                fig.add_trace(
                    go.Bar(
                        x=daily_actions.index,
                        y=daily_actions[action_name],
                        name=action_name,
                        marker_color=colors.get(action_name, "#1f77b4"),
                    )
                )

        fig.update_layout(
            barmode="stack",
            title="Daily Action Distribution",
            xaxis_title="Date",
            yaxis_title="Number of Actions",
            template="plotly_white",
            height=350,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily action data available.")

    st.markdown("---")

    # Symbol-wise Action Distribution
    st.subheader("Action Distribution by Symbol")

    # Symbol filter
    symbols = model_df["symbol"].unique().tolist()
    selected_symbol = st.selectbox(
        "Select Symbol",
        options=["All"] + symbols,
    )

    if selected_symbol != "All":
        symbol_df = model_df[model_df["symbol"] == selected_symbol]
    else:
        symbol_df = model_df

    # Action counts for selected symbol
    symbol_action_counts = symbol_df["action_name"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        # Table
        st.markdown(f"**Actions for {selected_symbol}**")
        for action_name, count in symbol_action_counts.items():
            pct = (count / len(symbol_df)) * 100
            st.write(f"- {action_name}: {count} ({pct:.1f}%)")

    with col2:
        # Metrics
        st.metric("Total Predictions", len(symbol_df))

        if "confidence" in symbol_df.columns:
            avg_conf = symbol_df["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")

    st.markdown("---")

    # Recent Predictions Table
    st.subheader("Recent Predictions")

    # Show last 50 predictions
    recent_df = model_df.sort_values("timestamp", ascending=False).head(50)

    # Format display
    display_df = recent_df.copy()

    if "confidence" in display_df.columns:
        display_df["confidence_formatted"] = display_df["confidence"].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
        )
    else:
        display_df["confidence_formatted"] = "N/A"

    if "expected_value" in display_df.columns:
        display_df["expected_value_formatted"] = display_df["expected_value"].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
    else:
        display_df["expected_value_formatted"] = "N/A"

    # Select columns for display
    display_columns = ["timestamp", "symbol", "action_name"]

    if "confidence" in display_df.columns:
        display_columns.append("confidence_formatted")

    if "expected_value" in display_df.columns:
        display_columns.append("expected_value_formatted")

    column_renames = {
        "timestamp": "Time",
        "symbol": "Symbol",
        "action_name": "Action",
        "confidence_formatted": "Confidence",
        "expected_value_formatted": "Expected Value",
    }

    st.dataframe(
        display_df[display_columns].rename(columns=column_renames),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    # Export functionality
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        csv = model_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"model_predictions_{start_date}_{end_date}.csv",
            mime="text/csv",
        )

    with col2:
        st.caption(f"Showing {len(recent_df)} recent predictions")
