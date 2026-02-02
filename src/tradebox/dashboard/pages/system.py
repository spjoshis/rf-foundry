"""System page for health monitoring and diagnostics."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tradebox.monitoring import MetricsQuery
from tradebox.dashboard.charts import ChartBuilder
from tradebox.dashboard.utils import format_percentage, calculate_color


def render(metrics: MetricsQuery) -> None:
    """
    Render system health page.

    Args:
        metrics: MetricsQuery instance

    Displays:
        - System health status
        - Error timeline and statistics
        - API latency metrics
        - Database status
        - Recent errors/alerts
    """
    st.title("⚙️ System Health")

    # System Status Overview
    st.subheader("🏥 System Status")

    col1, col2, col3, col4 = st.columns(4)

    # Check database connectivity
    try:
        db_status = metrics.store.conn.execute("SELECT 1").fetchone()
        db_healthy = db_status is not None
    except Exception:
        db_healthy = False

    with col1:
        if db_healthy:
            st.success("✅ Database")
        else:
            st.error("❌ Database")

    # Check recent metrics (last 24 hours)
    recent_portfolio = metrics.store.query_portfolio_history(days=1)

    with col2:
        if not recent_portfolio.empty:
            st.success("✅ Metrics Collection")
        else:
            st.warning("⚠️ No Recent Metrics")

    # Check for recent errors
    recent_errors = metrics.store.query_system_errors(days=1)

    with col3:
        if recent_errors.empty:
            st.success("✅ No Errors (24h)")
        elif len(recent_errors) < 5:
            st.warning(f"⚠️ {len(recent_errors)} Errors (24h)")
        else:
            st.error(f"❌ {len(recent_errors)} Errors (24h)")

    # Database file size
    db_path = Path(metrics.store.db_path)

    with col4:
        if db_path.exists():
            db_size_mb = db_path.stat().st_size / (1024 * 1024)
            st.metric("Database Size", f"{db_size_mb:.1f} MB")
        else:
            st.error("DB Not Found")

    st.markdown("---")

    # Error Analytics
    st.subheader("🚨 Error Analytics")

    # Date range for error analysis
    col1, col2 = st.columns(2)

    with col1:
        error_start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=7),
            max_value=datetime.now().date(),
            key="error_start",
        )

    with col2:
        error_end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            key="error_end",
        )

    # Fetch errors
    errors_df = metrics.store.query_system_errors_by_date(
        start_date=error_start_date, end_date=error_end_date
    )

    if not errors_df.empty:
        # Error statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_errors = len(errors_df)
            st.metric("Total Errors", total_errors)

        with col2:
            unique_components = errors_df["component"].nunique()
            st.metric("Affected Components", unique_components)

        with col3:
            error_severity = errors_df["severity"].value_counts()
            critical_count = error_severity.get("ERROR", 0) + error_severity.get("CRITICAL", 0)
            st.metric("Critical/Error", critical_count)

        with col4:
            # Calculate error rate (errors per day)
            date_range = (error_end_date - error_start_date).days + 1
            error_rate = total_errors / date_range if date_range > 0 else 0
            st.metric("Errors/Day", f"{error_rate:.1f}")

        # Error timeline chart
        st.markdown("**Error Timeline**")

        # Aggregate by date
        errors_df["date"] = pd.to_datetime(errors_df["timestamp"]).dt.date
        daily_errors = errors_df.groupby("date").size().reset_index(name="count")

        fig = ChartBuilder.error_timeline_chart(daily_errors)
        st.plotly_chart(fig, use_container_width=True)

        # Error breakdown by component
        st.markdown("**Errors by Component**")

        component_errors = errors_df["component"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            for component, count in component_errors.head(5).items():
                pct = (count / total_errors) * 100
                st.write(f"- **{component}**: {count} ({pct:.1f}%)")

        with col2:
            # Severity distribution
            st.markdown("**Severity Distribution**")
            severity_counts = errors_df["severity"].value_counts()
            for severity, count in severity_counts.items():
                pct = (count / total_errors) * 100
                st.write(f"- {severity}: {count} ({pct:.1f}%)")

    else:
        st.success("✅ No errors found in selected date range!")

    st.markdown("---")

    # API Latency Metrics
    st.subheader("⚡ API Performance")

    # Fetch trade metrics for latency analysis
    trades_df = metrics.store.query_trades_by_date(
        start_date=error_start_date, end_date=error_end_date
    )

    if not trades_df.empty and "latency_ms" in trades_df.columns:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_latency = trades_df["latency_ms"].mean()
            st.metric("Avg Latency", f"{avg_latency:.0f} ms")

        with col2:
            median_latency = trades_df["latency_ms"].median()
            st.metric("Median Latency", f"{median_latency:.0f} ms")

        with col3:
            p95_latency = trades_df["latency_ms"].quantile(0.95)
            st.metric("P95 Latency", f"{p95_latency:.0f} ms")

        with col4:
            max_latency = trades_df["latency_ms"].max()
            st.metric("Max Latency", f"{max_latency:.0f} ms")

        # Latency distribution histogram
        import plotly.express as px

        fig = px.histogram(
            trades_df,
            x="latency_ms",
            nbins=30,
            title="Latency Distribution",
            labels={"latency_ms": "Latency (ms)", "count": "Frequency"},
        )

        fig.update_traces(marker_color="#17becf")

        fig.update_layout(
            xaxis_title="Latency (ms)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No API latency data available for selected period.")

    st.markdown("---")

    # Database Statistics
    st.subheader("💾 Database Statistics")

    col1, col2, col3 = st.columns(3)

    # Count records in each table
    try:
        portfolio_count = metrics.store.conn.execute(
            "SELECT COUNT(*) FROM portfolio_metrics"
        ).fetchone()[0]

        trade_count = metrics.store.conn.execute(
            "SELECT COUNT(*) FROM trade_metrics"
        ).fetchone()[0]

        model_count = metrics.store.conn.execute(
            "SELECT COUNT(*) FROM model_metrics"
        ).fetchone()[0]

        error_count = metrics.store.conn.execute(
            "SELECT COUNT(*) FROM system_errors"
        ).fetchone()[0]

        with col1:
            st.metric("Portfolio Records", f"{portfolio_count:,}")
            st.metric("Trade Records", f"{trade_count:,}")

        with col2:
            st.metric("Model Predictions", f"{model_count:,}")
            st.metric("Error Records", f"{error_count:,}")

        with col3:
            # Database health
            if db_path.exists():
                st.metric("DB Path", "✅ Valid")
                st.caption(str(db_path))

                # WAL mode check
                wal_mode = metrics.store.conn.execute("PRAGMA journal_mode").fetchone()[0]
                st.metric("Journal Mode", wal_mode)

    except Exception as e:
        st.error(f"Failed to query database statistics: {e}")

    st.markdown("---")

    # Recent Errors Table
    st.subheader("Recent Errors")

    if not errors_df.empty:
        # Show last 20 errors
        recent_errors = errors_df.sort_values("timestamp", ascending=False).head(20)

        # Format display
        display_df = recent_errors.copy()

        # Truncate long messages
        display_df["message_short"] = display_df["message"].apply(
            lambda x: x[:100] + "..." if len(x) > 100 else x
        )

        st.dataframe(
            display_df[
                ["timestamp", "component", "severity", "message_short"]
            ].rename(
                columns={
                    "timestamp": "Time",
                    "component": "Component",
                    "severity": "Severity",
                    "message_short": "Error Message",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

        # Expandable full error details
        with st.expander("🔍 View Full Error Details"):
            selected_idx = st.selectbox(
                "Select error to view details",
                options=range(len(recent_errors)),
                format_func=lambda i: f"{recent_errors.iloc[i]['timestamp']} - {recent_errors.iloc[i]['component']}",
            )

            if selected_idx is not None:
                error_row = recent_errors.iloc[selected_idx]

                st.markdown(f"**Timestamp:** {error_row['timestamp']}")
                st.markdown(f"**Component:** {error_row['component']}")
                st.markdown(f"**Severity:** {error_row['severity']}")
                st.markdown(f"**Message:**")
                st.code(error_row["message"])

                if pd.notna(error_row.get("stack_trace")):
                    st.markdown(f"**Stack Trace:**")
                    st.code(error_row["stack_trace"])
    else:
        st.success("✅ No errors to display!")

    # Maintenance Actions
    st.markdown("---")
    st.subheader("🛠️ Maintenance")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Stats"):
            st.rerun()

    with col2:
        if st.button("🧹 Vacuum Database"):
            try:
                metrics.store.conn.execute("VACUUM")
                st.success("Database vacuumed successfully!")
            except Exception as e:
                st.error(f"Vacuum failed: {e}")

    with col3:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
