import json
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(layout="wide", page_title="RL Trading Strategy Report")

# ---------------------------
# Load Data - reports/backtest.json
# ---------------------------
project_root = Path(__file__).parent.parent.parent.parent
report_path = project_root / "reports" / "backtest.json"
with open(report_path) as f:
    data = json.load(f)

trades = pd.DataFrame(data["trades"])
equity = pd.DataFrame({
    "date": pd.to_datetime(data["equity_curve"]["dates"]),
    "equity": data["equity_curve"]["values"]
})
metrics = data["metrics"]
agent = data["agent_info"]
config = data["config"]
symbol = trades["symbol"].iloc[0] if hasattr(trades["symbol"], 'iloc') else trades["symbol"]

# ---------------------------
# Header
# ---------------------------
st.title("📈 PPO RL Trading Strategy – Interactive Backtest Report")
st.subheader("Symbol: " + symbol)

# ---------------------------
# Config Summary
# ---------------------------
with st.expander("⚙️ Backtest Configuration"):
    st.json(config)

# ---------------------------
# Metrics Dashboard
# ---------------------------
st.subheader("📊 Key Performance Metrics")

cols = st.columns(5)
cols[0].metric("Total Return", f"{metrics['total_return']:.2%}")
cols[1].metric("CAGR", f"{metrics['cagr']:.2%}")
cols[2].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
cols[3].metric("Max DD", f"{metrics['max_drawdown']:.2%}")
cols[4].metric("Win Rate", f"{metrics['win_rate']:.2%}")

# ---------------------------
# Equity Curve
# ---------------------------
st.subheader("💰 Equity Curve")

fig_equity = px.line(
    equity,
    x="date",
    y="equity",
    title="Equity Curve",
)
fig_equity.update_layout(height=400)
st.plotly_chart(fig_equity, use_container_width=True)

# ---------------------------
# Drawdown Curve
# ---------------------------
equity["peak"] = equity["equity"].cummax()
equity["drawdown"] = (equity["equity"] - equity["peak"]) / equity["peak"]

fig_dd = px.area(
    equity,
    x="date",
    y="drawdown",
    title="Drawdown Curve",
)
fig_dd.update_layout(height=300)
st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------
# Trades Table
# ---------------------------
st.subheader("📑 Trades")

trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"] = pd.to_datetime(trades["exit_date"])

st.dataframe(
    trades[
        [
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "quantity",
            "pnl",
            "pnl_pct",
        ]
    ],
    use_container_width=True,
)

# ---------------------------
# PnL Distribution
# ---------------------------
st.subheader("📉 Trade PnL Distribution")

fig_pnl = px.histogram(
    trades,
    x="pnl",
    nbins=30,
    title="Trade PnL Distribution",
)
st.plotly_chart(fig_pnl, use_container_width=True)

# ---------------------------
# Trade Timeline
# ---------------------------
st.subheader("⏱ Trade Timeline")

fig_timeline = go.Figure()

for _, t in trades.iterrows():
    color = "green" if t["pnl"] > 0 else "red"
    fig_timeline.add_trace(
        go.Scatter(
            x=[t["entry_date"], t["exit_date"]],
            y=[t["entry_price"], t["exit_price"]],
            mode="lines+markers",
            line=dict(color=color),
            showlegend=False,
        )
    )

fig_timeline.update_layout(
    title="Trade Entry → Exit Timeline",
    height=400,
)
st.plotly_chart(fig_timeline, use_container_width=True)

# ---------------------------
# Risk Metrics
# ---------------------------
st.subheader("⚠️ Risk Metrics")

risk_cols = st.columns(4)
risk_cols[0].metric("Volatility", f"{metrics['annualized_volatility']:.2%}")
risk_cols[1].metric("VaR (95%)", f"{metrics['var_95']:.2%}")
risk_cols[2].metric("CVaR (95%)", f"{metrics['cvar_95']:.2%}")
risk_cols[3].metric("Calmar", f"{metrics['calmar_ratio']:.2f}")


# ---------------------------
# Equity Curve with Trade Timeline
# ---------------------------
st.subheader("📈 Equity Curve with Trade Timeline")

fig = go.Figure()

# ---- Equity Curve (Primary Y-axis)
fig.add_trace(
    go.Scatter(
        x=equity["date"],
        y=equity["equity"],
        mode="lines",
        name="Equity Curve",
        fill="tozeroy",
        fillcolor="rgba(30, 144, 255, 0.25)",  # soft blue
        line=dict(color="royalblue", width=2),
        yaxis="y1",
    )
)

# ---- Trade Entry → Exit Lines (Secondary Y-axis)
for _, t in trades.iterrows():
    color = "green" if t["pnl"] > 0 else "red"

    fig.add_trace(
        go.Scatter(
            x=[t["entry_date"], t["exit_date"]],
            y=[t["entry_price"], t["exit_price"]],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
            showlegend=False,
            yaxis="y2",
        )
    )

# ---- Layout
fig.update_layout(
    height=520,
    title="Equity Curve (Area) with Trade Entry & Exit Overlay",
    xaxis=dict(title="Date"),
    yaxis=dict(
        title="Equity Value",
        side="left",
        showgrid=True,
    ),
    yaxis2=dict(
        title="Price",
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    hovermode="x unified",
)


st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Indicator Analysis
# ---------------------------

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("Configuration")

# symbol = st.sidebar.text_input("Stock Symbol", "BANKINDIA.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-12-31"))

ma_fast = st.sidebar.number_input("MA Fast", 10, 50, 20)
ma_slow = st.sidebar.number_input("MA Slow", 20, 200, 50)
bb_window = st.sidebar.number_input("BB Window", 10, 40, 20)
bb_std = st.sidebar.number_input("BB Std Dev", 1.0, 3.0, 2.0)
rsi_period = st.sidebar.number_input("RSI Period", 7, 30, 14)


# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    # 🔴 FIX: flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.astype(float)
    df.dropna(inplace=True)

    return df

print(f"Loading data for {symbol} from {start_date} to {end_date}...")
df = load_data(symbol, start_date, end_date)

if df.empty:
    st.error("No data found. Check symbol or date range.")
    st.stop()

# -----------------------
# Indicators
# -----------------------
df["MA_FAST"] = df["Close"].rolling(ma_fast).mean()
df["MA_SLOW"] = df["Close"].rolling(ma_slow).mean()

rolling_std = df["Close"].rolling(bb_window).std()
df["BB_UPPER"] = df["Close"].rolling(bb_window).mean() + bb_std * rolling_std
df["BB_LOWER"] = df["Close"].rolling(bb_window).mean() - bb_std * rolling_std

delta = df["Close"].diff()

# Ensure gain and loss are 1D arrays
gain = np.where(delta > 0, delta, 0)
if gain.ndim > 1:
    gain = gain.flatten()
loss = np.where(delta < 0, -delta, 0)
if loss.ndim > 1:
    loss = loss.flatten()

roll_up = pd.Series(gain, index=df.index).rolling(rsi_period).mean()
roll_down = pd.Series(loss, index=df.index).rolling(rsi_period).mean()
rs = roll_up / roll_down
df["RSI"] = 100 - (100 / (1 + rs))

# -----------------------
# Plot
# -----------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3],
    subplot_titles=("Price Action", "RSI")
)

# --- Moving Averages
fig.add_trace(
    go.Scatter(x=df.index, y=df["MA_FAST"], name=f"MA {ma_fast}"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["MA_SLOW"], name=f"MA {ma_slow}"),
    row=1, col=1
)

# --- Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ),
    row=1, col=1
)

# --- Bollinger Bands
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BB_UPPER"],
        name="BB Upper",
        line=dict(dash="dot"),
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["BB_LOWER"],
        name="BB Lower",
        line=dict(dash="dot"),
        fill="tonexty",
        fillcolor="rgba(135,206,250,0.2)"
    ),
    row=1, col=1
)

# --- RSI
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["RSI"],
        name="RSI",
        line=dict(color="orange")
    ),
    row=2, col=1
)

trades["entry_date"] = pd.to_datetime(trades["entry_date"])
trades["exit_date"] = pd.to_datetime(trades["exit_date"])

# -----------------------
# Trade Entry / Exit Overlay
# -----------------------
for _, t in trades.iterrows():
    color = "green" if t["pnl"] > 0 else "red"

    # Entry marker
    fig.add_trace(
        go.Scatter(
            x=[t["entry_date"]],
            y=[t["entry_price"]],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=10,
                color=color
            ),
            name="Entry",
            showlegend=False,
            hovertemplate=(
                "Entry<br>"
                "Date: %{x}<br>"
                "Price: %{y:.2f}<br>"
            ),
        ),
        row=1, col=1
    )

    # Exit marker
    fig.add_trace(
        go.Scatter(
            x=[t["exit_date"]],
            y=[t["exit_price"]],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=10,
                color=color
            ),
            name="Exit",
            showlegend=False,
            hovertemplate=(
                "Exit<br>"
                "Date: %{x}<br>"
                "Price: %{y:.2f}<br>"
            ),
        ),
        row=1, col=1
    )

    # Connecting line
    fig.add_trace(
        go.Scatter(
            x=[t["entry_date"], t["exit_date"]],
            y=[t["entry_price"], t["exit_price"]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=1
    )


fig.add_hline(y=10, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

# -----------------------
# Layout
# -----------------------
fig.update_layout(
    height=900,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.02)
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Agent Info
# ---------------------------
with st.expander("🤖 PPO Agent Configuration"):
    st.json(agent)

st.markdown("---")
st.caption("Generated by PPO RL Trading System")
