"""SQL schema definitions for metrics storage."""

# Portfolio metrics table
PORTFOLIO_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_metrics (
    timestamp DATETIME PRIMARY KEY,
    total_value REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    daily_return_pct REAL,
    sharpe_ratio REAL,
    max_drawdown_pct REAL
);

CREATE INDEX IF NOT EXISTS idx_portfolio_timestamp
ON portfolio_metrics(timestamp DESC);
"""

# Trade execution metrics table
TRADE_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_metrics (
    trade_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    intended_price REAL NOT NULL,
    filled_price REAL NOT NULL,
    slippage_pct REAL NOT NULL,
    commission REAL NOT NULL,
    latency_ms REAL NOT NULL,
    order_status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_timestamp
ON trade_metrics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_trade_symbol
ON trade_metrics(symbol);
"""

# Model prediction metrics table
MODEL_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    action INTEGER NOT NULL,
    confidence REAL,
    observation TEXT,
    reward REAL
);

CREATE INDEX IF NOT EXISTS idx_model_timestamp
ON model_metrics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_model_symbol
ON model_metrics(symbol);
"""

# System health metrics table
SYSTEM_METRICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_type TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT
);

CREATE INDEX IF NOT EXISTS idx_system_timestamp
ON system_metrics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_component
ON system_metrics(component);
"""

# Daily summaries table
DAILY_SUMMARIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_summaries (
    date DATE PRIMARY KEY,
    total_trades INTEGER NOT NULL DEFAULT 0,
    win_rate REAL NOT NULL DEFAULT 0.0,
    avg_return_pct REAL NOT NULL DEFAULT 0.0,
    sharpe_ratio REAL,
    max_drawdown_pct REAL,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    uptime_pct REAL NOT NULL DEFAULT 100.0,
    error_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_daily_date
ON daily_summaries(date DESC);
"""

# System errors table
SYSTEM_ERRORS_SCHEMA = """
CREATE TABLE IF NOT EXISTS system_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    component TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    stack_trace TEXT
);

CREATE INDEX IF NOT EXISTS idx_error_timestamp
ON system_errors(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_error_component
ON system_errors(component);

CREATE INDEX IF NOT EXISTS idx_error_severity
ON system_errors(severity);
"""

# All schemas combined
ALL_SCHEMAS = [
    PORTFOLIO_METRICS_SCHEMA,
    TRADE_METRICS_SCHEMA,
    MODEL_METRICS_SCHEMA,
    SYSTEM_METRICS_SCHEMA,
    DAILY_SUMMARIES_SCHEMA,
    SYSTEM_ERRORS_SCHEMA,
]
