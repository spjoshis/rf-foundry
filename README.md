# RL-Foundry

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready **reinforcement learning-based trading system** for Indian equity markets. RL-Foundry combines deep learning, technical analysis, and intelligent regime detection to create adaptive trading strategies for both end-of-day (EOD) swing trading and high-frequency intraday trading (5-minute bars).

## Table of Contents

- [Overview](#overview)
- [Why RL-Foundry?](#why-rl-foundry)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Configuration Examples](#configuration-examples)
- [Advanced Features](#advanced-features)
- [Model Inference & Prediction](#model-inference--prediction)
- [Important Constraints](#important-constraints)
- [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
- [Performance Tips & Best Practices](#performance-tips--best-practices)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Overview

RL-Foundry is a sophisticated trading system designed for managing capital with rigorous risk management, point-in-time correctness in data handling, and systematic performance validation. The system employs state-of-the-art reinforcement learning algorithms (PPO) with advanced deep learning architectures including CNNs for price pattern recognition, regime detection for market state awareness, and intelligent action masking for constraint-aware decision making.

**Tech Stack**: Python 3.10+, Gymnasium, Stable-Baselines3, PyTorch, Streamlit, Optuna, scikit-learn

## Why RL-Foundry?

### Key Differentiators

**Production-Ready Architecture**: Unlike academic RL implementations, RL-Foundry is built for real-world deployment with proper risk management, monitoring, and broker integration.

**Point-in-Time Correctness**: Strict adherence to temporal data integrity prevents look-ahead bias—a common pitfall in trading systems that leads to unrealistic backtest results.

**Regime-Aware Trading**: Automatic market regime detection (bull/bear/sideways) with intelligent action masking prevents poor decisions in unfavorable market conditions.

**Explainable AI**: Built-in explainability features help understand why the model makes specific trading decisions, crucial for building trust and debugging.

**Dual Timeframe Support**: Trade both EOD (swing trading) and intraday (5-minute bars) with the same codebase, allowing strategy diversification.

**Indian Market Optimized**: Tailored for Indian equity markets with realistic transaction costs (STT, GST, stamp duty), market hours, circuit breakers, and long-only constraints.

**Comprehensive Testing**: Extensive unit and integration tests ensure reliability and correctness across all components.

**End-to-End Pipeline**: From data ingestion to production deployment, every component is included and battle-tested.

## Features

### Core Trading System
- **Dual Environment Support**:
  - EOD Trading: End-of-day swing trading with daily bars
  - Intraday Trading: 5-minute bar trading (75 bars/session, Indian market hours 9:15-15:30)
- **Gymnasium-Compatible RL Environment**: Fully customizable with realistic transaction costs, slippage, and multi-asset sampling
- **PPO with Custom Networks**: Leverages Stable-Baselines3 with custom CNN and hybrid feature extractors
- **Advanced Technical Analysis**: 25+ indicators including RSI, MACD, ATR, Bollinger Bands, SMA/EMA, ADX, and more

### Deep Learning & AI
- **CNN-Based Feature Extraction**: Convolutional networks for raw OHLCV price pattern recognition
- **Hybrid Architecture**: Combines CNN for price patterns with MLP for technical indicators
- **Regime Detection**: Automatic bull/bear/sideways market state detection using volatility and trend analysis
- **Action Masking**: Regime-aware action constraints (e.g., no buying in strong downtrends)
- **Explainability Module**:
  - Integrated Captum for feature attribution and saliency analysis
  - Attention visualization for understanding model decisions
  - Text-based trade explanations for interpretability

### Risk Management & Execution
- **Comprehensive Risk Management**: Pre-trade validators, circuit breakers, position sizing strategies
- **Position Sizing**: Multiple strategies including Fixed, Kelly Criterion, Volatility-based, and Risk Parity
- **Transaction Cost Modeling**: Realistic Indian market costs (STT, GST, stamp duty, brokerage)
- **Broker Integration**:
  - Abstract broker interface
  - Paper trading implementation
  - Zerodha Kite API integration with retry logic and error handling

### Data & Features
- **Point-in-Time Correct Pipeline**: Temporal splits, forward-fill validation, no look-ahead bias
- **Parquet-Based Storage**: 10x compression vs CSV, efficient columnar format
- **Multi-Source Data**: Yahoo Finance integration, extensible to NSE bhavcopy
- **Feature Engineering**: Automated technical indicator calculation with configurable parameters

### Training & Optimization
- **Configuration-Driven**: YAML-based experiment configs for reproducibility
- **Hyperparameter Tuning**: Optuna integration for automated optimization
- **Custom Callbacks**: Training monitoring, checkpointing, early stopping
- **Vectorized Environments**: Parallel training with multiple environment instances

### Backtesting & Analysis
- **Robust Backtesting Engine**: Walk-forward validation, multiple performance metrics
- **Performance Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, win rate
- **Benchmark Comparisons**: Compare against buy-and-hold and market indices
- **Detailed Reporting**: HTML reports with equity curves, trade analysis, and risk metrics

### Production & Deployment
- **Orchestration System**: Automated workflow scheduling for paper/live trading
- **Real-Time Monitoring**:
  - SQLite-based metrics collection
  - Event-driven architecture for system monitoring
  - Performance tracking and aggregation
- **Streamlit Dashboard**:
  - Interactive web interface for monitoring
  - Real-time portfolio visualization
  - Model performance analytics
  - System health monitoring
- **Paper Trading Deployment**: Safe testing environment before live trading
- **Production Logging**: Structured logging with loguru for debugging and audit trails

## Requirements

- **Python**: 3.10 or higher
- **Poetry**: For dependency management (see installation below)
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: Minimum 8GB RAM (16GB+ recommended for vectorized environments)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/RL-Foundry.git
cd RL-Foundry
```

### 2. Install Poetry (if not already installed)

On macOS/Linux:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

On Windows (PowerShell):
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

For other installation methods, see [Poetry Documentation](https://python-poetry.org/docs/#installation).

### 3. Install Dependencies

```bash
poetry install
```

This installs all project dependencies and development tools as specified in `pyproject.toml`.

### 4. Set Up Pre-Commit Hooks (Optional but Recommended)

```bash
poetry run pre-commit install
```

This ensures code quality checks (black formatting, ruff linting, mypy type checking) run automatically before each commit.

## Quick Start

### 1. Train an EOD Trading Agent

Train a baseline PPO agent with technical indicators:
```bash
poetry run python scripts/train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --timesteps 500000 \
    --n-envs 4
```

Monitor training progress in real-time:
```bash
tensorboard --logdir logs/
```
Then open your browser to `http://localhost:6006`.

### 2. Train with Advanced Features

**Regime Detection & Action Masking:**
```bash
poetry run python scripts/train.py \
    --config configs/experiments/regime_action_mask.yaml
```

**Active Trading (Frequent Signals):**
```bash
poetry run python scripts/train.py \
    --config configs/experiments/active_trading.yaml
```

**CNN-Based Feature Extraction:**
```bash
poetry run python scripts/train_cnn_example.py
```

### 3. Train Intraday Trading Agent

Train on 5-minute bars for intraday trading:
```bash
poetry run python scripts/train_intraday.py \
    --symbol RELIANCE.NS \
    --start 2020-01-01 \
    --timesteps 500000 \
    --n-envs 4
```

### 4. Run Predictions

Make predictions with a trained model:
```bash
# Basic inference (10 steps with detailed logging)
poetry run python scripts/inference.py \
    --model models/exp007_active_trading_20251219_235451 \
    --symbol RELIANCE.NS

# Custom steps with simple output
poetry run python scripts/inference.py \
    --model models/exp007_active_trading_20251219_235451 \
    --symbol RELIANCE.NS \
    --steps 50 \
    --output-format simple

# Save predictions to JSON for analysis
poetry run python scripts/inference.py \
    --model models/exp007_active_trading_20251219_235451 \
    --symbol TCS.NS \
    --steps 100 \
    --save-log predictions_tcs.json
```

### 5. Run Backtests

Backtest a trained model:
```bash
poetry run python scripts/backtest.py \
    --model models/ppo_best.zip \
    --symbol RELIANCE.NS \
    --start 2023-01-01 \
    --end 2023-12-31
```

### 6. Launch Dashboard

Start the Streamlit dashboard for monitoring:
```bash
poetry run python scripts/dashboard.py --port 8501
```

Access the dashboard at `http://localhost:8501` to view:
- Real-time portfolio performance
- Trade history and analytics
- Model performance metrics
- System health monitoring

### 7. Deploy to Paper Trading

Deploy a trained model to paper trading:
```bash
poetry run python scripts/deploy_paper_trading.py \
    --model models/ppo_best.zip \
    --config configs/orchestration/paper_eod.yaml
```

### 8. Validate Regime Detection

Visualize regime detection on market data:
```bash
poetry run python scripts/validate_regime.py \
    --symbol ^NSEI \
    --start 2020-01-01 \
    --end 2024-12-31
```

## Project Structure

```
tradebox-rl/
├── src/tradebox/                    # Main package
│   ├── data/                        # Data loading, validation, caching
│   │   ├── loaders/
│   │   │   └── yahoo_loader.py      # Yahoo Finance data loader
│   │   ├── validation.py            # Data quality checks
│   │   └── splitter.py              # Time-series train/val/test splitting
│   ├── features/                    # Feature engineering
│   │   ├── technical.py             # Technical indicators (RSI, MACD, SMA, etc.)
│   │   ├── extractor.py             # Feature extraction pipeline
│   │   ├── analyzer.py              # Feature analysis utilities
│   │   └── regime.py                # Market regime detection (bull/bear/sideways)
│   ├── env/                         # Gymnasium trading environments
│   │   ├── trading_env.py           # EOD trading environment
│   │   ├── intraday_env.py          # Intraday 5-minute bar environment
│   │   ├── rewards.py               # Reward function implementations
│   │   ├── costs.py                 # Transaction cost modeling
│   │   ├── action_mask.py           # Regime-based action masking
│   │   └── wrappers.py              # Environment wrappers and utilities
│   ├── models/                      # Deep learning models
│   │   ├── cnn_extractor.py         # CNN for OHLCV pattern recognition
│   │   ├── hybrid_extractor.py      # Hybrid CNN + MLP architecture
│   │   ├── feature_extractor.py     # Custom feature extractors
│   │   └── trading_cnn.py           # Trading-specific CNN architectures
│   ├── agents/                      # RL training pipeline
│   │   ├── base_agent.py            # Base agent class
│   │   ├── callbacks.py             # Training callbacks (checkpointing, logging)
│   │   └── serialization.py         # Model save/load utilities
│   ├── backtest/                    # Backtesting engine
│   │   ├── config.py                # Backtest configuration
│   │   ├── metrics.py               # Performance metrics calculation
│   │   ├── analyzer.py              # Backtest analysis
│   │   └── report.py                # Report generation
│   ├── risk/                        # Risk management
│   │   ├── validators.py            # Pre-trade validators
│   │   ├── position_sizers.py       # Position sizing strategies
│   │   └── circuit_breakers.py      # Circuit breaker logic
│   ├── execution/                   # Broker APIs
│   │   ├── base_broker.py           # Abstract broker interface
│   │   ├── paper_broker.py          # Paper trading broker
│   │   ├── kite_broker.py           # Zerodha Kite integration
│   │   ├── retry.py                 # Retry logic for API calls
│   │   ├── exceptions.py            # Execution exceptions
│   │   └── config.py                # Execution configuration
│   ├── orchestration/               # Automated trading workflows
│   │   ├── workflow.py              # Workflow orchestration
│   │   ├── scheduler.py             # Job scheduling
│   │   ├── state.py                 # State management
│   │   ├── config.py                # Orchestration config
│   │   └── exceptions.py            # Orchestration exceptions
│   ├── monitoring/                  # Production monitoring
│   │   ├── events.py                # Event definitions
│   │   ├── collector.py             # Metrics collection
│   │   ├── store.py                 # SQLite metrics storage
│   │   ├── aggregator.py            # Metrics aggregation
│   │   ├── query.py                 # Query interface
│   │   └── schemas.py               # Data schemas
│   ├── dashboard/                   # Streamlit web dashboard
│   │   ├── app.py                   # Main dashboard app
│   │   ├── pages/
│   │   │   ├── overview.py          # Portfolio overview page
│   │   │   ├── trading.py           # Trading analytics page
│   │   │   ├── model.py             # Model performance page
│   │   │   └── system.py            # System monitoring page
│   │   ├── charts.py                # Chart components
│   │   └── utils.py                 # Dashboard utilities
│   ├── explainability/              # Model interpretability
│   │   ├── trade_explainer.py       # Trade decision explanations
│   │   ├── attention_viz.py         # Attention visualization
│   │   └── text_generator.py        # Natural language explanations
│   └── utils/                       # General utilities
├── configs/                         # Configuration files
│   ├── experiments/                 # Experiment configs
│   │   ├── exp001_baseline.yaml     # Baseline PPO
│   │   ├── active_trading.yaml      # Frequent trading signals
│   │   ├── regime_detection.yaml    # With regime detection
│   │   ├── regime_action_mask.yaml  # Regime-aware action masking
│   │   └── frequent_trading*.yaml   # High-frequency experiments
│   ├── execution/                   # Broker configs
│   │   └── kite_broker.yaml         # Zerodha Kite configuration
│   ├── orchestration/               # Workflow configs
│   │   ├── paper_eod.yaml           # Paper trading EOD
│   │   └── live_eod.yaml            # Live trading EOD
│   └── position_sizing/             # Position sizing configs
│       ├── fixed.yaml               # Fixed position sizing
│       ├── kelly.yaml               # Kelly Criterion
│       ├── volatility.yaml          # Volatility-based
│       └── risk_parity.yaml         # Risk parity
├── scripts/                         # CLI entry points
│   ├── train.py                     # Train EOD agent
│   ├── train_intraday.py            # Train intraday agent
│   ├── train_cnn_example.py         # Train with CNN features
│   ├── backtest.py                  # Run backtests
│   ├── inference.py                 # Model inference/prediction
│   ├── predict.py                   # Alternative prediction script
│   ├── dashboard.py                 # Launch Streamlit dashboard
│   ├── deploy_paper_trading.py      # Deploy to paper trading
│   ├── orchestrate.py               # Run orchestration workflows
│   ├── validate_regime.py           # Validate regime detection
│   ├── compare_eod_vs_intraday.py   # Compare trading strategies
│   ├── generate_intraday_report.py  # Generate intraday reports
│   └── test_*.py                    # Various testing scripts
├── tests/                           # Unit and integration tests
│   ├── unit/                        # Unit tests (by module)
│   │   ├── test_data/
│   │   ├── test_features/
│   │   ├── test_env/
│   │   ├── test_agents/
│   │   ├── test_risk/
│   │   └── test_execution/
│   └── integration/                 # Integration tests
├── examples/                        # Example notebooks and scripts
├── docs/                            # Documentation
├── data/                            # Data storage (gitignored)
│   ├── raw/                         # Raw OHLCV data
│   └── processed/                   # Processed features
├── models/                          # Saved model weights (gitignored)
├── logs/                            # Training logs & TensorBoard (gitignored)
├── reports/                         # Generated reports (gitignored)
├── cache/                           # Cache directory (gitignored)
├── pyproject.toml                   # Poetry configuration
├── poetry.lock                      # Dependency lock file
└── README.md                        # This file
```

## Development Workflow

### Running Tests

Run all tests with coverage:
```bash
poetry run pytest --cov=src/tradebox --cov-report=html
```

Run a specific test module:
```bash
poetry run pytest tests/unit/test_data/test_yahoo_loader.py -v
```

Run integration tests only:
```bash
poetry run pytest tests/integration/ -v
```

Run test rewards:
```
poetry run pytest tests/unit/test_env/test_rewards.py -v --cov=src/tradebox/env/rewards
```

### Code Quality Checks

**Format code with Black**:
```bash
poetry run black src/ tests/ scripts/
```

**Lint with Ruff**:
```bash
poetry run ruff check src/ tests/ scripts/
```

**Type checking with mypy**:
```bash
poetry run mypy src/tradebox
```

**Run all quality checks** (pre-commit):
```bash
poetry run pre-commit run --all-files
```

### Code Engineering Standards

All code follows production-grade Python standards:

- **Type Hints**: All function signatures use type hints from `typing` module
- **Docstrings**: Google or NumPy style with parameters, returns, exceptions
- **Error Handling**: Custom exceptions with proper logging
- **Testing**: Pytest with comprehensive unit and integration test coverage
- **Formatting**: PEP 8 compliance via Black + Ruff
- **Type Checking**: Static type checking with mypy
- **Design**: SOLID principles and appropriate design patterns
- **Security**: Input validation, no hardcoded secrets, secure API handling
- **Pre-commit Hooks**: Automated quality checks before commits

## Configuration Examples

RL-Foundry uses YAML-based configuration for reproducible experiments. Here are some examples:

### Baseline PPO Configuration

```yaml
# configs/experiments/exp001_baseline.yaml
experiment:
  name: exp001_baseline_ppo
  description: Baseline PPO with technical indicators

data:
  symbols: ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

features:
  technical:
    sma_periods: [20, 50, 200]
    rsi_period: 14
    atr_period: 14
    use_macd: true
    use_bbands: true

env:
  action_space: "Discrete(3)"  # Hold, Buy, Sell
  reward_type: "risk_adjusted"
  initial_capital: 100000.0
  lookback_window: 60
  max_episode_steps: 500

cost:
  brokerage_pct: 0.0003  # 0.03%
  stt_pct: 0.001         # 0.1%
  transaction_cost_pct: 0.00025
  slippage_pct: 0.0005

agent:
  algorithm: "PPO"
  learning_rate: 0.0003
  network_arch: [256, 256]
  batch_size: 64
  n_epochs: 10

training:
  total_timesteps: 2000000
  n_envs: 8
  eval_freq: 10000
  save_freq: 50000
```

### Regime-Aware Action Masking

```yaml
# configs/experiments/regime_action_mask.yaml
experiment:
  name: regime_action_mask
  description: Action masking based on market regime

features:
  regime_detection:
    enabled: true
    lookback_period: 20
    volatility_threshold: 0.02
    trend_threshold: 0.01

action_mask:
  enabled: true
  mask_buy_in_downtrend: true      # No buying in strong downtrends
  mask_sell_in_uptrend: true       # No selling in strong uptrends
  downtrend_threshold: -0.02       # -2% trend
  uptrend_threshold: 0.02          # +2% trend
```

### Position Sizing Strategies

```yaml
# configs/position_sizing/kelly.yaml
position_sizing:
  strategy: "kelly"
  kelly_fraction: 0.25       # Use 25% of full Kelly (for safety)
  max_position_size: 0.5     # Never exceed 50% of capital
  min_position_size: 0.05    # Minimum 5% of capital

# configs/position_sizing/volatility.yaml
position_sizing:
  strategy: "volatility"
  target_volatility: 0.15    # Target 15% annualized volatility
  atr_period: 14
  atr_multiplier: 2.0
```

### Orchestration for Paper Trading

```yaml
# configs/orchestration/paper_eod.yaml
orchestration:
  mode: "paper"              # paper or live
  schedule: "0 15 30 * * *"  # 3:30 PM daily (after market close)

  workflow:
    - fetch_latest_data
    - generate_features
    - load_model
    - generate_signals
    - validate_signals
    - execute_trades
    - update_portfolio
    - log_metrics

  broker:
    type: "paper"
    initial_capital: 100000.0

  risk_limits:
    max_position_size: 0.3
    max_daily_loss: 0.02       # 2% max daily loss
    max_drawdown: 0.10         # 10% max drawdown
```

## Important Constraints

### Indian Market Specifics
- **No Shorting**: Long-only constraint for retail traders
- **Transaction Costs**: STT, GST, stamp duty = 0.15-0.2% round-trip
- **Market Hours**: 9:15 AM - 3:30 PM IST
- **Settlement**: T+1 for equity delivery
- **Circuit Breakers**: 10%, 15%, 20% price limits

### Non-Stationarity
- Market regimes change (bull/bear/sideways)
- Feature-return relationships drift over time
- **Solution**: Online learning, quarterly retraining, drift detection

### Data Quality
- Stock splits and dividends require adjustment
- Corporate actions (mergers, delistings) need handling
- Survivorship bias (only current NIFTY 50, not historical members)

## Common Pitfalls to Avoid

1. **Look-ahead Bias**: Using future information in features (always use point-in-time data)
2. **Data Snooping**: Optimizing on test set (use validation set for hyperparameter tuning)
3. **Overfitting to Backtests**: Training until test performance peaks (use early stopping on validation set)
4. **Ignoring Transaction Costs**: Unrealistic strategies with excessive trading frequency
5. **Survivorship Bias**: Only testing on currently successful stocks (include delisted stocks)
6. **Insufficient Testing**: Not testing error handling, edge cases, and failure modes
7. **Direct to Production**: Skipping paper trading validation (always test in paper mode first)
8. **Excessive Complexity**: Over-engineering before validating basic approach (start simple)
9. **Ignoring Regime Changes**: Not accounting for market regime shifts (use regime detection)
10. **Poor Risk Management**: Not implementing proper position sizing and stop losses
11. **Inadequate Monitoring**: Not tracking performance metrics and model drift in production
12. **Data Quality Issues**: Not validating data quality, handling missing values, and corporate actions

## Dependencies

Key dependencies managed via Poetry:

- **RL Framework**: `gymnasium`, `stable-baselines3`
- **Deep Learning**: `torch`, `captum` (for explainability)
- **Data Processing**: `pandas`, `numpy`, `pyarrow`, `scikit-learn`
- **Technical Analysis**: `ta` (Technical Analysis library)
- **Data Sources**: `yfinance`
- **Configuration**: `pydantic`, `omegaconf`, `hydra-core`, `pyyaml`
- **Optimization**: `optuna` (hyperparameter tuning)
- **Monitoring & Visualization**: `tensorboard`, `matplotlib`, `plotly`, `seaborn`, `streamlit`
- **Broker Integration**: `kiteconnect` (Zerodha Kite API)
- **Logging**: `loguru`
- **Utilities**: `tabulate`
- **Development**: `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `pre-commit`

For complete list and versions, see `pyproject.toml`.

## Advanced Features

### Regime Detection & Validation

RL-Foundry includes sophisticated market regime detection that automatically identifies bull, bear, and sideways markets:

```bash
# Validate regime detection visually
poetry run python scripts/validate_regime.py \
    --symbol ^NSEI \
    --start 2020-01-01 \
    --end 2024-12-31
```

This generates visualization showing:
- Price movements with regime overlays
- Regime indicators (volatility, trend strength)
- Regime transition points
- Distribution statistics

### Streamlit Dashboard

Launch an interactive web dashboard for real-time monitoring:

```bash
poetry run python scripts/dashboard.py --port 8501
```

Dashboard features:
- **Overview Page**: Portfolio value, P&L, key metrics
- **Trading Page**: Trade history, position analysis, performance breakdown
- **Model Page**: Model metrics, prediction accuracy, feature importance
- **System Page**: System health, resource usage, error logs

### Model Explainability

Use the explainability module to understand model decisions:

```python
from tradebox.explainability import TradeExplainer

explainer = TradeExplainer(model, env)
explanation = explainer.explain_trade(observation)
print(explanation.text_summary)  # Natural language explanation
explainer.plot_feature_importance()  # Visual attribution
```

Features:
- Captum integration for gradient-based attribution
- Attention visualization for CNN models
- Natural language trade explanations
- Feature importance ranking

### Production Deployment

Deploy to paper trading for validation:

```bash
poetry run python scripts/deploy_paper_trading.py \
    --model models/ppo_best.zip \
    --config configs/orchestration/paper_eod.yaml
```

Monitor deployed system:
```bash
# Check system health
poetry run python scripts/test_paper_deployment.py

# View metrics in dashboard
poetry run python scripts/dashboard.py --db-path data/metrics.db
```

### Hyperparameter Optimization

Use Optuna for automated hyperparameter tuning:

```python
# See configs/experiments/ for examples of Optuna integration
poetry run python scripts/train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --optimize \
    --n-trials 50
```

### Intraday vs EOD Comparison

Compare EOD and intraday strategies:

```bash
poetry run python scripts/compare_eod_vs_intraday.py \
    --eod-model models/eod_ppo.zip \
    --intraday-model models/intraday_ppo.zip \
    --symbol RELIANCE.NS
```

## Model Inference & Prediction

The `scripts/inference.py` script provides comprehensive prediction capabilities:

### Core Functionality
- Loads saved PPO agents from checkpoints
- Creates trading environments with real market data
- Runs step-by-step predictions with detailed logging
- Supports multiple output formats for analysis

### CLI Arguments
- `--model` (required): Path to trained model directory or .zip file
- `--symbol` (required): Stock symbol (e.g., RELIANCE.NS, TCS.NS, ^NSEI)
- `--steps`: Number of prediction steps (default: 10)
- `--episode`: Run full episode until termination
- `--env-type`: Environment type (eod/intraday)
- `--output-format`: Logging format (simple/detailed/json)
- `--save-log`: Save predictions to JSON file
- `--verbose`: Enable debug-level logging

### Output Formats

1. **Detailed** (default): Complete market state, portfolio state, and agent decisions with timestamps
2. **Simple**: Compact single-line format per step for quick review
3. **JSON**: Structured output saved to file for programmatic analysis

### Usage Examples

```bash
# Basic usage (10 steps with detailed logging)
poetry run python scripts/inference.py \
    --model models/exp007_active_trading_20251219_235451 \
    --symbol RELIANCE.NS

# Custom steps with simple output
poetry run python scripts/inference.py \
    --model models/exp007_active_trading_20251219_235451 \
    --symbol RELIANCE.NS \
    --steps 50 \
    --output-format simple

# Save predictions to JSON for analysis
poetry run python scripts/inference.py \
    --model models/exp007_active_trading_20251219_235451 \
    --symbol TCS.NS \
    --steps 100 \
    --save-log predictions_tcs.json

# Run full episode with intraday environment
poetry run python scripts/inference.py \
    --model models/intraday_ppo.zip \
    --symbol RELIANCE.NS \
    --episode \
    --env-type intraday
```

## Performance Tips & Best Practices

### Training Optimization

1. **Start Simple**: Begin with baseline PPO before adding complex features
2. **Use Vectorized Environments**: Train with `n_envs=4` or higher for faster convergence
3. **Monitor Validation Performance**: Use callbacks to track validation Sharpe ratio
4. **Hyperparameter Tuning**: Use Optuna for systematic optimization
5. **Feature Selection**: Start with fewer features and add incrementally
6. **Normalize Observations**: Ensure features are on similar scales
7. **Appropriate Lookback**: 60 bars for EOD, 60-120 bars for intraday

### Model Selection

- **EOD Trading**: Start with MLP networks, add CNN if needed
- **Intraday Trading**: CNN or hybrid architecture recommended for pattern recognition
- **Regime-Based**: Enable action masking for regime-aware trading
- **High Frequency**: Consider smaller networks for faster inference

### Risk Management

1. Always implement position sizing (Kelly, volatility-based, or risk parity)
2. Set maximum position size limits (e.g., 30% of capital)
3. Use circuit breakers for drawdown protection
4. Model realistic transaction costs
5. Include slippage in simulations
6. Test with different market regimes

### Data Quality

- Validate data for missing values, outliers, and corporate actions
- Use point-in-time data to avoid look-ahead bias
- Implement forward-fill with validation
- Handle stock splits and dividends correctly
- Test on out-of-sample data (not used in training/validation)

### Production Deployment

1. **Always start with paper trading** for at least 2-3 months
2. Monitor for model drift using validation metrics
3. Implement proper logging and alerting
4. Use the dashboard for real-time monitoring
5. Set up automated daily health checks
6. Keep human oversight in the loop
7. Start with small capital and scale gradually

## Troubleshooting

### Common Issues

**Training is slow:**
- Increase `n_envs` for parallel environments
- Use smaller network architecture
- Reduce `max_episode_steps`
- Use `--quick` flag for testing

**Poor validation performance:**
- Check for look-ahead bias in features
- Reduce network complexity (may be overfitting)
- Increase training timesteps
- Try different reward functions
- Enable regime detection

**High transaction costs:**
- Reduce trading frequency (adjust reward penalties)
- Increase position holding time incentives
- Use action masking to prevent excessive trading
- Verify cost model parameters

**Model not learning:**
- Check observation space is properly normalized
- Verify reward function is providing learning signal
- Try different learning rates
- Increase network size
- Check for data quality issues

**Errors during deployment:**
- Verify broker credentials and API access
- Check network connectivity
- Validate data feed availability
- Review logs in `logs/` directory
- Use `--verbose` flag for detailed debugging

### Getting Help

- Check existing issues on GitHub
- Review test files for usage examples
- Enable debug logging with `--verbose`
- Examine TensorBoard logs for training insights
- Use the dashboard for system monitoring

## Contributing

Contributions are welcome! Areas for improvement:

- Additional technical indicators
- More reward function variants
- Enhanced regime detection algorithms
- Support for more data sources (NSE, BSE)
- Additional broker integrations
- Improved explainability features
- Documentation and examples

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

Built with:
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for environment interface
- [PyTorch](https://pytorch.org/) for deep learning
- [Streamlit](https://streamlit.io/) for dashboard
- [yfinance](https://github.com/ranaroussi/yfinance) for market data

## Disclaimer

**IMPORTANT - READ CAREFULLY:**

This software is provided for **educational and research purposes only**.

### Risk Warning

- Trading financial instruments carries **substantial risk of financial loss**, including complete loss of principal
- **Past performance is not indicative of future results**
- No warranty or guarantee of profitability is provided
- The markets can be volatile and unpredictable

### Legal Disclaimer

- The authors and contributors are **not responsible for any financial losses** incurred through use of this software
- This software is provided "as is" without warranty of any kind
- Users are solely responsible for their trading decisions and financial outcomes
- Not financial advice - consult with qualified financial advisors before trading

### Best Practices

Before deploying any live trading system:

1. **Thorough backtesting**: Test extensively on historical data
2. **Paper trading**: Run in paper mode for minimum 2-3 months
3. **Start small**: Begin with minimal capital you can afford to lose
4. **Strict risk management**: Implement position sizing and stop losses
5. **Continuous monitoring**: Track performance and model drift
6. **Professional advice**: Consult financial and legal professionals
7. **Regulatory compliance**: Ensure compliance with local trading regulations

**USE AT YOUR OWN RISK**

