"""Backtesting engine for RL trading agents."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from tradebox.agents.base_agent import BaseAgent
from tradebox.backtest.config import BacktestConfig
from tradebox.env import TradingEnv, EnvConfig


@dataclass
class Trade:
    """
    Record of a single trade.

    Attributes:
        entry_date: Date of entry
        exit_date: Date of exit (None if still open)
        symbol: Stock symbol
        action: 'buy' or 'sell'
        entry_price: Price at entry
        exit_price: Price at exit (None if still open)
        quantity: Number of shares
        commission: Total commission paid
        pnl: Profit/loss (None if still open)
        pnl_pct: P&L percentage (None if still open)
        duration_days: Trade duration in days
    """

    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    commission: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    duration_days: Optional[int]


@dataclass
class BacktestResult:
    """
    Complete backtest results.

    Attributes:
        config: Backtest configuration used
        trades: List of all trades executed
        equity_curve: Portfolio value over time
        daily_returns: Daily returns series
        positions: Position history DataFrame
        metrics: Performance metrics dictionary
        agent_info: Information about the agent used
    """

    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict[str, Any] = field(default_factory=dict)
    agent_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "initial_capital": self.config.initial_capital,
                "commission_pct": self.config.commission_pct,
                "slippage_pct": self.config.slippage_pct,
            },
            "trades": [
                {
                    "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                    "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                    "symbol": t.symbol,
                    "action": t.action,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                }
                for t in self.trades
            ],
            "metrics": self.metrics,
            "agent_info": self.agent_info,
        }


class BacktestEngine:
    """
    Backtest engine for RL trading agents.

    Runs a trained agent on historical data and tracks all trades,
    portfolio value, and performance metrics.

    Example:
        >>> from tradebox.agents import PPOAgent
        >>> from tradebox.backtest import BacktestEngine, BacktestConfig
        >>>
        >>> agent = PPOAgent.load("models/ppo_best.zip")
        >>> config = BacktestConfig(initial_capital=100000)
        >>> engine = BacktestEngine(config)
        >>>
        >>> result = engine.run(
        ...     agent=agent,
        ...     data=test_data,
        ...     features=test_features,
        ...     symbol="RELIANCE.NS"
        ... )
        >>>
        >>> print(f"Final value: ₹{result.equity_curve.iloc[-1]:,.0f}")
        >>> print(f"Total trades: {len(result.trades)}")
        >>> print(f"Sharpe ratio: {result.metrics['sharpe_ratio']:.2f}")
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()
        logger.info(f"BacktestEngine initialized with capital: ₹{self.config.initial_capital:,.0f}")

    def run(
        self,
        agent: BaseAgent,
        data: pd.DataFrame,
        features: pd.DataFrame,
        symbol: str = "STOCK",
        env_config: Optional[EnvConfig] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            agent: Trained RL agent
            data: OHLCV DataFrame with Date index
            features: Technical features DataFrame (aligned with data)
            symbol: Stock symbol for logging
            env_config: Optional environment config (uses defaults if None)

        Returns:
            BacktestResult with trades, equity curve, and metrics

        Raises:
            ValueError: If data/features are misaligned or invalid
        """
        logger.info(f"Starting backtest for {symbol}: {len(data)} days")

        # Validate inputs
        if len(data) != len(features):
            raise ValueError(
                f"Data ({len(data)}) and features ({len(features)}) length mismatch"
            )

        # Create environment
        if env_config is None:
            env_config = EnvConfig(
                initial_capital=self.config.initial_capital,
                lookback_window=60,
                max_episode_steps=len(data) - 61,
            )

        env = TradingEnv(data, features, env_config)

        # Initialize tracking
        trades: List[Trade] = []
        equity_curve_values: List[float] = []
        equity_curve_dates: List[datetime] = []
        positions_history: List[Dict[str, Any]] = []

        # Track current position
        current_position: Optional[Dict[str, Any]] = None

        # Run episode
        obs, info = env.reset()
        done = False
        step_count = 0

        while not done:
            # Get agent action with action masking support
            # Extract action mask from info dict if available (for regime-conditioned masking)
            action_mask = info.get("action_mask", None)
            action, _ = agent.predict(obs, deterministic=True, action_mask=action_mask)

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            logger.info(f"Step {step_count}:")
            logger.info(f"  Regime: {info.get('regime_state')}, Bias: {info.get('trend_bias')}")
            logger.info(f"  Action mask: {action_mask}")
            logger.info(f"  Predicted action: {action}")
            # logger.info(f"  Action distribution: {model.policy.get_distribution(obs).distribution.probs}")


            # Get current state
            current_date = pd.to_datetime(data.iloc[env.current_step]["Date"])
            current_price = float(data.iloc[env.current_step]["Close"])
            portfolio_value = info["portfolio_value"]
            position = info["position"]
            cash = info["cash"]

            # Record equity curve
            equity_curve_values.append(portfolio_value)
            equity_curve_dates.append(current_date)

            # Track position changes
            positions_history.append({
                "date": current_date,
                "position": position,
                "cash": cash,
                "portfolio_value": portfolio_value,
                "price": current_price,
            })

            # Detect trades (position changes)
            if action == 1 and position > 0 and current_position is None:
                # New buy
                current_position = {
                    "entry_date": current_date,
                    "entry_price": current_price,
                    "quantity": position,
                    "action": "buy",
                }

                # Commented to reduct context window - Gopal
                # logger.debug(
                #     f"BUY: {position} shares @ ₹{current_price:.2f} on {current_date.date()}"
                # )

            elif action == 2 and position == 0 and current_position is not None:
                # Sell (close position)
                commission = self.config.commission_pct * (
                    current_position["entry_price"] * current_position["quantity"]
                    + current_price * current_position["quantity"]
                )
                pnl = (
                    current_price - current_position["entry_price"]
                ) * current_position["quantity"] - commission
                pnl_pct = pnl / (
                    current_position["entry_price"] * current_position["quantity"]
                )
                duration = (current_date - current_position["entry_date"]).days

                trade = Trade(
                    entry_date=current_position["entry_date"],
                    exit_date=current_date,
                    symbol=symbol,
                    action="buy_sell",
                    entry_price=current_position["entry_price"],
                    exit_price=current_price,
                    quantity=current_position["quantity"],
                    commission=commission,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    duration_days=duration,
                )
                trades.append(trade)

                # Commented to reduct context window - Gopal
                # logger.debug(
                #     f"SELL: {current_position['quantity']} shares @ ₹{current_price:.2f} "
                #     f"on {current_date.date()}, P&L: ₹{pnl:.2f} ({pnl_pct*100:.2f}%)"
                # )

                current_position = None

            step_count += 1

        # Close any open position at end
        if current_position is not None:
            final_date = data.index[-1] if isinstance(data.index[-1], pd.Timestamp) else data.iloc[-1]["Date"]
            final_price = float(data.iloc[-1]["Close"])
            commission = self.config.commission_pct * (
                current_position["entry_price"] * current_position["quantity"]
                + final_price * current_position["quantity"]
            )
            pnl = (
                final_price - current_position["entry_price"]
            ) * current_position["quantity"] - commission
            pnl_pct = pnl / (
                current_position["entry_price"] * current_position["quantity"]
            )
            duration = (final_date - current_position["entry_date"]).days

            trade = Trade(
                entry_date=current_position["entry_date"],
                exit_date=final_date,
                symbol=symbol,
                action="buy_sell",
                entry_price=current_position["entry_price"],
                exit_price=final_price,
                quantity=current_position["quantity"],
                commission=commission,
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration_days=duration,
            )
            trades.append(trade)
            logger.debug(f"Closed final position at end: P&L ₹{pnl:.2f}")

        # Create result
        equity_curve = pd.Series(
            equity_curve_values,
            index=pd.DatetimeIndex(equity_curve_dates),
            name="portfolio_value",
        )

        daily_returns = equity_curve.pct_change().dropna()

        positions_df = pd.DataFrame(positions_history)
        if not positions_df.empty:
            positions_df.set_index("date", inplace=True)

        # Get agent info
        agent_info = {
            "type": agent.__class__.__name__,
            "parameters": agent.get_parameters() if hasattr(agent, "get_parameters") else {},
        }

        result = BacktestResult(
            config=self.config,
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            positions=positions_df,
            metrics={},  # Will be filled by MetricsCalculator
            agent_info=agent_info,
        )

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"final value: ₹{equity_curve.iloc[-1]:,.0f}"
        )

        return result
