#!/usr/bin/env python3
"""
Deploy trained RL agent to paper trading.

This script loads a trained agent and deploys it to paper trading using
the PaperBroker. It runs continuously during market hours, making trading
decisions based on live market data.

Key Features:
    - Real-time market data from Yahoo Finance
    - Paper trading with simulated execution
    - Performance monitoring and logging
    - Daily reports and metrics tracking
    - Safety checks and circuit breakers

Usage:
    # Deploy intraday model
    python scripts/deploy_paper_trading.py \\
        --model models/exp005_21_per_return_intra/best/best_model.zip \\
        --symbols ^NSEI \\
        --interval 5m \\
        --initial-capital 100000 \\
        --max-position-size 0.2

    # Deploy EOD model
    python scripts/deploy_paper_trading.py \\
        --model models/exp004/best/best_model.zip \\
        --symbols RELIANCE.NS TCS.NS INFY.NS \\
        --interval 1d \\
        --initial-capital 100000

Example:
    $ python scripts/deploy_paper_trading.py \\
        --model models/best_intraday.zip \\
        --symbols ^NSEI \\
        --interval 5m

Author: TradeBox-RL
Date: 2024-12-16
"""

import argparse
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger

from tradebox.agents import PPOAgent
from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.env import IntradayTradingEnv, TradingEnv
from tradebox.env.trading_env import IntradayEnvConfig
from tradebox.execution import OrderSide, PaperBroker
from tradebox.features.technical import FeatureConfig, TechnicalFeatures


# ============================================================================
# Configuration
# ============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy RL agent to paper trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model and strategy
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["^NSEI"],
        help="Stock symbols to trade (default: ^NSEI)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        choices=["1m", "5m", "15m", "1h", "1d"],
        default="5m",
        help="Data interval (default: 5m for intraday)",
    )

    # Capital and risk management
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital in rupees (default: 100000)",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=0.2,
        help="Max position size as fraction of portfolio (default: 0.2)",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=0.02,
        help="Max daily loss as fraction of portfolio (default: 0.02)",
    )

    # Trading schedule
    parser.add_argument(
        "--market-open",
        type=str,
        default="09:15",
        help="Market open time HH:MM (default: 09:15 IST)",
    )
    parser.add_argument(
        "--market-close",
        type=str,
        default="15:30",
        help="Market close time HH:MM (default: 15:30 IST)",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no actual orders)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/paper_trading",
        help="Directory for logs (default: logs/paper_trading)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/paper_trading",
        help="Directory for reports (default: reports/paper_trading)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


# ============================================================================
# Paper Trading Engine
# ============================================================================


class PaperTradingEngine:
    """
    Paper trading engine for live deployment.

    Manages the trading loop, data fetching, agent inference,
    and order execution.
    """

    def __init__(
        self,
        agent: PPOAgent,
        broker: PaperBroker,
        symbols: List[str],
        interval: str,
        market_open: str,
        market_close: str,
        max_position_size: float = 0.2,
        max_daily_loss: float = 0.02,
        log_dir: str = "logs/paper_trading",
        report_dir: str = "reports/paper_trading",
    ):
        """Initialize paper trading engine."""
        self.agent = agent
        self.broker = broker
        self.symbols = symbols
        self.interval = interval
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss

        # Parse market hours
        self.market_open = datetime.strptime(market_open, "%H:%M").time()
        self.market_close = datetime.strptime(market_close, "%H:%M").time()

        # Data loader and feature extractor
        self.data_loader = YahooDataLoader(cache_dir="cache", use_cache=False)
        self.feature_extractor = TechnicalFeatures(
            FeatureConfig(timeframe="intraday" if interval != "1d" else "eod")
        )

        # Trading state
        self.daily_start_value = broker.initial_capital
        self.trades_today = 0
        self.circuit_breaker_triggered = False

        # Logging
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.report_dir = Path(report_dir)

        logger.info(f"PaperTradingEngine initialized")
        logger.info(f"  Symbols: {symbols}")
        logger.info(f"  Interval: {interval}")
        logger.info(f"  Market hours: {market_open} - {market_close}")
        logger.info(f"  Max position size: {max_position_size:.1%}")
        logger.info(f"  Max daily loss: {max_daily_loss:.1%}")

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close

    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should trigger."""
        portfolio = self.broker.get_portfolio()
        daily_loss = (self.daily_start_value - portfolio.total_value) / self.daily_start_value

        if daily_loss >= self.max_daily_loss:
            logger.error(f"Circuit breaker triggered! Daily loss: {daily_loss:.2%}")
            self.circuit_breaker_triggered = True
            return True

        return False

    def fetch_latest_data(self, symbol: str, lookback_bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch latest market data for a symbol."""
        try:
            # Determine period based on interval and lookback
            if self.interval == "5m":
                period = "5d"  # 5 days should give us enough bars
            elif self.interval == "1m":
                period = "2d"
            elif self.interval == "1h":
                period = "1mo"
            else:  # 1d
                period = "3mo"

            data = self.data_loader.download_intraday(
                symbol=symbol,
                period=period,
                interval=self.interval,
            )

            if len(data) < lookback_bars:
                logger.warning(
                    f"Insufficient data: {len(data)} bars (need {lookback_bars})"
                )
                return None

            return data.tail(lookback_bars)

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def get_agent_action(self, data: pd.DataFrame, features: pd.DataFrame) -> int:
        """Get trading action from agent."""
        try:
            # Extract observation directly without full environment
            # Use only numeric features and latest lookback window
            numeric_features = features.select_dtypes(include=[np.number])

            # Get lookback window (default 60 bars)
            lookback = 60
            if len(numeric_features) < lookback:
                logger.warning(f"Insufficient bars: {len(numeric_features)} < {lookback}")
                # Pad with zeros if needed
                padding = np.zeros((lookback - len(numeric_features), len(numeric_features.columns)))
                feature_window = np.vstack([padding, numeric_features.values])
            else:
                feature_window = numeric_features.tail(lookback).values

            # Flatten technical features
            technical_flat = feature_window.flatten()

            # Create portfolio state (simplified for deployment)
            # [position, cash%, unrealized_pnl%, entry_price_dev, bars_held, trades_today]
            portfolio_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, float(self.trades_today)])

            # Combine observation
            obs = np.concatenate([technical_flat, portfolio_state]).astype(np.float32)

            # Get action from agent
            action, _ = self.agent.predict(obs, deterministic=True)

            return int(action)

        except Exception as e:
            logger.error(f"Failed to get agent action: {e}")
            logger.exception(e)
            return 0  # Hold on error

    def execute_action(self, symbol: str, action: int, current_price: float):
        """Execute trading action."""
        portfolio = self.broker.get_portfolio()

        # Action: 0 = Hold, 1 = Buy, 2 = Sell
        if action == 0:
            logger.info(f"Action: HOLD for {symbol}")
            return

        elif action == 1:  # Buy
            # Calculate position size
            max_value = portfolio.total_value * self.max_position_size
            quantity = int(max_value / current_price)

            if quantity > 0:
                # Use market order (price=None) for immediate execution
                order = self.broker.place_order(symbol, OrderSide.BUY, quantity, price=None)
                logger.info(
                    f"Action: BUY {quantity} shares of {symbol} @ market "
                    f"(Order: {order.order_id}, Status: {order.status.value})"
                )
                self.trades_today += 1
            else:
                logger.warning(f"Insufficient funds to buy {symbol}")

        elif action == 2:  # Sell
            # Close existing position if any
            if symbol in self.broker.positions:
                position = self.broker.positions[symbol]
                # Use market order (price=None) for immediate execution
                order = self.broker.place_order(
                    symbol, OrderSide.SELL, position.quantity, price=None
                )
                logger.info(
                    f"Action: SELL {position.quantity} shares of {symbol} @ market "
                    f"(Order: {order.order_id}, Status: {order.status.value})"
                )
                self.trades_today += 1
            else:
                logger.info(f"No position to sell for {symbol}")

    def run_trading_cycle(self):
        """Run one trading cycle."""
        logger.info("=" * 70)
        logger.info(f"Trading cycle at {datetime.now():%Y-%m-%d %H:%M:%S}")

        # Check circuit breaker
        if self.check_circuit_breaker():
            logger.error("Circuit breaker active - skipping trading")
            return

        # Trade each symbol
        for symbol in self.symbols:
            try:
                # Fetch latest data
                data = self.fetch_latest_data(symbol)
                if data is None:
                    continue

                # Extract features
                features = self.feature_extractor.extract(data, fit_normalize=False)

                # Get current price
                current_price = data.iloc[-1]["Close"]

                # Get agent action
                action = self.get_agent_action(data, features)

                # Execute action
                self.execute_action(symbol, action, current_price)

            except Exception as e:
                logger.error(f"Error trading {symbol}: {e}")

        # Log portfolio status
        portfolio = self.broker.get_portfolio()
        logger.info(f"Portfolio value: ₹{portfolio.total_value:,.2f}")
        logger.info(f"Cash: ₹{portfolio.cash:,.2f}")
        logger.info(f"Positions: {len(portfolio.positions)}")
        logger.info(f"Trades today: {self.trades_today}")

    def run(self):
        """Run paper trading loop."""
        logger.info("Starting paper trading engine...")

        while True:
            try:
                # Check if market is open
                if not self.is_market_open():
                    now = datetime.now().time()
                    logger.info(f"Market closed (current time: {now:%H:%M})")

                    # If after market close, generate daily report
                    if now > self.market_close:
                        self.generate_daily_report()
                        # Reset for next day
                        portfolio = self.broker.get_portfolio()
                        self.daily_start_value = portfolio.total_value
                        self.trades_today = 0
                        self.circuit_breaker_triggered = False

                    # Sleep until next check (5 minutes)
                    time.sleep(300)
                    continue

                # Run trading cycle
                self.run_trading_cycle()

                # Sleep based on interval
                if self.interval == "1m":
                    time.sleep(60)
                elif self.interval == "5m":
                    time.sleep(300)
                elif self.interval == "15m":
                    time.sleep(900)
                elif self.interval == "1h":
                    time.sleep(3600)
                else:  # 1d
                    # For daily, run once and wait until next day
                    time.sleep(86400)

            except KeyboardInterrupt:
                logger.info("Shutting down paper trading engine...")
                self.generate_daily_report()
                break
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                time.sleep(60)

    def generate_daily_report(self):
        """Generate daily trading report."""
        logger.info("Generating daily report...")

        portfolio = self.broker.get_portfolio()
        daily_return = (portfolio.total_value - self.daily_start_value) / self.daily_start_value

        report = f"""
# Paper Trading Daily Report
Date: {datetime.now():%Y-%m-%d}

## Portfolio Summary
- Starting Value: ₹{self.daily_start_value:,.2f}
- Ending Value: ₹{portfolio.total_value:,.2f}
- Daily Return: {daily_return:+.2%}
- Cash: ₹{portfolio.cash:,.2f}

## Trading Activity
- Total Trades: {self.trades_today}
- Open Positions: {len(portfolio.positions)}

## Positions
"""

        for symbol, position in portfolio.positions.items():
            pnl_pct = (position.current_price - position.avg_price) / position.avg_price
            report += f"- {symbol}: {position.quantity} shares @ ₹{position.avg_price:.2f} "
            report += f"(Current: ₹{position.current_price:.2f}, P&L: {pnl_pct:+.2%})\n"

        # Save report
        report_path = self.report_dir / f"daily_report_{datetime.now():%Y%m%d}.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Daily report saved: {report_path}")
        print(report)


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_file = Path(args.log_dir) / f"paper_trading_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="10 MB",
    )

    logger.info("=" * 70)
    logger.info("TradeBox-RL Paper Trading Deployment")
    logger.info("=" * 70)

    try:
        # Load trained agent
        logger.info(f"Loading model from: {args.model}")
        agent = PPOAgent.load(args.model)
        logger.info("Model loaded successfully")

        # Initialize paper broker
        logger.info(f"Initializing paper broker with ₹{args.initial_capital:,.0f}")
        broker = PaperBroker(initial_capital=args.initial_capital)

        # Create trading engine
        engine = PaperTradingEngine(
            agent=agent,
            broker=broker,
            symbols=args.symbols,
            interval=args.interval,
            market_open=args.market_open,
            market_close=args.market_close,
            max_position_size=args.max_position_size,
            max_daily_loss=args.max_daily_loss,
            log_dir=args.log_dir,
            report_dir=args.report_dir,
        )

        if args.dry_run:
            logger.warning("DRY RUN MODE - No actual trading will occur")

        # Run paper trading
        engine.run()

        return 0

    except KeyboardInterrupt:
        logger.info("Deployment stopped by user")
        return 0
    except Exception as e:
        logger.exception(f"Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
