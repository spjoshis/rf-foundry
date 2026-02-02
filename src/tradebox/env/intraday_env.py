"""Gymnasium-compatible intraday trading environment for 5-minute bar trading."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger

from tradebox.env.costs import TransactionCostModel
from tradebox.env.rewards import create_reward_function
from tradebox.env.trading_env import IntradayEnvConfig


class IntradayTradingEnv(gym.Env):
    """
    Gymnasium-compatible intraday trading environment for 5-minute bars.

    CNN-Compatible Implementation: Uses Dict observation space for raw price patterns.

    Key Differences from EOD:
    - Timesteps are 5-minute bars (75 per trading session)
    - Episodes span multiple sessions (default: 10 sessions = 750 bars)
    - Positions forced to close at 3:30 PM (end of session)
    - Session boundaries handled with observation reset
    - Indicators use bar-based periods (not days)

    Action Space:
        Discrete(3): {0: Hold, 1: Buy, 2: Sell}

    Observation Space:
        Dict with keys:
        - "price": Box(0, inf, shape=(60, 5)) - OHLCV window for CNN
        - "indicators": Box(-inf, inf, shape=(n_indicators,)) - Technical indicators
        - "portfolio": Box(-inf, inf, shape=(6,)) - Portfolio state
            [position%, cash%, unrealized_pnl%, entry_price_dev, bars_held, trades_today]

        Example sizes:
        - price: 60 bars × 5 (OHLCV) = 300 floats
        - indicators: 27 floats (RSI, MACD, SMA, etc.)
        - portfolio: 6 floats

    Episode:
        - Random start position at a session beginning
        - Ends when reaching max_episode_steps or all sessions completed
        - Portfolio starts with initial_capital in cash, no position
        - Positions forced to close at end of each session (3:30 PM)

    Example:
        >>> from tradebox.data.loaders.yahoo_loader import YahooDataLoader
        >>> from tradebox.features.extractor import FeatureExtractor
        >>> from pathlib import Path
        >>>
        >>> loader = YahooDataLoader(Path("data/intraday"))
        >>> data = loader.download_intraday("RELIANCE.NS", period="60d", interval="5m")
        >>> extractor = FeatureExtractor(FeatureExtractorConfig())
        >>> features = extractor.extract("RELIANCE", data, fit_normalize=True)
        >>> config = IntradayEnvConfig()
        >>> env = IntradayTradingEnv(data, features, config)
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(1)  # Buy
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        config: IntradayEnvConfig,
    ) -> None:
        """
        Initialize intraday trading environment.

        Args:
            data: Intraday OHLCV DataFrame with Date column (datetime)
            features: Features DataFrame (technical only for Phase 1, same index as data)
            config: Intraday environment configuration

        Raises:
            ValueError: If data/features are invalid or misaligned
        """
        super().__init__()

        self.config = config
        self.data = data.copy()
        self.features = features.copy()

        # Validate data
        self._validate_data()

        # Filter features to only numeric columns (exclude Date, etc.)
        self.numeric_features = self.features.select_dtypes(include=[np.number])

        # Initialize cost and reward models
        self.cost_model = TransactionCostModel(self.config.cost_config)
        self.reward_function = create_reward_function(self.config.reward_config)

        # Define action space: {0: Hold, 1: Buy, 2: Sell}
        self.action_space = spaces.Discrete(3)

        # Define observation space as Dict for CNN compatibility
        # CNN expects: price (OHLCV window), indicators, portfolio state
        n_indicators = len(self.numeric_features.columns)
        n_portfolio = 6  # [position, cash%, unrealized_pnl%, entry_price_dev, bars_held, trades_today]

        self.observation_space = spaces.Dict(
            {
                "price": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.config.lookback_window, 5),  # (60, 5) for OHLCV
                    dtype=np.float32,
                ),
                "indicators": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_indicators,),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(n_portfolio,),
                    dtype=np.float32,
                ),
            }
        )

        logger.info(
            f"IntradayTradingEnv initialized: "
            f"data={len(self.data)} bars, "
            f"obs_space=Dict(price:{self.config.lookback_window}x5, indicators:{n_indicators}, portfolio:{n_portfolio}), "
            f"sessions={self.config.sessions_per_episode}, "
            f"bars_per_session={self.config.bars_per_session}"
        )

        # State variables (initialized in reset)
        self.current_step: int = 0
        self.current_session: int = 0
        self.bars_in_session: int = 0
        self.session_start_bar: int = 0
        self.trades_today: int = 0

        # Portfolio state
        self.cash: float = 0.0
        self.position: int = 0  # Number of shares
        self.entry_price: float = 0.0
        self.entry_bar: int = 0
        self.total_trades: int = 0
        self.portfolio_value_history: list = []

        # For reward tracking
        self.prev_portfolio_value: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Starts at a random session beginning, ensuring enough data for lookback window
        and full episodes.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation
            info: Initial info dict
        """
        super().reset(seed=seed)

        # Reset portfolio
        self.cash = self.config.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.total_trades = 0
        self.portfolio_value_history = []
        self.prev_portfolio_value = self.config.initial_capital

        # Reset session tracking
        self.current_session = 0
        self.bars_in_session = 0
        self.trades_today = 0

        # Choose random starting session (but leave room for lookback + full episode)
        min_start_bar = self.config.lookback_window
        max_start_session = (
            len(self.data) - self.config.max_episode_steps - self.config.lookback_window
        ) // self.config.bars_per_session

        if max_start_session < 1:
            raise ValueError(
                f"Not enough data: need at least "
                f"{self.config.lookback_window + self.config.max_episode_steps} bars, "
                f"have {len(self.data)}"
            )

        # Random session start
        start_session = self.np_random.integers(0, max_start_session)
        self.session_start_bar = (
            min_start_bar + start_session * self.config.bars_per_session
        )
        self.current_step = self.session_start_bar

        # Reset reward function
        self.reward_function.reset()

        logger.debug(
            f"Episode reset: starting at bar {self.current_step} "
            f"(session {start_session})"
        )

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep (5-minute bar) in the environment.

        Args:
            action: Action to take {0: Hold, 1: Buy, 2: Sell}

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional info dict
        """
        # Execute the action (buy/sell/hold)
        self._execute_action(action)

        # Move to next timestep
        self.current_step += 1
        self.bars_in_session += 1

        # Check for session end (force closure at 3:30 PM)
        session_ended = self._check_session_end()

        # Get observation for next state
        obs = self._get_observation()

        # Calculate reward
        current_portfolio_value = self._get_portfolio_value()
        reward = self.reward_function.calculate(
            prev_value=self.prev_portfolio_value,
            current_value=current_portfolio_value,
            action=action,
            step=self.current_step - self.session_start_bar,
        )
        self.prev_portfolio_value = current_portfolio_value

        # Track portfolio value
        self.portfolio_value_history.append(current_portfolio_value)

        # Check termination conditions
        terminated = False
        truncated = False

        # Episode ends if we've completed all sessions
        if self.current_session >= self.config.sessions_per_episode:
            terminated = True
            logger.debug(
                f"Episode terminated: completed {self.current_session} sessions"
            )

        # Truncate if we run out of data
        if self.current_step >= len(self.data) - 1:
            truncated = True
            logger.debug("Episode truncated: reached end of data")

        # Truncate if portfolio value drops too low
        if current_portfolio_value < 0.3 * self.config.initial_capital:
            truncated = True
            logger.warning(
                f"Episode truncated: portfolio value dropped to "
                f"{current_portfolio_value:.2f} (70% loss)"
            )

        info = self._get_info()
        info["session_ended"] = session_ended

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> None:
        """
        Execute trading action (buy/sell/hold).

        Args:
            action: Action to execute {0: Hold, 1: Buy, 2: Sell}
        """
        current_price = self.data.iloc[self.current_step]["Close"]
        current_volume = self.data.iloc[self.current_step].get("Volume", 1000000)

        # Get ATR for dynamic slippage calculation (if available)
        atr = self.features.iloc[self.current_step].get("ATR", current_price * 0.01)

        if action == 0:  # Hold
            pass

        elif action == 1:  # Buy
            if self.position == 0:  # Only buy if no position
                # Calculate position size (50% of capital for Phase 1)
                position_size_pct = 0.50
                order_value = self.cash * position_size_pct

                # Calculate shares to buy
                shares = int(order_value / current_price)

                if shares > 0:
                    # Calculate execution costs
                    total_cost, cost_breakdown = self.cost_model.calculate_buy_cost(
                        shares=shares,
                        price=current_price,
                        volume=current_volume,
                        atr=atr,
                    )

                    # Execute buy
                    self.position = shares
                    self.entry_price = current_price
                    self.entry_bar = self.current_step
                    self.cash -= total_cost
                    self.total_trades += 1
                    self.trades_today += 1

                    logger.debug(
                        f"BUY: {shares} shares @ {current_price:.2f}, "
                        f"cost={total_cost:.2f}, cash={self.cash:.2f}"
                    )

        elif action == 2:  # Sell
            if self.position > 0:  # Only sell if we have position
                # Calculate execution revenue
                total_revenue, cost_breakdown = self.cost_model.calculate_sell_revenue(
                    shares=self.position,
                    price=current_price,
                    volume=current_volume,
                    atr=atr,
                )

                # Execute sell
                self.cash += total_revenue
                realized_pnl = total_revenue - (self.position * self.entry_price)

                logger.debug(
                    f"SELL: {self.position} shares @ {current_price:.2f}, "
                    f"revenue={total_revenue:.2f}, pnl={realized_pnl:.2f}"
                )

                # Clear position
                self.position = 0
                self.entry_price = 0.0
                self.entry_bar = 0
                self.total_trades += 1
                self.trades_today += 1

    def _check_session_end(self) -> bool:
        """
        Check if current session has ended and handle session boundary.

        Returns:
            True if session ended, False otherwise
        """
        # Check if we've reached end of session (75 bars)
        if self.bars_in_session >= self.config.bars_per_session:
            logger.debug(
                f"Session {self.current_session} ended at bar {self.current_step}"
            )

            # Force close any open position (regulatory requirement)
            if self.position > 0 and self.config.force_close_eod:
                current_price = self.data.iloc[self.current_step - 1]["Close"]
                current_volume = self.data.iloc[self.current_step - 1].get(
                    "Volume", 1000000
                )
                atr = self.features.iloc[self.current_step - 1].get(
                    "ATR", current_price * 0.01
                )

                total_revenue, _ = self.cost_model.calculate_sell_revenue(
                    shares=self.position,
                    price=current_price,
                    volume=current_volume,
                    atr=atr,
                )

                self.cash += total_revenue
                logger.warning(
                    f"FORCED EOD CLOSURE: {self.position} shares @ {current_price:.2f}"
                )

                self.position = 0
                self.entry_price = 0.0
                self.entry_bar = 0

            # Move to next session
            self.current_session += 1
            self.bars_in_session = 0
            self.trades_today = 0
            self.session_start_bar = self.current_step

            return True

        return False

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construct observation from current state.

        Returns:
            Dict observation with keys:
                - "price": (lookback_window, 5) OHLCV data
                - "indicators": (n_indicators,) technical indicators
                - "portfolio": (6,) portfolio state
        """
        # 1. Price data (OHLCV window)
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step

        # Extract OHLCV columns from data
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        price_window = self.data[ohlcv_cols].iloc[start_idx:end_idx].values

        # Pad with zeros if we don't have full lookback window
        if len(price_window) < self.config.lookback_window:
            padding = np.zeros((self.config.lookback_window - len(price_window), 5))
            price_window = np.vstack([padding, price_window])

        # 2. Technical indicators (current bar)
        indicators = self.numeric_features.iloc[self.current_step].values

        # 3. Portfolio state (6 features)
        portfolio_state = self._get_portfolio_state()

        # Return Dict observation
        obs = {
            "price": price_window.astype(np.float32),
            "indicators": indicators.astype(np.float32),
            "portfolio": portfolio_state.astype(np.float32),
        }

        return obs

    def _get_portfolio_state(self) -> np.ndarray:
        """
        Get current portfolio state features.

        Returns:
            Array of 6 portfolio features
        """
        current_price = self.data.iloc[self.current_step]["Close"]
        portfolio_value = self._get_portfolio_value()

        # Position size as fraction of portfolio (0 to 1)
        position_pct = (
            (self.position * current_price) / portfolio_value if portfolio_value > 0 else 0.0
        )

        # Cash as fraction of portfolio
        cash_pct = self.cash / portfolio_value if portfolio_value > 0 else 1.0

        # Unrealized PnL percentage
        unrealized_pnl_pct = self._get_unrealized_pnl_pct()

        # Entry price deviation (how far current price is from entry)
        entry_price_dev = (
            (current_price - self.entry_price) / self.entry_price
            if self.entry_price > 0
            else 0.0
        )

        # Bars held (normalized by session length)
        bars_held = (
            (self.current_step - self.entry_bar) / self.config.bars_per_session
            if self.entry_bar > 0
            else 0.0
        )

        # Trades today (normalized)
        trades_today_norm = self.trades_today / 10.0  # Assume max 10 trades/day

        return np.array(
            [
                position_pct,
                cash_pct,
                unrealized_pnl_pct,
                entry_price_dev,
                bars_held,
                trades_today_norm,
            ],
            dtype=np.float32,
        )

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value (cash + position value)."""
        current_price = self.data.iloc[self.current_step]["Close"]
        position_value = self.position * current_price
        return self.cash + position_value

    def _get_unrealized_pnl_pct(self) -> float:
        """Calculate unrealized PnL as percentage of entry value."""
        if self.position == 0 or self.entry_price == 0:
            return 0.0

        current_price = self.data.iloc[self.current_step]["Close"]
        entry_value = self.position * self.entry_price
        current_value = self.position * current_price
        unrealized_pnl = current_value - entry_value

        return unrealized_pnl / entry_value

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for current state."""
        current_price = self.data.iloc[self.current_step]["Close"]

        return {
            "step": self.current_step,
            "session": self.current_session,
            "bars_in_session": self.bars_in_session,
            "portfolio_value": self._get_portfolio_value(),
            "cash": self.cash,
            "position": self.position,
            "position_value": self.position * current_price,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "unrealized_pnl_pct": self._get_unrealized_pnl_pct(),
            "total_trades": self.total_trades,
            "trades_today": self.trades_today,
            "bars_held": self.current_step - self.entry_bar if self.entry_bar > 0 else 0,
        }

    def _validate_data(self) -> None:
        """Validate input data and features."""
        # Check data has required columns
        required_cols = ["Open", "High", "Low", "Close"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Data missing required column: {col}")

        # Check data length is sufficient
        min_length = self.config.lookback_window + self.config.max_episode_steps
        if len(self.data) < min_length:
            raise ValueError(
                f"Data too short: need at least {min_length} bars, got {len(self.data)}"
            )

        # Check features and data have same length
        if len(self.features) != len(self.data):
            raise ValueError(
                f"Data and features length mismatch: "
                f"data={len(self.data)}, features={len(self.features)}"
            )

        logger.info("Data validation passed")
