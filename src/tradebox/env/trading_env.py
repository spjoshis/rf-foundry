"""Gymnasium-compatible trading environment for RL agents."""

import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger

from tradebox.env.action_mask import ActionMaskConfig, RegimeActionMask
from tradebox.env.costs import CostConfig, TransactionCostModel
from tradebox.env.rewards import RewardConfig, create_reward_function



@dataclass
class EnvConfig:
    """
    Configuration for trading environment.

    Attributes:
        initial_capital: Starting portfolio value (default: ₹100,000)
        lookback_window: Number of historical days in observation (default: 60)
        max_episode_steps: Maximum steps per episode (default: 500)
        cost_config: Transaction cost configuration
        reward_config: Reward function configuration
        action_mask_config: Action masking configuration (default: disabled)
    """

    initial_capital: float = 100000.0
    lookback_window: int = 60
    max_episode_steps: int = 500
    cost_config: Optional[CostConfig] = None
    reward_config: Optional[RewardConfig] = None
    action_mask_config: Optional[ActionMaskConfig] = None

    def __post_init__(self):
        """Initialize default configs if not provided."""
        # To reduce context window - Gopal
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )
        if self.cost_config is None:
            self.cost_config = CostConfig()
        if self.reward_config is None:
            self.reward_config = RewardConfig()
        if self.action_mask_config is None:
            self.action_mask_config = ActionMaskConfig()


@dataclass
class IntradayEnvConfig(EnvConfig):
    """
    Configuration for intraday trading environment.

    Extends EnvConfig with intraday-specific settings for 5-minute bar trading.

    Attributes:
        bar_interval_minutes: Interval between bars in minutes (default: 5)
        bars_per_session: Number of bars per trading session (default: 75)
                         Indian market: 9:15 AM - 3:30 PM = 6h 15min = 75 bars
        sessions_per_episode: Number of trading sessions per episode (default: 10)
        force_close_eod: Whether to force close positions at end of day (default: True)
        market_open_time: Market opening time in HH:MM format (default: "09:15" IST)
        market_close_time: Market closing time in HH:MM format (default: "15:30" IST)
        overnight_gap_handling: How to handle overnight gaps - "reset_observation" or "carry"
                               (default: "reset_observation")

    Example:
        >>> config = IntradayEnvConfig(
        ...     initial_capital=100000.0,
        ...     lookback_window=60,  # 60 bars = 5 hours
        ...     max_episode_steps=750,  # 10 sessions × 75 bars
        ...     bars_per_session=75,
        ...     sessions_per_episode=10,
        ...     force_close_eod=True
        ... )
    """

    # Intraday-specific settings
    bar_interval_minutes: int = 5
    bars_per_session: int = 75  # 6h 15min market hours
    sessions_per_episode: int = 10  # 10 trading days per episode
    force_close_eod: bool = True  # Flatten positions at 3:30 PM

    # Override EOD defaults for intraday
    lookback_window: int = 60  # 60 bars = 5 hours (not 60 days)
    max_episode_steps: int = 750  # 10 sessions × 75 bars

    # Session timing (Indian market)
    market_open_time: str = "09:15"  # IST
    market_close_time: str = "15:30"  # IST

    # Observation handling at session boundaries
    overnight_gap_handling: str = "reset_observation"  # or "carry"

    def __post_init__(self):
        """Initialize and validate intraday-specific configuration."""
        super().__post_init__()

        # Validate max_episode_steps matches sessions × bars_per_session
        expected_steps = self.sessions_per_episode * self.bars_per_session
        if self.max_episode_steps != expected_steps:
            logger.warning(
                f"max_episode_steps ({self.max_episode_steps}) doesn't match "
                f"sessions_per_episode × bars_per_session ({expected_steps}). "
                f"Using max_episode_steps={expected_steps}"
            )
            self.max_episode_steps = expected_steps

        # Validate overnight_gap_handling
        valid_gap_handling = ["reset_observation", "carry"]
        if self.overnight_gap_handling not in valid_gap_handling:
            raise ValueError(
                f"overnight_gap_handling must be one of {valid_gap_handling}, "
                f"got: {self.overnight_gap_handling}"
            )


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for EOD swing trading.

    Supports both technical-only and techno-fundamental feature inputs.

    Action Space:
        Discrete(3): {0: Hold, 1: Buy, 2: Sell}

    Observation Space:
        Box: Flattened vector of:
        - Technical features (60-day window, windowed)
        - Fundamental features (current quarter values, static - NOT windowed)
        - Portfolio state (4 values)

        Example sizes:
        - Technical only: 60 × 27 + 4 = 1,624 floats
        - Techno-fundamental: 60 × 27 + 16 + 4 = 1,640 floats

    Episode:
        - Random start position in data (leaving room for lookback)
        - Ends when reaching end of data or max_episode_steps
        - Portfolio starts with initial_capital in cash, no position

    Example:
        >>> from tradebox.data.loaders.yahoo_loader import YahooDataLoader
        >>> from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig
        >>> loader = YahooDataLoader()
        >>> data = loader.download('RELIANCE.NS', '2020-01-01', '2024-12-31')
        >>> config = FeatureExtractorConfig()
        >>> extractor = FeatureExtractor(config)
        >>> features = extractor.extract("RELIANCE", data, fit_normalize=True)
        >>> env = TradingEnv(data, features, EnvConfig())
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(1)  # Buy
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        config: EnvConfig,
    ) -> None:
        """
        Initialize trading environment.

        Args:
            data: OHLCV DataFrame with Date index
            features: Features DataFrame (technical + fundamental, same index as data)
            config: Environment configuration

        Raises:
            ValueError: If data/features are invalid or misaligned

        Note:
            The features DataFrame can contain:
            - Technical features only (backward compatible)
            - Technical + fundamental features (new techno-fundamental mode)

            Fundamental features are automatically detected and treated as STATIC
            in the observation space (current values only, not windowed like technicals).
        """
        super().__init__()

        # To reduce context window - Gopal
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )

        # Validation
        if len(data) != len(features):
            raise ValueError(
                f"Data ({len(data)}) and features ({len(features)}) "
                "must have same length"
            )

        if len(data) < config.lookback_window + config.max_episode_steps:
            raise ValueError(
                f"Data length ({len(data)}) must be at least "
                f"lookback_window + max_episode_steps "
                f"({config.lookback_window + config.max_episode_steps})"
            )

        self.data = data
        self.features = features
        self.config = config

        # Initialize transaction cost model, reward function, and action masker
        self.cost_model = TransactionCostModel(config.cost_config)
        self.reward_fn = create_reward_function(config.reward_config)
        self.action_masker = RegimeActionMask(config.action_mask_config)

        # Define action space: {Hold, Buy, Sell}
        self.action_space = spaces.Discrete(3)

        # Identify technical vs fundamental features
        numeric_features = features.select_dtypes(include=[np.number])
        tech_cols, fund_cols = self._split_feature_columns(numeric_features.columns)

        # Store for later use in _get_observation()
        self._technical_cols = tech_cols
        self._fundamental_cols = fund_cols

        # Define observation space as Dict for CNN compatibility
        # CNN expects: price (OHLCV window), indicators, fundamentals, portfolio state
        n_technical = len(tech_cols)
        n_fundamental = len(fund_cols)
        n_portfolio = 4

        # Build observation space - only include fundamentals if present
        obs_space_dict = {
            "price": spaces.Box(
                low=0,
                high=np.inf,
                shape=(config.lookback_window, 5),  # (60, 5) for OHLCV
                dtype=np.float32,
            ),
            "indicators": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_technical,),
                dtype=np.float32,
            ),
            "portfolio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_portfolio,),
                dtype=np.float32,
            ),
        }

        # Only add fundamentals if we have any (avoid empty arrays)
        if n_fundamental > 0:
            obs_space_dict["fundamentals"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_fundamental,),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(obs_space_dict)

        # Episode state (initialized in reset())
        self.current_step = 0
        self.episode_start_step = 0
        self.cash = config.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.total_trades = 0

        logger.info(
            f"TradingEnv initialized: {len(data)} timesteps, "
            f"obs_space=Dict(price:{config.lookback_window}x5, indicators:{n_technical}, "
            f"fundamentals:{n_fundamental}, portfolio:{n_portfolio})"
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment to start of new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        # Random start position (leave room for lookback and episode)
        max_start = len(self.data) - self.config.max_episode_steps - 1
        min_start = self.config.lookback_window
        if max_start <= min_start:
            self.current_step = min_start
        else:
            self.current_step = self.np_random.integers(min_start, max_start)

        self.episode_start_step = self.current_step

        # Reset portfolio to initial state
        self.cash = self.config.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.total_trades = 0

        # Reset reward function state
        self.reward_fn.reset()

        observation = self._get_observation()
        info = self._get_info()

        logger.debug(
            f"Episode reset: start_step={self.current_step}, "
            f"initial_capital={self.cash:.2f}"
        )

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the episode.

        Args:
            action: Action to take (0=Hold, 1=Buy, 2=Sell)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            - observation: Current observation (flattened features + portfolio state)
            - reward: Reward for this step
            - terminated: Whether episode ended naturally (reached end of data)
            - truncated: Whether episode was truncated (max steps reached)
            - info: Additional information dict
        """
        # Store previous value for reward calculation
        prev_value = self._get_portfolio_value()

        # Execute action
        self._execute_action(action)

        # Move to next time step
        self.current_step += 1

        # Calculate reward
        current_value = self._get_portfolio_value()
        reward = self.reward_fn.calculate(
            prev_value=prev_value,
            current_value=current_value,
            action=action,
            step=self.current_step - self.episode_start_step,
            position=self.position
        )

        # Check termination conditions (convert to Python bool for SB3 compatibility)
        terminated = bool(self.current_step >= len(self.data) - 1)
        truncated = bool(
            self.current_step - self.episode_start_step >= self.config.max_episode_steps
        )

        observation = self._get_observation()
        info = self._get_info()

        if terminated or truncated:
            logger.debug(
                f"Episode ended: steps={self.current_step - self.episode_start_step}, "
                f"final_value={current_value:.2f}, "
                f"return={(current_value / self.config.initial_capital - 1) * 100:.2f}%, "
                f"trades={self.total_trades}"
            )

        return observation, float(reward), terminated, truncated, info

    def _execute_action(self, action: int) -> None:
        """
        Execute trading action (Buy/Sell/Hold).

        Args:
            action: Action to execute (0=Hold, 1=Buy, 2=Sell)
        """
        current_price = float(self.data.iloc[self.current_step]["Close"])

        if action == 1:  # Buy
            if self.position == 0 and self.cash > 0:
                # Calculate maximum shares we can buy
                # Use cost model to get exact cost including all fees
                shares_to_buy = int(self.cash / current_price * 0.99)  # Conservative estimate

                if shares_to_buy > 0:
                    # Calculate actual cost with all fees
                    total_cost, breakdown = self.cost_model.calculate_buy_cost(
                        shares_to_buy, current_price
                    )

                    # If we can afford it, execute trade
                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        self.position = shares_to_buy
                        self.entry_price = current_price
                        self.total_trades += 1

                        logger.debug(
                            f"BUY: {shares_to_buy} shares @ ₹{current_price:.2f}, "
                            f"cost=₹{total_cost:.2f} ({breakdown['effective_rate']*100:.3f}%)"
                        )

        elif action == 2:  # Sell
            if self.position > 0:
                # Calculate proceeds after all fees
                net_proceeds, breakdown = self.cost_model.calculate_sell_proceeds(
                    self.position, current_price
                )

                self.cash += net_proceeds
                realized_pnl = net_proceeds - (self.position * self.entry_price)

                logger.debug(
                    f"SELL: {self.position} shares @ ₹{current_price:.2f}, "
                    f"proceeds=₹{net_proceeds:.2f}, "
                    f"PnL=₹{realized_pnl:.2f} ({breakdown['effective_rate']*100:.3f}%)"
                )

                self.position = 0
                self.entry_price = 0.0
                self.total_trades += 1

        # action == 0 (Hold) does nothing

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Build observation dict from features and portfolio state.

        Returns:
            Dict observation with keys:
                - "price": (lookback_window, 5) OHLCV data
                - "indicators": (n_technical,) technical indicators (current values)
                - "fundamentals": (n_fundamental,) fundamental features (current values)
                - "portfolio": (4,) portfolio state
        """
        # Extract lookback window
        start_idx = self.current_step - self.config.lookback_window
        end_idx = self.current_step

        # 1. Price data (OHLCV window)
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        price_window = self.data[ohlcv_cols].iloc[start_idx:end_idx].values.astype(np.float32)

        numeric_features = self.features.select_dtypes(include=[np.number])

        # 2. Technical indicators (current day values)
        if len(self._technical_cols) > 0:
            indicators = (
                numeric_features[self._technical_cols]
                .iloc[self.current_step]
                .values.astype(np.float32)
            )
        else:
            indicators = np.array([], dtype=np.float32)

        # 3. Fundamental features (current quarter's values)
        if len(self._fundamental_cols) > 0:
            fundamentals = (
                numeric_features[self._fundamental_cols]
                .iloc[self.current_step]
                .values.astype(np.float32)
            )
        else:
            fundamentals = np.array([], dtype=np.float32)

        # 4. Portfolio state features
        current_price = float(self.data.iloc[self.current_step]["Close"])
        portfolio_state = np.array(
            [
                self.position / 1000.0,  # Normalized position size
                (current_price - self.entry_price) / current_price
                if self.entry_price > 0
                else 0.0,  # Price change from entry
                self._get_unrealized_pnl_pct(),  # Unrealized P&L %
                self.cash / self.config.initial_capital,  # Cash %
            ],
            dtype=np.float32,
        )

        # Return Dict observation (only include fundamentals if present)
        obs = {
            "price": price_window,
            "indicators": indicators,
            "portfolio": portfolio_state,
        }

        # Only add fundamentals if we have any
        if len(self._fundamental_cols) > 0:
            obs["fundamentals"] = fundamentals

        return obs

    def _get_portfolio_value(self) -> float:
        """
        Calculate current total portfolio value.

        Returns:
            Total value = cash + position_value
        """
        current_price = float(self.data.iloc[self.current_step]["Close"])
        position_value = self.position * current_price
        return self.cash + position_value

    def _get_unrealized_pnl_pct(self) -> float:
        """
        Calculate unrealized P&L percentage for current position.

        Returns:
            P&L % if position > 0, else 0.0
        """
        if self.position == 0:
            return 0.0
        current_price = float(self.data.iloc[self.current_step]["Close"])
        return (current_price - self.entry_price) / self.entry_price

    def _get_regime_info(self) -> Tuple[int, int]:
        """
        Extract regime state and trend bias from current features.

        Returns:
            Tuple of (regime_state, trend_bias)
            - regime_state: 0 (ranging), 1 (transition), 2 (trending)
            - trend_bias: -1 (down), 0 (neutral), 1 (up)

        Raises:
            ValueError: If regime columns not found in features
        """
        regime_col = self.config.action_mask_config.regime_column
        bias_col = self.config.action_mask_config.trend_bias_column

        if regime_col not in self.features.columns:
            raise ValueError(
                f"Regime column '{regime_col}' not found in features. "
                f"Available columns: {list(self.features.columns)}"
            )

        if bias_col not in self.features.columns:
            raise ValueError(
                f"Trend bias column '{bias_col}' not found in features. "
                f"Available columns: {list(self.features.columns)}"
            )

        regime_state = int(self.features[regime_col].iloc[self.current_step])
        trend_bias = int(self.features[bias_col].iloc[self.current_step])

        return regime_state, trend_bias

    def _get_info(self) -> Dict[str, Any]:
        """
        Build info dictionary with episode metadata.

        Returns:
            Dictionary with step, portfolio value, cash, position, trades,
            and action_mask (if masking enabled)
        """
        info = {
            "step": int(self.current_step),
            "portfolio_value": float(self._get_portfolio_value()),
            "cash": float(self.cash),
            "position": int(self.position),
            "total_trades": int(self.total_trades),
            "entry_price": float(self.entry_price),
            "unrealized_pnl_pct": float(self._get_unrealized_pnl_pct()),
        }

        # Add action mask if masking is enabled
        if self.config.action_mask_config.enabled:
            regime_state, trend_bias = self._get_regime_info()
            action_mask = self.action_masker.get_mask(regime_state, trend_bias)
            info["action_mask"] = action_mask

        return info

    def _split_feature_columns(
        self, columns: pd.Index
    ) -> Tuple[list, list]:
        """
        Split feature columns into technical and fundamental features.

        Technical features are identified by common prefixes/patterns:
        - SMA_, EMA_, RSI, MACD, ATR, BB_, Stoch_, ROC, Volume_, OBV, Returns_Std

        Fundamental features are identified by common patterns:
        - PE_, PB_, PS_, ROE, ROA, Profit_Margin, Operating_Margin, Gross_Margin
        - Debt_to_Equity, Current_Ratio, Interest_Coverage
        - Revenue_Growth, EPS_Growth, Book_Value_Growth, Asset_Growth

        Args:
            columns: Index of column names from features DataFrame

        Returns:
            Tuple of (technical_cols, fundamental_cols) as lists
        """
        # Technical feature patterns
        technical_patterns = [
            "SMA_", "EMA_", "RSI", "MACD", "ATR", "BB_",
            "Stoch_", "ROC", "Volume_", "OBV", "Returns_Std",
            "Close_SMA", "Close_EMA",  # Ratio features
            # Directional and regime features
            "ADX", "Plus_DI", "Minus_DI",
            "regime_state", "regime_strength", "trend_bias", "regime_persistence"
        ]

        # Fundamental feature patterns
        fundamental_patterns = [
            "PE_", "PB_", "PS_", "ROE", "ROA",
            "Profit_Margin", "Operating_Margin", "Gross_Margin",
            "Debt_to_Equity", "Current_Ratio", "Interest_Coverage",
            "Revenue_Growth", "EPS_Growth", "Book_Value_Growth", "Asset_Growth"
        ]

        technical_cols = []
        fundamental_cols = []

        for col in columns:
            # Check if it's a technical feature
            is_technical = any(pattern in col for pattern in technical_patterns)

            # Check if it's a fundamental feature
            is_fundamental = any(pattern in col for pattern in fundamental_patterns)

            if is_technical:
                technical_cols.append(col)
            elif is_fundamental:
                fundamental_cols.append(col)
            # Else: ignore (e.g., OHLCV columns which aren't features)

        return technical_cols, fundamental_cols
