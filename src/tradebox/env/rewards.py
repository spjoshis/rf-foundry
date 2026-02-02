"""Reward function implementations for RL trading environment."""
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from loguru import logger



@dataclass
class RewardConfig:
    """
    Configuration for reward function.

    Attributes:
        reward_type: Type of reward function ('simple', 'risk_adjusted', 'sharpe',
            'sortino', 'volatility_penalized', 'enhanced_drawdown', 'calmar',
            'active_trading')
        drawdown_penalty: Penalty weight for drawdown (default: 0.5)
        trade_penalty: Penalty for each trade to discourage overtrading (default: 0.001)
        sharpe_window: Window size for rolling Sharpe calculation (default: 20)
        sortino_window: Window size for rolling Sortino calculation (default: 20)
        risk_free_rate: Annual risk-free rate for Sortino calculation (default: 0.06 = 6%)
        mar: Minimum acceptable return for Sortino downside deviation (default: 0.0)
        volatility_window: Window size for rolling volatility calculation (default: 20)
        volatility_penalty: Penalty weight for volatility (default: 0.1)
        enhanced_drawdown_penalty: Penalty weight for quadratic drawdown (default: 2.0)
        duration_penalty: Penalty weight for drawdown duration (default: 0.5)
        calmar_window: Window size for rolling Calmar ratio calculation (default: 60)
        min_dd_threshold: Minimum drawdown threshold for Calmar (default: 0.01 = 1%)
        max_holding_days: Maximum days to hold a position before penalty (default: 3)
        holding_penalty: Penalty per day beyond max_holding_days (default: 0.001)
        trade_completion_bonus: Bonus for completing a round-trip trade (default: 0.002)
        inactivity_penalty: Penalty for not trading for extended periods (default: 0.0005)
        inactivity_threshold: Days of inactivity before penalty applies (default: 5)
    """

    reward_type: str = "risk_adjusted"
    drawdown_penalty: float = 0.5
    trade_penalty: float = 0.001
    sharpe_window: int = 20

    # Sortino reward parameters
    sortino_window: int = 20
    risk_free_rate: float = 0.06  # 6% annual (typical Indian FD rate)
    mar: float = 0.0  # Minimum acceptable return

    # Volatility-penalized reward parameters
    volatility_window: int = 20
    volatility_penalty: float = 0.1

    # Enhanced drawdown reward parameters
    enhanced_drawdown_penalty: float = 2.0
    duration_penalty: float = 0.5

    # Calmar reward parameters
    calmar_window: int = 60
    min_dd_threshold: float = 0.01  # 1% minimum to prevent division by zero

    # Active trading reward parameters (for frequent trading)
    max_holding_days: int = 3  # Maximum days to hold before penalty
    holding_penalty: float = 0.001  # Penalty per day beyond max_holding_days
    trade_completion_bonus: float = 0.002  # Bonus for completing buy-sell cycle
    inactivity_penalty: float = 0.0005  # Penalty for extended periods without trading
    inactivity_threshold: int = 5  # Days without trading before penalty applies


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.

    All reward functions must implement the calculate() method.
    Provides common infrastructure for tracking returns history and drawdown.

    Example:
        >>> config = RewardConfig(reward_type='simple')
        >>> reward_fn = SimpleReward(config)
        >>> reward = reward_fn.calculate(prev_value=10000, current_value=10100,
        ...                               action=1, step=5)
    """

    def __init__(self, config: RewardConfig) -> None:
        """
        Initialize reward function.

        Args:
            config: Reward configuration
        """
        self.config = config
        self.returns_history: List[float] = []
        self.peak_value: float = 0.0

        # To reduce context window - Gopal
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config}")

    @abstractmethod
    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int
    ) -> float:
        """
        Calculate reward for a single step.

        Args:
            prev_value: Portfolio value at previous step
            current_value: Portfolio value at current step
            action: Action taken (0=Hold, 1=Buy, 2=Sell)
            step: Current step number in episode

        Returns:
            Reward value (float)
        """
        pass

    def reset(self) -> None:
        """Reset reward function state for new episode."""
        self.returns_history = []
        self.peak_value = 0.0
        logger.debug(f"{self.__class__.__name__} reset")


class SimpleReward(RewardFunction):
    """
    Simple return-based reward function.

    Reward = (current_value - prev_value) / prev_value

    This is the simplest reward: just the percentage change in portfolio value.
    Does not penalize risk or overtrading.

    Example:
        >>> config = RewardConfig(reward_type='simple')
        >>> reward_fn = SimpleReward(config)
        >>> reward = reward_fn.calculate(10000, 10100, action=1, step=5)
        >>> # reward = (10100 - 10000) / 10000 = 0.01 (1% gain)
    """

    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int, position
    ) -> float:
        """Calculate simple percentage return."""
        if prev_value <= 0:
            logger.warning(f"Invalid prev_value: {prev_value}, returning 0 reward")
            return 0.0

        daily_return = (current_value - prev_value) / prev_value

        logger.debug(
            f"Step {step}: Simple reward = {daily_return:.4f} "
            f"(value: {prev_value:.2f} -> {current_value:.2f})"
        )

        return daily_return


class RiskAdjustedReward(RewardFunction):

    def calculate(self, prev_value, current_value, action, step, position):

        if prev_value <= 0:
            return 0.0

        # 1. Return
        daily_return = (current_value - prev_value) / prev_value

        if abs(daily_return) < 0.0002:
            daily_return = 0.0

        # 2. Peak & drawdown
        self.peak_value = max(self.peak_value, current_value)
        drawdown = (self.peak_value - current_value) / self.peak_value

        dd_delta = drawdown - getattr(self, "prev_drawdown", 0.0)
        self.prev_drawdown = drawdown

        # 3. Penalize drawdown ONLY when worsening AND in position
        dd_penalty = 0.0
        if step % 10 == 0 and position != 0 and dd_delta > 0:
            dd_penalty = self.config.drawdown_penalty * dd_delta

        # if position != 0 and dd_delta > 0:
        #     dd_penalty = self.config.drawdown_penalty * dd_delta

        if position == 0:
            self.entry_value = current_value

        # 4. Penalize ONLY entry (not exit)
        is_entry = (position == 0 and action == 1)
        trade_penalty = self.config.trade_penalty if is_entry else 0.0

        if action == 2 and position == 1:
            realized_pnl = (current_value - getattr(self, "entry_value", 0.0)) / getattr(self, "entry_value", 0.0)
            if realized_pnl > 0:
                reward += 0.001

        # 5. Final reward
        reward = daily_return - dd_penalty - trade_penalty

        # 6. Normalize + clip
        reward = np.tanh(reward * 10)
        logger.debug(f"Rewards is {reward}")

        return reward



class SharpeReward(RewardFunction):
    """
    Sharpe ratio-based reward function.

    Uses rolling Sharpe ratio over a window of recent returns.
    If window not filled yet, returns simple daily return.

    Sharpe = (mean_return / std_return) × sqrt(252)

    Where 252 is the number of trading days in a year (annualization factor).

    This encourages consistent returns with low volatility.

    Example:
        >>> config = RewardConfig(sharpe_window=20)
        >>> reward_fn = SharpeReward(config)
        >>> for step in range(30):
        ...     reward = reward_fn.calculate(prev_val, curr_val, action, step)
    """

    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int
    ) -> float:
        """Calculate Sharpe-based reward."""
        if prev_value <= 0:
            logger.warning(f"Invalid prev_value: {prev_value}, returning 0 reward")
            return 0.0

        # Calculate daily return
        daily_return = (current_value - prev_value) / prev_value
        self.returns_history.append(daily_return)

        # If not enough history, return simple daily return
        if len(self.returns_history) < self.config.sharpe_window:
            # Commented to reduct context window - Gopal
            # logger.debug(
            #     f"Step {step}: Sharpe reward = {daily_return:.4f} "
            #     f"(window not filled: {len(self.returns_history)}/{self.config.sharpe_window})"
            # )
            return daily_return

        # Calculate rolling Sharpe ratio
        recent_returns = self.returns_history[-self.config.sharpe_window :]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        # Avoid division by zero
        if std_return == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252)

        # Commented to reduct context window - Gopal
        # logger.debug(
        #     f"Step {step}: Sharpe reward = {sharpe_ratio:.4f} "
        #     f"(mean: {mean_return:.4f}, std: {std_return:.4f})"
        # )

        return sharpe_ratio


class SortinoReward(RewardFunction):
    """
    Sortino ratio-based reward function (penalizes only downside volatility).

    Uses rolling Sortino ratio over a window of recent returns.
    Unlike Sharpe ratio, only penalizes downside volatility (negative moves),
    allowing beneficial upside volatility. If window not filled yet, returns
    simple daily return.

    Sortino = (mean_return - risk_free_rate) / downside_std × sqrt(252)

    Where downside_std only considers returns below the MAR (Minimum Acceptable Return).
    This aligns better with investor psychology: large gains are good, large losses are bad.

    Example:
        >>> config = RewardConfig(reward_type='sortino', sortino_window=20,
        ...                       risk_free_rate=0.06, mar=0.0)
        >>> reward_fn = SortinoReward(config)
        >>> for step in range(30):
        ...     reward = reward_fn.calculate(prev_val, curr_val, action, step)
    """

    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int
    ) -> float:
        """
        Calculate Sortino-based reward.

        Args:
            prev_value: Portfolio value at previous step
            current_value: Portfolio value at current step
            action: Action taken (0=Hold, 1=Buy, 2=Sell) - not used in calculation
            step: Current step number in episode

        Returns:
            Sortino ratio if window filled, else simple daily return
        """
        if prev_value <= 0:
            logger.warning(f"Invalid prev_value: {prev_value}, returning 0 reward")
            return 0.0

        # Calculate daily return
        daily_return = (current_value - prev_value) / prev_value
        self.returns_history.append(daily_return)

        # Bootstrap period: use simple return until window filled
        if len(self.returns_history) < self.config.sortino_window:
            logger.debug(
                f"Step {step}: Sortino reward = {daily_return:.4f} "
                f"(bootstrap: {len(self.returns_history)}/{self.config.sortino_window})"
            )
            return daily_return

        # Calculate rolling Sortino ratio
        recent_returns = self.returns_history[-self.config.sortino_window :]
        mean_return = np.mean(recent_returns)

        # Downside deviation: only consider returns below MAR
        downside_returns = [min(0, r - self.config.mar) for r in recent_returns]
        downside_variance = np.mean([r**2 for r in downside_returns])
        downside_std = np.sqrt(downside_variance)

        # Avoid division by zero (all returns above MAR)
        if downside_std == 0:
            logger.debug(
                f"Step {step}: Sortino reward = 0.0 "
                f"(no downside volatility - all returns above MAR)"
            )
            return 0.0

        # Annualized Sortino ratio
        rf_daily = self.config.risk_free_rate / 252
        sortino_ratio = (mean_return - rf_daily) / downside_std * np.sqrt(252)

        logger.debug(
            f"Step {step}: Sortino reward = {sortino_ratio:.4f} "
            f"(mean: {mean_return:.4f}, downside_std: {downside_std:.4f})"
        )

        return sortino_ratio


class VolatilityPenalizedReward(RewardFunction):
    """
    Volatility-penalized reward with drawdown and trade penalties.

    Reward = daily_return - λ_vol × rolling_volatility - λ_dd × drawdown - λ_trade × is_trade

    Where:
    - daily_return = (current_value - prev_value) / prev_value
    - rolling_volatility = annualized std of returns over window
    - drawdown = (peak_value - current_value) / peak_value
    - is_trade = 1.0 if action in [Buy, Sell], else 0.0

    This encourages stable, consistent returns by directly penalizing return fluctuations.
    Combines volatility penalty with existing drawdown and trade penalties for comprehensive
    risk management.

    Example:
        >>> config = RewardConfig(reward_type='volatility_penalized',
        ...                       volatility_penalty=0.1, volatility_window=20)
        >>> reward_fn = VolatilityPenalizedReward(config)
        >>> reward = reward_fn.calculate(10000, 10100, action=1, step=5)
    """

    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int
    ) -> float:
        """
        Calculate volatility-penalized reward.

        Args:
            prev_value: Portfolio value at previous step
            current_value: Portfolio value at current step
            action: Action taken (0=Hold, 1=Buy, 2=Sell)
            step: Current step number in episode

        Returns:
            Reward with volatility, drawdown, and trade penalties applied
        """
        if prev_value <= 0:
            logger.warning(f"Invalid prev_value: {prev_value}, returning 0 reward")
            return 0.0

        # Daily return component
        daily_return = (current_value - prev_value) / prev_value
        self.returns_history.append(daily_return)

        # Volatility penalty component (bootstrap period: no penalty)
        volatility_penalty = 0.0
        if len(self.returns_history) >= self.config.volatility_window:
            recent_returns = self.returns_history[-self.config.volatility_window :]
            rolling_std = np.std(recent_returns)
            # Annualize volatility: std × sqrt(252)
            annualized_vol = rolling_std * np.sqrt(252)
            volatility_penalty = self.config.volatility_penalty * annualized_vol

        # Drawdown component
        self.peak_value = max(self.peak_value, current_value)
        drawdown = 0.0
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value

        # Trade penalty (discourage overtrading)
        is_trade = 1.0 if action in [1, 2] else 0.0  # 1=Buy, 2=Sell

        # Combine components
        reward = (
            daily_return
            - volatility_penalty
            - self.config.drawdown_penalty * drawdown
            - self.config.trade_penalty * is_trade
        )

        logger.debug(
            f"Step {step}: Volatility-penalized reward = {reward:.4f} "
            f"(return: {daily_return:.4f}, vol_penalty: {volatility_penalty:.4f}, "
            f"dd: {drawdown:.4f}, trade_penalty: {self.config.trade_penalty * is_trade:.4f})"
        )

        return reward


class EnhancedDrawdownReward(RewardFunction):
    """
    Enhanced drawdown reward with quadratic penalty and duration tracking.

    Reward = daily_return - λ_dd × DD² - λ_dur × DD_duration - λ_trade × is_trade

    Where:
    - daily_return = (current_value - prev_value) / prev_value
    - DD = current drawdown percentage
    - DD² = squared drawdown (penalizes large drawdowns exponentially)
    - DD_duration = (steps_since_peak / 252) normalized drawdown duration
    - is_trade = 1.0 if action in [Buy, Sell], else 0.0

    This creates exponentially increasing penalties for larger drawdowns and also
    penalizes prolonged underwater periods. Strongly encourages capital preservation.

    Quadratic penalty examples:
    - 5% DD → penalty = 2.0 × 0.0025 = 0.005
    - 10% DD → penalty = 2.0 × 0.01 = 0.02 (4x worse)
    - 20% DD → penalty = 2.0 × 0.04 = 0.08 (16x worse)

    Example:
        >>> config = RewardConfig(reward_type='enhanced_drawdown',
        ...                       enhanced_drawdown_penalty=2.0, duration_penalty=0.5)
        >>> reward_fn = EnhancedDrawdownReward(config)
        >>> reward = reward_fn.calculate(10000, 9500, action=2, step=10)
    """

    def __init__(self, config: RewardConfig) -> None:
        """
        Initialize enhanced drawdown reward function.

        Args:
            config: Reward configuration
        """
        super().__init__(config)
        self.steps_since_peak: int = 0  # Track drawdown duration

    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int
    ) -> float:
        """
        Calculate enhanced drawdown reward with quadratic and duration penalties.

        Args:
            prev_value: Portfolio value at previous step
            current_value: Portfolio value at current step
            action: Action taken (0=Hold, 1=Buy, 2=Sell)
            step: Current step number in episode

        Returns:
            Reward with quadratic drawdown and duration penalties applied
        """
        if prev_value <= 0:
            logger.warning(f"Invalid prev_value: {prev_value}, returning 0 reward")
            return 0.0

        # Daily return component
        daily_return = (current_value - prev_value) / prev_value

        # Update peak tracking
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.steps_since_peak = 0  # Reset duration counter at new peak
        else:
            self.steps_since_peak += 1

        # Quadratic drawdown penalty
        drawdown = 0.0
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value

        drawdown_penalty = self.config.enhanced_drawdown_penalty * (drawdown**2)

        # Duration penalty (normalized by trading year: 252 days)
        duration_penalty = self.config.duration_penalty * (self.steps_since_peak / 252)

        # Trade penalty
        is_trade = 1.0 if action in [1, 2] else 0.0
        trade_penalty = self.config.trade_penalty * is_trade

        # Combine components
        reward = daily_return - drawdown_penalty - duration_penalty - trade_penalty

        logger.debug(
            f"Step {step}: Enhanced DD reward = {reward:.4f} "
            f"(return: {daily_return:.4f}, dd²_penalty: {drawdown_penalty:.4f}, "
            f"dur_penalty: {duration_penalty:.4f} [{self.steps_since_peak} steps], "
            f"trade_penalty: {trade_penalty:.4f})"
        )

        return reward

    def reset(self) -> None:
        """Reset reward function state for new episode."""
        super().reset()
        self.steps_since_peak = 0
        logger.debug(f"{self.__class__.__name__} reset (steps_since_peak cleared)")


class CalmarReward(RewardFunction):
    """
    Calmar ratio-based reward function.

    Uses rolling Calmar ratio over a window of portfolio values.
    Calmar ratio = CAGR / max_drawdown, a professional trading metric that
    directly optimizes return per unit of worst-case risk.

    Reward = (CAGR_rolling / max_DD_rolling) - λ_trade × is_trade

    Where:
    - CAGR_rolling = annualized return over rolling window
    - max_DD_rolling = maximum drawdown over rolling window
    - λ_trade = trade penalty to discourage overtrading

    If window not filled yet, returns simple daily return. Uses minimum DD threshold
    to prevent division by very small numbers.

    Example:
        >>> config = RewardConfig(reward_type='calmar', calmar_window=60,
        ...                       min_dd_threshold=0.01)
        >>> reward_fn = CalmarReward(config)
        >>> for step in range(100):
        ...     reward = reward_fn.calculate(prev_val, curr_val, action, step)
    """

    def __init__(self, config: RewardConfig) -> None:
        """
        Initialize Calmar reward function.

        Args:
            config: Reward configuration
        """
        super().__init__(config)
        self.value_history: List[float] = []
        self.window_max_dd: float = 0.0

    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int
    ) -> float:
        """
        Calculate Calmar-based reward.

        Args:
            prev_value: Portfolio value at previous step
            current_value: Portfolio value at current step
            action: Action taken (0=Hold, 1=Buy, 2=Sell)
            step: Current step number in episode

        Returns:
            Calmar ratio if window filled, else simple daily return
        """
        if prev_value <= 0:
            logger.warning(f"Invalid prev_value: {prev_value}, returning 0 reward")
            return 0.0

        # Track portfolio values
        self.value_history.append(current_value)

        # Bootstrap period: use simple return until window filled
        if len(self.value_history) < self.config.calmar_window:
            daily_return = (current_value - prev_value) / prev_value
            logger.debug(
                f"Step {step}: Calmar reward = {daily_return:.4f} "
                f"(bootstrap: {len(self.value_history)}/{self.config.calmar_window})"
            )
            return daily_return

        # Rolling CAGR calculation
        window_values = self.value_history[-self.config.calmar_window :]
        start_val = window_values[0]
        end_val = window_values[-1]

        if start_val <= 0:
            logger.warning(f"Invalid start_val: {start_val}, returning 0 reward")
            return 0.0

        # CAGR = (end_val / start_val) ^ (252 / window) - 1
        cagr = (end_val / start_val) ** (252 / self.config.calmar_window) - 1

        # Rolling max drawdown
        self.window_max_dd = self._calculate_max_drawdown(window_values)

        # Prevent division by zero: use minimum threshold
        max_dd = max(self.window_max_dd, self.config.min_dd_threshold)

        # Calmar ratio
        calmar = cagr / max_dd

        # Trade penalty
        is_trade = 1.0 if action in [1, 2] else 0.0
        trade_penalty = self.config.trade_penalty * is_trade

        reward = calmar - trade_penalty

        logger.debug(
            f"Step {step}: Calmar reward = {reward:.4f} "
            f"(CAGR: {cagr:.4f}, max_DD: {max_dd:.4f}, calmar: {calmar:.4f})"
        )

        return reward

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """
        Calculate maximum drawdown over a list of portfolio values.

        Args:
            values: List of portfolio values

        Returns:
            Maximum drawdown as a fraction (0.0 to 1.0)
        """
        if not values:
            return 0.0

        peak = values[0]
        max_dd = 0.0

        for val in values:
            if val > peak:
                peak = val
            if peak > 0:
                dd = (peak - val) / peak
                max_dd = max(max_dd, dd)

        return max_dd

    def reset(self) -> None:
        """Reset reward function state for new episode."""
        super().reset()
        self.value_history = []
        self.window_max_dd = 0.0
        logger.debug(f"{self.__class__.__name__} reset (value_history cleared)")


class ActiveTradingReward(RewardFunction):
    """
    Active trading reward function that encourages frequent short-term trades.

    Designed to address the "buy and hold forever" problem by:
    1. Penalizing holding positions beyond max_holding_days
    2. Rewarding trade completions (buy-sell cycles)
    3. Penalizing extended periods of inactivity
    4. Using simple returns as base reward without trade penalty

    Reward = daily_return + trade_completion_bonus - holding_penalty - inactivity_penalty

    Where:
    - daily_return = (current_value - prev_value) / prev_value
    - trade_completion_bonus = bonus when a sell completes a trade cycle
    - holding_penalty = penalty × (days_held - max_holding_days) if days_held > max_holding_days
    - inactivity_penalty = penalty if no trades for inactivity_threshold days

    This reward function is specifically designed for swing trading strategies
    with holding periods of 1-3 days.

    Example:
        >>> config = RewardConfig(
        ...     reward_type='active_trading',
        ...     max_holding_days=3,
        ...     holding_penalty=0.001,
        ...     trade_completion_bonus=0.002,
        ...     inactivity_penalty=0.0005,
        ...     inactivity_threshold=5
        ... )
        >>> reward_fn = ActiveTradingReward(config)
        >>> reward = reward_fn.calculate(10000, 10100, action=1, step=5)
    """

    def __init__(self, config: RewardConfig) -> None:
        """
        Initialize active trading reward function.

        Args:
            config: Reward configuration with active trading parameters
        """
        super().__init__(config)
        self.days_in_position: int = 0  # Track how long we've held current position
        self.days_since_last_trade: int = 0  # Track inactivity
        self.in_position: bool = False  # Track if we're currently in a position
        self.last_action: int = 0  # Track previous action for trade completion detection

    def calculate(
        self, prev_value: float, current_value: float, action: int, step: int
    ) -> float:
        """
        Calculate active trading reward with holding and inactivity penalties.

        Args:
            prev_value: Portfolio value at previous step
            current_value: Portfolio value at current step
            action: Action taken (0=Hold, 1=Buy, 2=Sell)
            step: Current step number in episode

        Returns:
            Reward encouraging active trading with short holding periods
        """
        if prev_value <= 0:
            logger.warning(f"Invalid prev_value: {prev_value}, returning 0 reward")
            return 0.0

        # Base reward: simple daily return
        daily_return = (current_value - prev_value) / prev_value

        # Initialize reward components
        trade_bonus = 0.0
        holding_penalty_value = 0.0
        inactivity_penalty_value = 0.0

        # Track position and trades
        if action == 1:  # Buy
            if not self.in_position:
                self.in_position = True
                self.days_in_position = 0
                self.days_since_last_trade = 0

        elif action == 2:  # Sell
            if self.in_position:
                # Trade completion bonus for completing a buy-sell cycle
                trade_bonus = self.config.trade_completion_bonus
                self.in_position = False
                self.days_in_position = 0
                self.days_since_last_trade = 0

        else:  # Hold (action == 0)
            self.days_since_last_trade += 1

            if self.in_position:
                self.days_in_position += 1

                # Holding penalty if beyond max days
                if self.days_in_position > self.config.max_holding_days:
                    excess_days = self.days_in_position - self.config.max_holding_days
                    holding_penalty_value = self.config.holding_penalty * excess_days

        # Inactivity penalty if not trading for too long
        if self.days_since_last_trade > self.config.inactivity_threshold:
            inactivity_penalty_value = self.config.inactivity_penalty

        # Combine components
        reward = (
            daily_return
            + trade_bonus
            - holding_penalty_value
            - inactivity_penalty_value
        )

        return reward

    def reset(self) -> None:
        """Reset reward function state for new episode."""
        super().reset()
        self.days_in_position = 0
        self.days_since_last_trade = 0
        self.in_position = False
        self.last_action = 0
        logger.debug(f"{self.__class__.__name__} reset (position tracking cleared)")


def create_reward_function(config: RewardConfig) -> RewardFunction:
    """
    Factory function to create reward function from config.

    Args:
        config: Reward configuration

    Returns:
        RewardFunction instance

    Raises:
        ValueError: If reward_type is not recognized

    Example:
        >>> config = RewardConfig(reward_type='risk_adjusted')
        >>> reward_fn = create_reward_function(config)
        >>> isinstance(reward_fn, RiskAdjustedReward)
        True

        >>> config = RewardConfig(reward_type='sortino', sortino_window=20)
        >>> reward_fn = create_reward_function(config)
        >>> isinstance(reward_fn, SortinoReward)
        True
    """
    reward_map = {
        "simple": SimpleReward,
        "risk_adjusted": RiskAdjustedReward,
        "sharpe": SharpeReward,
        "sortino": SortinoReward,
        "volatility_penalized": VolatilityPenalizedReward,
        "enhanced_drawdown": EnhancedDrawdownReward,
        "calmar": CalmarReward,
        "active_trading": ActiveTradingReward,
    }

    if config.reward_type not in reward_map:
        raise ValueError(
            f"Unknown reward_type: {config.reward_type}. "
            f"Must be one of {list(reward_map.keys())}"
        )

    reward_class = reward_map[config.reward_type]
    logger.info(f"Creating {reward_class.__name__} with config: {config}")
    return reward_class(config)
