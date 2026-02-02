"""Transaction cost and slippage models for realistic trading simulation."""

from dataclasses import dataclass
from typing import Dict, Tuple

from loguru import logger


@dataclass
class CostConfig:
    """
    Configuration for transaction costs (Zerodha NSE equity delivery).

    Attributes:
        brokerage_pct: Brokerage as percentage of trade value
        brokerage_cap: Maximum brokerage per trade (₹20 for Zerodha)
        stt_pct: Securities Transaction Tax (0.1% on sell side for delivery, 0.025% intraday)
        transaction_charges_pct: NSE transaction charges
        gst_rate: GST rate on brokerage + transaction charges
        stamp_duty_pct: Stamp duty on buy side
        slippage_pct: Estimated slippage percentage (for EOD, flat model)

        # Intraday-specific parameters (dynamic slippage model)
        use_dynamic_slippage: Whether to use dynamic slippage (bid-ask + impact)
        base_spread_bps: Base bid-ask spread in basis points (default: 2.5 bps)
        spread_volatility_multiplier: Scale spread with volatility (ATR)
        max_spread_bps: Maximum spread cap in basis points (default: 10 bps)
        impact_coefficient: Market impact coefficient for volume-based impact
        max_impact_pct: Maximum market impact cap as percentage
    """

    # Fixed costs (same for EOD and intraday)
    brokerage_pct: float = 0.0003  # 0.03%
    brokerage_cap: float = 20.0  # ₹20 per trade
    stt_pct: float = 0.001  # 0.1% on sell side (delivery)
    transaction_charges_pct: float = 0.0000325  # 0.00325%
    gst_rate: float = 0.18  # 18% on brokerage + transaction charges
    stamp_duty_pct: float = 0.00015  # 0.015% on buy side

    # EOD slippage (flat model)
    slippage_pct: float = 0.001  # 0.1% slippage

    # Intraday dynamic slippage model
    use_dynamic_slippage: bool = False  # Enable for intraday trading
    base_spread_bps: float = 2.5  # 2.5 basis points (0.025%)
    spread_volatility_multiplier: bool = True  # Scale with ATR
    max_spread_bps: float = 10.0  # Cap at 10 bps
    impact_coefficient: float = 0.2  # Calibrated to NSE markets
    max_impact_pct: float = 0.005  # Cap at 0.5%


class TransactionCostModel:
    """
    Calculate realistic transaction costs for Indian equity trading.

    Models Zerodha-style costs including:
    - Brokerage (capped at ₹20)
    - STT (Securities Transaction Tax)
    - Transaction charges
    - GST
    - Stamp duty
    - Slippage

    Typical round-trip cost: 0.15-0.2%

    Example:
        >>> config = CostConfig()
        >>> model = TransactionCostModel(config)
        >>> total_cost, breakdown = model.calculate_buy_cost(100, 1000)
        >>> print(f"Total: ₹{total_cost:.2f}")
    """

    def __init__(self, config: CostConfig) -> None:
        """
        Initialize transaction cost model.

        Args:
            config: Cost configuration parameters
        """
        self.config = config
        logger.info("TransactionCostModel initialized with Zerodha-style costs")

    def calculate_buy_cost(
        self,
        shares: int,
        price: float,
        volume: float = 1000000,
        atr: float = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total cost to buy shares including all fees.

        Args:
            shares: Number of shares to buy
            price: Price per share
            volume: Bar volume (for dynamic slippage calculation)
            atr: Average True Range (for volatility-based slippage)

        Returns:
            Tuple of (total_cost, breakdown_dict)
            - total_cost: Total amount to debit (trade value + all costs)
            - breakdown: Dictionary with itemized costs

        Example:
            >>> model = TransactionCostModel(CostConfig())
            >>> cost, breakdown = model.calculate_buy_cost(100, 1000)
            >>> print(f"Effective rate: {breakdown['effective_rate']*100:.3f}%")
        """
        trade_value = shares * price

        # Brokerage (capped at ₹20)
        brokerage = min(trade_value * self.config.brokerage_pct, self.config.brokerage_cap)

        # Transaction charges
        transaction_charges = trade_value * self.config.transaction_charges_pct

        # GST on brokerage + transaction charges
        gst = (brokerage + transaction_charges) * self.config.gst_rate

        # Stamp duty (buy side only)
        stamp_duty = trade_value * self.config.stamp_duty_pct

        # Slippage (pay higher price)
        if self.config.use_dynamic_slippage:
            # Dynamic slippage: bid-ask spread + market impact
            if atr is None:
                atr = price * 0.01  # Default to 1% of price
            slippage = self._calculate_dynamic_slippage(
                shares, price, volume, atr, direction="buy"
            )
        else:
            # Flat slippage
            slippage = trade_value * self.config.slippage_pct

        total_cost = brokerage + transaction_charges + gst + stamp_duty + slippage

        breakdown = {
            "trade_value": trade_value,
            "brokerage": brokerage,
            "transaction_charges": transaction_charges,
            "gst": gst,
            "stamp_duty": stamp_duty,
            "slippage": slippage,
            "total_cost": total_cost,
            "effective_rate": total_cost / trade_value if trade_value > 0 else 0.0,
        }

        logger.debug(
            f"Buy cost: ₹{total_cost:.2f} ({breakdown['effective_rate']*100:.3f}%) "
            f"for {shares} shares @ ₹{price:.2f}"
        )

        return trade_value + total_cost, breakdown

    def calculate_sell_proceeds(
        self,
        shares: int,
        price: float,
        volume: float = 1000000,
        atr: float = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate net proceeds from selling shares after all fees.

        Args:
            shares: Number of shares to sell
            price: Price per share
            volume: Bar volume (for dynamic slippage calculation)
            atr: Average True Range (for volatility-based slippage)

        Returns:
            Tuple of (net_proceeds, breakdown_dict)
            - net_proceeds: Amount credited to account (trade value - all costs)
            - breakdown: Dictionary with itemized costs

        Example:
            >>> model = TransactionCostModel(CostConfig())
            >>> proceeds, breakdown = model.calculate_sell_proceeds(100, 1100)
            >>> print(f"Net proceeds: ₹{proceeds:.2f}")
        """
        trade_value = shares * price

        # Brokerage (capped at ₹20)
        brokerage = min(trade_value * self.config.brokerage_pct, self.config.brokerage_cap)

        # STT (sell side only)
        stt = trade_value * self.config.stt_pct

        # Transaction charges
        transaction_charges = trade_value * self.config.transaction_charges_pct

        # GST
        gst = (brokerage + transaction_charges) * self.config.gst_rate

        # Slippage (receive lower price)
        if self.config.use_dynamic_slippage:
            # Dynamic slippage: bid-ask spread + market impact
            if atr is None:
                atr = price * 0.01  # Default to 1% of price
            slippage = self._calculate_dynamic_slippage(
                shares, price, volume, atr, direction="sell"
            )
        else:
            # Flat slippage
            slippage = trade_value * self.config.slippage_pct

        total_cost = brokerage + stt + transaction_charges + gst + slippage

        breakdown = {
            "trade_value": trade_value,
            "brokerage": brokerage,
            "stt": stt,
            "transaction_charges": transaction_charges,
            "gst": gst,
            "slippage": slippage,
            "total_cost": total_cost,
            "effective_rate": total_cost / trade_value if trade_value > 0 else 0.0,
        }

        logger.debug(
            f"Sell proceeds: ₹{trade_value - total_cost:.2f} "
            f"({breakdown['effective_rate']*100:.3f}% cost) "
            f"for {shares} shares @ ₹{price:.2f}"
        )

        return trade_value - total_cost, breakdown

    def calculate_round_trip_cost_pct(self, price: float = 1000.0) -> float:
        """
        Calculate round-trip cost percentage (buy + sell at same price).

        Args:
            price: Price per share (default: ₹1000)

        Returns:
            Round-trip cost as percentage (should be ~0.15-0.2%)

        Example:
            >>> model = TransactionCostModel(CostConfig())
            >>> rt_cost = model.calculate_round_trip_cost_pct()
            >>> print(f"Round-trip: {rt_cost*100:.3f}%")
        """
        shares = 100

        # Buy cost
        buy_total, _ = self.calculate_buy_cost(shares, price)
        buy_cost = buy_total - (shares * price)

        # Sell proceeds
        sell_proceeds, _ = self.calculate_sell_proceeds(shares, price)
        sell_cost = (shares * price) - sell_proceeds

        # Total round-trip cost
        total_cost = buy_cost + sell_cost
        round_trip_pct = total_cost / (shares * price)

        logger.info(f"Round-trip cost: {round_trip_pct*100:.3f}%")

        return round_trip_pct

    def calculate_sell_revenue(
        self,
        shares: int,
        price: float,
        volume: float = 1000000,
        atr: float = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Alias for calculate_sell_proceeds (for consistency with intraday environment).

        Returns the same result as calculate_sell_proceeds.
        """
        return self.calculate_sell_proceeds(shares, price, volume, atr)

    def _calculate_dynamic_slippage(
        self,
        shares: int,
        price: float,
        volume: float,
        atr: float,
        direction: str,
    ) -> float:
        """
        Calculate dynamic slippage based on bid-ask spread and market impact.

        Uses a two-component model:
        1. Bid-ask spread: Base spread scaled by volatility (ATR)
        2. Market impact: Square root model based on volume participation

        Args:
            shares: Number of shares to trade
            price: Price per share
            volume: Bar volume
            atr: Average True Range (volatility measure)
            direction: "buy" or "sell"

        Returns:
            Total slippage cost in rupees
        """
        trade_value = shares * price

        # Component 1: Bid-ask spread
        # Base spread in basis points
        spread_bps = self.config.base_spread_bps

        # Scale by volatility if enabled
        if self.config.spread_volatility_multiplier and atr > 0:
            # Volatility multiplier: normalized to 1% ATR
            volatility_ratio = (atr / price) / 0.01
            spread_bps *= max(1.0, volatility_ratio)

        # Cap spread
        spread_bps = min(spread_bps, self.config.max_spread_bps)

        # Convert bps to decimal and apply half-spread cost
        spread_cost_pct = (spread_bps / 10000) / 2  # Half-spread
        spread_cost = trade_value * spread_cost_pct

        # Component 2: Market impact (square root model)
        # Volume participation rate
        volume_participation = shares / volume if volume > 0 else 0.0

        # Market impact (square root law)
        impact_pct = self.config.impact_coefficient * (volume_participation ** 0.5) * (atr / price)

        # Cap impact
        impact_pct = min(impact_pct, self.config.max_impact_pct)

        impact_cost = trade_value * impact_pct

        # Total slippage
        total_slippage = spread_cost + impact_cost

        logger.debug(
            f"Dynamic slippage ({direction}): spread={spread_cost:.2f} ({spread_bps:.2f}bps), "
            f"impact={impact_cost:.2f} ({impact_pct*100:.3f}%), "
            f"total={total_slippage:.2f}"
        )

        return total_slippage
