"""Live trading broker implementation for Zerodha Kite Connect API."""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event, Lock, Thread
from typing import Dict, List, Optional

from kiteconnect import KiteConnect
from kiteconnect.exceptions import (
    DataException,
    GeneralException,
    NetworkException,
    TokenException,
)
from loguru import logger

from tradebox.execution.base_broker import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    Portfolio,
    Position,
)
from tradebox.execution.exceptions import (
    CircuitBreakerTriggered,
    OrderPlacementException,
    RateLimitException,
)
from tradebox.execution.retry import RetryConfig, RetryHandler, RetryableException
from tradebox.risk.circuit_breakers import APIFailureCircuitBreaker


@dataclass
class KiteBrokerConfig:
    """
    Configuration for KiteBroker.

    Attributes:
        api_key: Kite API key (loaded from env: KITE_API_KEY)
        api_secret: Kite API secret (loaded from env: KITE_API_SECRET)
        access_token: Access token (optional, generated if None)

        # Reconciliation settings
        reconciliation_enabled: Enable background reconciliation
        reconciliation_interval_seconds: Seconds between reconciliation runs

        # Rate limiting
        max_orders_per_second: Maximum orders per second (Kite limit: 10)
        max_orders_per_minute: Maximum orders per minute (Kite limit: 200)
        max_orders_per_day: Maximum orders per day (Kite limit: 3000)

        # Retry configuration
        retry_config: RetryConfig for exponential backoff

        # Cache settings
        cache_ttl_seconds: Time-to-live for cached data (default: 5 seconds)
        positions_cache_ttl: TTL for positions cache (default: 10 seconds)
    """

    api_key: str
    api_secret: str
    access_token: Optional[str] = None

    reconciliation_enabled: bool = True
    reconciliation_interval_seconds: int = 30  # Every 30 seconds

    max_orders_per_second: int = 8  # Conservative (limit: 10)
    max_orders_per_minute: int = 180  # Conservative (limit: 200)
    max_orders_per_day: int = 2500  # Conservative (limit: 3000)

    retry_config: Optional[RetryConfig] = None

    cache_ttl_seconds: int = 5
    positions_cache_ttl: int = 10


class KiteBroker(BaseBroker):
    """
    Live trading broker for Zerodha Kite Connect API.

    Features:
    - Exponential backoff retry logic for network failures
    - Automatic order reconciliation with broker state
    - Thread-safe operation with local caching
    - Circuit breaker integration for API failures
    - Comprehensive error handling and logging

    Attributes:
        kite: KiteConnect API client instance
        config: KiteBrokerConfig with API credentials and settings
        _orders_cache: Local cache of Order objects (Dict[order_id, Order])
        _positions_cache: Local cache of Position objects (Dict[symbol, Position])
        _cache_lock: Thread lock for cache synchronization
        _reconciliation_thread: Background thread for periodic reconciliation
        _reconciliation_enabled: Flag to control reconciliation
        circuit_breaker: APIFailureCircuitBreaker for tracking failures
        retry_handler: RetryHandler for exponential backoff logic

    Example:
        >>> from tradebox.execution.config import load_kite_broker_config
        >>> config = load_kite_broker_config()
        >>> broker = KiteBroker(config)
        >>>
        >>> # Place order
        >>> order = broker.place_order("RELIANCE", OrderSide.BUY, 10)
        >>> print(f"Order {order.order_id}: {order.status}")
    """

    def __init__(
        self,
        config: KiteBrokerConfig,
        circuit_breaker: Optional[APIFailureCircuitBreaker] = None,
    ) -> None:
        """
        Initialize KiteBroker.

        Args:
            config: Broker configuration
            circuit_breaker: Optional circuit breaker for API failures

        Raises:
            TokenException: If authentication fails
            RuntimeError: If initialization fails
        """
        self.config = config

        # Initialize Kite API client
        self.kite = KiteConnect(api_key=config.api_key)

        # Set access token if provided, otherwise needs login flow
        if config.access_token:
            self.kite.set_access_token(config.access_token)
        else:
            logger.warning(
                "No access token provided. Call generate_session() "
                "after user login flow."
            )

        # Initialize retry handler
        retry_config = config.retry_config or RetryConfig()
        self.retry_handler = RetryHandler(retry_config)

        # Circuit breaker for API failures
        self.circuit_breaker = circuit_breaker or APIFailureCircuitBreaker()

        # Local caches
        self._orders_cache: Dict[str, Order] = {}
        self._positions_cache: Dict[str, Position] = {}
        self._cache_lock = Lock()

        # Cache metadata
        self._orders_cache_time: Optional[datetime] = None
        self._positions_cache_time: Optional[datetime] = None

        # Rate limiting tracking
        self._order_times: List[datetime] = []
        self._daily_order_count = 0
        self._daily_reset_date: Optional[datetime] = None

        # Reconciliation thread
        self._reconciliation_thread: Optional[Thread] = None
        self._reconciliation_stop_event = Event()

        if config.reconciliation_enabled:
            self._start_reconciliation()

        logger.info(
            f"KiteBroker initialized (reconciliation: "
            f"{config.reconciliation_enabled})"
        )

    def generate_session(self, request_token: str) -> Dict[str, str]:
        """
        Generate session and set access token after user login.

        Args:
            request_token: Request token from Kite login redirect

        Returns:
            Session data with access_token

        Raises:
            TokenException: If session generation fails
        """
        try:
            data = self.kite.generate_session(
                request_token,
                api_secret=self.config.api_secret,
            )
            self.kite.set_access_token(data["access_token"])
            logger.info("Session generated successfully")
            return data
        except TokenException as e:
            logger.error(f"Session generation failed: {e}")
            self.circuit_breaker.record_failure(e)
            raise

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Optional[float] = None,
    ) -> Order:
        """
        Place a trading order with retry logic.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            side: Buy or sell
            quantity: Number of shares
            price: Limit price (None for market order)

        Returns:
            Order object with order details

        Raises:
            ValueError: If order is invalid
            RuntimeError: If order placement fails after retries
            CircuitBreakerTriggered: If circuit breaker is triggered
            RateLimitException: If rate limit would be exceeded
        """
        # Validate order
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        # Check circuit breaker
        if self.circuit_breaker.is_triggered:
            raise CircuitBreakerTriggered(
                f"Trading halted: {self.circuit_breaker.trigger_reason}"
            )

        # Check rate limits
        self._check_rate_limits()

        # Convert to Kite format
        transaction_type = (
            self.kite.TRANSACTION_TYPE_BUY
            if side == OrderSide.BUY
            else self.kite.TRANSACTION_TYPE_SELL
        )

        order_type = (
            self.kite.ORDER_TYPE_MARKET
            if price is None
            else self.kite.ORDER_TYPE_LIMIT
        )

        # Place order with retry logic
        try:
            order_id = self._place_order_with_retry(
                symbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                price=price,
            )

            # Create Order object
            order = Order(
                order_id=str(order_id),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,
                timestamp=datetime.now(),
            )

            # Update cache
            with self._cache_lock:
                self._orders_cache[order.order_id] = order

            # Track for rate limiting
            self._record_order_placed()

            logger.info(
                f"Order placed: {order_id} - {side.value} {quantity} "
                f"{symbol} @ {price or 'MARKET'}"
            )

            # Fetch actual order status from Kite (best practice)
            return self._sync_order_status(order)

        except RetryableException as e:
            # All retries exhausted
            logger.error(f"Order placement failed after retries: {e}")
            self.circuit_breaker.record_failure(e)
            raise OrderPlacementException(f"Failed to place order: {e}") from e

    def _place_order_with_retry(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: Optional[float],
    ) -> str:
        """
        Place order with automatic retry on failures.

        This method uses retry logic via retry_handler.

        Returns:
            Order ID string

        Raises:
            RetryableException: On network/timeout errors (triggers retry)
            ValueError: On invalid parameters (no retry)
        """

        @self.retry_handler.retry()
        def _do_place_order():
            try:
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    tradingsymbol=symbol,
                    exchange=self.kite.EXCHANGE_NSE,
                    transaction_type=transaction_type,
                    quantity=quantity,
                    order_type=order_type,
                    price=price,
                    product=self.kite.PRODUCT_CNC,  # Delivery trading
                    validity=self.kite.VALIDITY_DAY,
                )

                return str(order_id)

            except NetworkException as e:
                # Network/timeout errors are retryable
                logger.warning(f"Network error placing order: {e}")
                raise RetryableException(f"Network error: {e}") from e

            except (TokenException, DataException) as e:
                # Authentication/data errors are NOT retryable
                logger.error(f"Non-retryable error placing order: {e}")
                raise ValueError(f"Order placement failed: {e}") from e

            except GeneralException as e:
                # Check if it's a timeout (retryable) or validation error (not)
                error_msg = str(e).lower()
                if "timeout" in error_msg or "503" in error_msg or "504" in error_msg:
                    logger.warning(f"Timeout error placing order: {e}")
                    raise RetryableException(f"Timeout: {e}") from e
                else:
                    logger.error(f"Order rejected: {e}")
                    raise ValueError(f"Order rejected: {e}") from e

        return _do_place_order()

    def _sync_order_status(self, order: Order) -> Order:
        """
        Synchronize order status with Kite API.

        CRITICAL: After placing an order, always check the orderbook
        to get the actual status. The API may timeout but the order
        could still be placed successfully.

        Args:
            order: Order object to sync

        Returns:
            Updated Order object
        """
        try:
            # Fetch order details from Kite
            kite_orders = self.kite.orders()

            # Find matching order
            for kite_order in kite_orders:
                if str(kite_order["order_id"]) == order.order_id:
                    # Update order with Kite data
                    order.status = self._map_kite_status(kite_order["status"])
                    order.filled_price = kite_order.get("average_price")
                    order.filled_quantity = kite_order.get("filled_quantity", 0)

                    # Update cache
                    with self._cache_lock:
                        self._orders_cache[order.order_id] = order

                    logger.debug(
                        f"Order {order.order_id} synced: " f"status={order.status.value}"
                    )
                    break

            return order

        except Exception as e:
            logger.warning(f"Failed to sync order status: {e}")
            # Return original order if sync fails
            return order

    def _map_kite_status(self, kite_status: str) -> OrderStatus:
        """
        Map Kite order status to our OrderStatus enum.

        Kite statuses: PENDING, OPEN, COMPLETE, REJECTED, CANCELLED
        Our statuses: PENDING, FILLED, REJECTED, CANCELLED

        Args:
            kite_status: Status string from Kite API

        Returns:
            OrderStatus enum
        """
        status_map = {
            "PENDING": OrderStatus.PENDING,
            "OPEN": OrderStatus.PENDING,  # Order placed but not filled
            "COMPLETE": OrderStatus.FILLED,
            "REJECTED": OrderStatus.REJECTED,
            "CANCELLED": OrderStatus.CANCELLED,
        }

        return status_map.get(
            kite_status.upper(),
            OrderStatus.PENDING,  # Default to pending
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None if no position
        """
        # Check cache freshness
        if self._is_positions_cache_fresh():
            with self._cache_lock:
                return self._positions_cache.get(symbol)

        # Refresh cache
        self._refresh_positions_cache()

        with self._cache_lock:
            return self._positions_cache.get(symbol)

    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects
        """
        # Refresh cache if stale
        if not self._is_positions_cache_fresh():
            self._refresh_positions_cache()

        with self._cache_lock:
            return list(self._positions_cache.values())

    def _refresh_positions_cache(self) -> None:
        """
        Refresh positions cache from Kite API.

        Raises:
            RetryableException: On network errors
        """

        @self.retry_handler.retry()
        def _do_refresh():
            try:
                # Fetch positions from Kite
                kite_positions = self.kite.positions()

                # Extract day positions (net = day + overnight)
                day_positions = kite_positions.get("day", [])

                new_cache = {}

                for kite_pos in day_positions:
                    # Only include positions with non-zero quantity
                    quantity = kite_pos.get("quantity", 0)
                    if quantity == 0:
                        continue

                    symbol = kite_pos["tradingsymbol"]

                    position = Position(
                        symbol=symbol,
                        quantity=quantity,
                        avg_price=kite_pos.get("average_price", 0.0),
                        current_price=kite_pos.get("last_price", 0.0),
                        unrealized_pnl=kite_pos.get("pnl", 0.0),
                        realized_pnl=kite_pos.get("realised", 0.0),
                    )

                    new_cache[symbol] = position

                # Update cache
                with self._cache_lock:
                    self._positions_cache = new_cache
                    self._positions_cache_time = datetime.now()

                logger.debug(f"Positions cache refreshed: {len(new_cache)} positions")

            except NetworkException as e:
                logger.warning(f"Network error refreshing positions: {e}")
                raise RetryableException(f"Network error: {e}") from e

            except Exception as e:
                logger.error(f"Failed to refresh positions: {e}")
                self.circuit_breaker.record_failure(e)
                raise

        _do_refresh()

    def get_portfolio(self) -> Portfolio:
        """
        Get complete portfolio information.

        Returns:
            Portfolio object with cash, positions, and total value
        """
        # Get positions
        positions_list = self.get_positions()
        positions_dict = {p.symbol: p for p in positions_list}

        # Get cash balance
        cash = self.get_cash()

        # Create portfolio
        portfolio = Portfolio(
            cash=cash,
            positions=positions_dict,
        )
        portfolio.update_total_value()

        return portfolio

    def get_cash(self) -> float:
        """
        Get available cash balance.

        Returns:
            Available cash in rupees

        Raises:
            RuntimeError: If cash fetch fails
        """

        @self.retry_handler.retry()
        def _do_get_cash():
            try:
                # Fetch margins from Kite
                margins = self.kite.margins()

                # Get equity segment margin
                equity_margin = margins.get("equity", {})
                available_cash = equity_margin.get("available", {}).get("cash", 0.0)

                logger.debug(f"Available cash: ₹{available_cash:,.2f}")
                return available_cash

            except NetworkException as e:
                logger.warning(f"Network error fetching cash: {e}")
                raise RetryableException(f"Network error: {e}") from e

            except Exception as e:
                logger.error(f"Failed to fetch cash: {e}")
                self.circuit_breaker.record_failure(e)
                raise RuntimeError(f"Failed to fetch cash: {e}") from e

        return _do_get_cash()

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order identifier

        Returns:
            Order object or None if not found
        """
        # Check cache first
        with self._cache_lock:
            if order_id in self._orders_cache:
                cached_order = self._orders_cache[order_id]

                # For pending orders, sync with Kite for latest status
                if cached_order.status == OrderStatus.PENDING:
                    return self._sync_order_status(cached_order)

                return cached_order

        # Not in cache, fetch from Kite
        try:
            kite_orders = self.kite.orders()

            for kite_order in kite_orders:
                if str(kite_order["order_id"]) == order_id:
                    order = self._convert_kite_order(kite_order)

                    # Update cache
                    with self._cache_lock:
                        self._orders_cache[order_id] = order

                    return order

        except Exception as e:
            logger.warning(f"Failed to fetch order {order_id}: {e}")

        return None

    def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
    ) -> List[Order]:
        """
        Get orders with optional filters.

        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)

        Returns:
            List of Order objects

        Raises:
            RuntimeError: If fetch fails
        """
        # Fetch all orders from Kite
        try:
            kite_orders = self.kite.orders()

            orders = []
            for kite_order in kite_orders:
                order = self._convert_kite_order(kite_order)

                # Apply filters
                if symbol is not None and order.symbol != symbol:
                    continue

                if status is not None and order.status != status:
                    continue

                orders.append(order)

                # Update cache
                with self._cache_lock:
                    self._orders_cache[order.order_id] = order

            return orders

        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            self.circuit_breaker.record_failure(e)
            raise RuntimeError(f"Failed to fetch orders: {e}") from e

    def _convert_kite_order(self, kite_order: Dict) -> Order:
        """
        Convert Kite order dict to our Order object.

        Args:
            kite_order: Order dict from Kite API

        Returns:
            Order object
        """
        return Order(
            order_id=str(kite_order["order_id"]),
            symbol=kite_order["tradingsymbol"],
            side=(
                OrderSide.BUY
                if kite_order["transaction_type"] == "BUY"
                else OrderSide.SELL
            ),
            quantity=kite_order["quantity"],
            price=kite_order.get("price"),
            status=self._map_kite_status(kite_order["status"]),
            timestamp=kite_order.get("order_timestamp", datetime.now()),
            filled_price=kite_order.get("average_price"),
            filled_quantity=kite_order.get("filled_quantity", 0),
        )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order identifier

        Returns:
            True if cancelled successfully, False otherwise
        """

        @self.retry_handler.retry()
        def _do_cancel():
            try:
                # Check if order exists and is cancellable
                order = self.get_order(order_id)

                if order is None:
                    logger.warning(f"Order not found: {order_id}")
                    return False

                if order.status != OrderStatus.PENDING:
                    logger.warning(
                        f"Cannot cancel order {order_id}: "
                        f"status is {order.status.value}"
                    )
                    return False

                # Cancel via Kite API
                self.kite.cancel_order(
                    variety=self.kite.VARIETY_REGULAR,
                    order_id=order_id,
                )

                # Update cache
                order.status = OrderStatus.CANCELLED
                with self._cache_lock:
                    self._orders_cache[order_id] = order

                logger.info(f"Order cancelled: {order_id}")
                return True

            except NetworkException as e:
                logger.warning(f"Network error cancelling order: {e}")
                raise RetryableException(f"Network error: {e}") from e

            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                return False

        return _do_cancel()

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get current market quote.

        NEVER cached - always fetch fresh for trading decisions.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with price information
        """

        @self.retry_handler.retry()
        def _do_get_quote():
            try:
                # Kite requires exchange:symbol format
                instrument = f"NSE:{symbol}"

                quote = self.kite.quote([instrument])

                if instrument not in quote:
                    logger.warning(f"No quote data for {symbol}")
                    return {}

                quote_data = quote[instrument]

                depth_buy = quote_data.get("depth", {}).get("buy", [])
                depth_sell = quote_data.get("depth", {}).get("sell", [])

                return {
                    "symbol": symbol,
                    "last": quote_data.get("last_price", 0.0),
                    "bid": depth_buy[0].get("price", 0.0) if depth_buy else 0.0,
                    "ask": depth_sell[0].get("price", 0.0) if depth_sell else 0.0,
                    "volume": quote_data.get("volume", 0),
                    "timestamp": datetime.now().isoformat(),
                }

            except NetworkException as e:
                logger.warning(f"Network error fetching quote: {e}")
                raise RetryableException(f"Network error: {e}") from e

            except Exception as e:
                logger.error(f"Failed to fetch quote for {symbol}: {e}")
                return {}

        return _do_get_quote()

    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close entire position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Order object for the closing trade, or None if no position
        """
        pos = self.get_position(symbol)

        if pos is None or pos.quantity == 0:
            logger.warning(f"No position to close: {symbol}")
            return None

        # Place market sell order for full quantity
        return self.place_order(symbol, OrderSide.SELL, pos.quantity)

    # ==================== Reconciliation System ====================

    def _start_reconciliation(self) -> None:
        """Start background reconciliation thread."""
        self._reconciliation_stop_event.clear()
        self._reconciliation_thread = Thread(
            target=self._reconciliation_loop,
            daemon=True,
            name="KiteBrokerReconciliation",
        )
        self._reconciliation_thread.start()
        logger.info("Reconciliation thread started")

    def _reconciliation_loop(self) -> None:
        """
        Background reconciliation loop.

        Periodically syncs local cache with Kite API to ensure consistency.
        """
        while not self._reconciliation_stop_event.is_set():
            try:
                # Sleep first (configurable interval)
                self._reconciliation_stop_event.wait(
                    self.config.reconciliation_interval_seconds
                )

                if self._reconciliation_stop_event.is_set():
                    break

                # Perform reconciliation
                self._reconcile_orders()
                self._reconcile_positions()

            except Exception as e:
                logger.error(f"Reconciliation error: {e}")
                # Continue running despite errors

        logger.info("Reconciliation thread stopped")

    def _reconcile_orders(self) -> None:
        """
        Reconcile orders between local cache and Kite API.

        Checks for discrepancies and logs/alerts on mismatches.
        """
        try:
            # Fetch all orders from Kite
            kite_orders = self.kite.orders()

            # Build Kite orders dict
            kite_orders_dict = {str(order["order_id"]): order for order in kite_orders}

            # Compare with cache
            with self._cache_lock:
                cached_order_ids = set(self._orders_cache.keys())
                kite_order_ids = set(kite_orders_dict.keys())

                # Check for orders in cache but not in Kite (shouldn't happen)
                missing_from_kite = cached_order_ids - kite_order_ids
                if missing_from_kite:
                    logger.warning(
                        f"Reconciliation: {len(missing_from_kite)} orders "
                        f"in cache but not in Kite: {missing_from_kite}"
                    )

                # Check for orders in Kite but not in cache (could be manual orders)
                missing_from_cache = kite_order_ids - cached_order_ids
                if missing_from_cache:
                    logger.info(
                        f"Reconciliation: {len(missing_from_cache)} new orders "
                        f"from Kite (possibly manual): {missing_from_cache}"
                    )

                    # Add to cache
                    for order_id in missing_from_cache:
                        order = self._convert_kite_order(kite_orders_dict[order_id])
                        self._orders_cache[order_id] = order

                # Check for status discrepancies
                discrepancies = []
                for order_id in cached_order_ids & kite_order_ids:
                    cached_order = self._orders_cache[order_id]
                    kite_order = kite_orders_dict[order_id]

                    kite_status = self._map_kite_status(kite_order["status"])

                    if cached_order.status != kite_status:
                        discrepancies.append(
                            f"{order_id}: cache={cached_order.status.value}, "
                            f"kite={kite_status.value}"
                        )

                        # Update cache to match Kite (source of truth)
                        cached_order.status = kite_status
                        cached_order.filled_price = kite_order.get("average_price")
                        cached_order.filled_quantity = kite_order.get(
                            "filled_quantity", 0
                        )
                        self._orders_cache[order_id] = cached_order

                if discrepancies:
                    logger.warning(
                        f"Reconciliation: {len(discrepancies)} status "
                        f"discrepancies fixed: {discrepancies[:5]}"
                    )

            logger.debug("Order reconciliation completed")

        except Exception as e:
            logger.error(f"Order reconciliation failed: {e}")

    def _reconcile_positions(self) -> None:
        """
        Reconcile positions between local cache and Kite API.

        Ensures position quantities and P&L are accurate.
        """
        try:
            # Fetch positions from Kite
            kite_positions = self.kite.positions()
            day_positions = kite_positions.get("day", [])

            # Build Kite positions dict
            kite_positions_dict = {}
            for kite_pos in day_positions:
                symbol = kite_pos["tradingsymbol"]
                quantity = kite_pos.get("quantity", 0)

                if quantity != 0:
                    kite_positions_dict[symbol] = kite_pos

            # Compare with cache
            with self._cache_lock:
                cached_symbols = set(self._positions_cache.keys())
                kite_symbols = set(kite_positions_dict.keys())

                # Check for positions in cache but not in Kite
                missing_from_kite = cached_symbols - kite_symbols
                if missing_from_kite:
                    logger.warning(
                        f"Reconciliation: {len(missing_from_kite)} positions "
                        f"in cache but not in Kite: {missing_from_kite}"
                    )

                    # Remove from cache
                    for symbol in missing_from_kite:
                        del self._positions_cache[symbol]

                # Check for positions in Kite but not in cache
                missing_from_cache = kite_symbols - cached_symbols
                if missing_from_cache:
                    logger.info(
                        f"Reconciliation: {len(missing_from_cache)} new positions "
                        f"from Kite: {missing_from_cache}"
                    )

                    # Add to cache
                    for symbol in missing_from_cache:
                        kite_pos = kite_positions_dict[symbol]
                        position = Position(
                            symbol=symbol,
                            quantity=kite_pos.get("quantity", 0),
                            avg_price=kite_pos.get("average_price", 0.0),
                            current_price=kite_pos.get("last_price", 0.0),
                            unrealized_pnl=kite_pos.get("pnl", 0.0),
                            realized_pnl=kite_pos.get("realised", 0.0),
                        )
                        self._positions_cache[symbol] = position

                # Check for quantity discrepancies
                discrepancies = []
                for symbol in cached_symbols & kite_symbols:
                    cached_pos = self._positions_cache[symbol]
                    kite_pos = kite_positions_dict[symbol]

                    kite_qty = kite_pos.get("quantity", 0)

                    if cached_pos.quantity != kite_qty:
                        discrepancies.append(
                            f"{symbol}: cache={cached_pos.quantity}, " f"kite={kite_qty}"
                        )

                        # Update cache to match Kite (source of truth)
                        cached_pos.quantity = kite_qty
                        cached_pos.avg_price = kite_pos.get("average_price", 0.0)
                        cached_pos.current_price = kite_pos.get("last_price", 0.0)
                        cached_pos.unrealized_pnl = kite_pos.get("pnl", 0.0)
                        cached_pos.realized_pnl = kite_pos.get("realised", 0.0)
                        self._positions_cache[symbol] = cached_pos

                if discrepancies:
                    logger.warning(
                        f"Reconciliation: {len(discrepancies)} position "
                        f"discrepancies fixed: {discrepancies}"
                    )

                # Update cache timestamp
                self._positions_cache_time = datetime.now()

            logger.debug("Position reconciliation completed")

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")

    def stop_reconciliation(self) -> None:
        """Stop background reconciliation thread."""
        if self._reconciliation_thread and self._reconciliation_thread.is_alive():
            logger.info("Stopping reconciliation thread...")
            self._reconciliation_stop_event.set()
            self._reconciliation_thread.join(timeout=5.0)
            logger.info("Reconciliation thread stopped")

    # ==================== Rate Limiting ====================

    def _check_rate_limits(self) -> None:
        """
        Check rate limits before placing order.

        Kite limits:
        - 10 orders per second
        - 200 orders per minute
        - 3000 orders per day

        Raises:
            RateLimitException: If rate limit would be exceeded
        """
        now = datetime.now()

        # Check daily limit
        if self._daily_reset_date is None or now.date() > self._daily_reset_date:
            # Reset daily counter
            self._daily_order_count = 0
            self._daily_reset_date = now.date()

        if self._daily_order_count >= self.config.max_orders_per_day:
            raise RateLimitException(
                f"Daily order limit reached: "
                f"{self._daily_order_count}/{self.config.max_orders_per_day}"
            )

        # Remove old timestamps (older than 1 minute)
        cutoff_time = now - timedelta(minutes=1)
        self._order_times = [t for t in self._order_times if t > cutoff_time]

        # Check per-minute limit
        if len(self._order_times) >= self.config.max_orders_per_minute:
            raise RateLimitException(
                f"Per-minute order limit reached: "
                f"{len(self._order_times)}/{self.config.max_orders_per_minute}"
            )

        # Check per-second limit
        one_second_ago = now - timedelta(seconds=1)
        recent_orders = sum(1 for t in self._order_times if t > one_second_ago)

        if recent_orders >= self.config.max_orders_per_second:
            # Brief pause to avoid hitting limit
            sleep_time = 1.0 - (now - self._order_times[-1]).total_seconds()
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

    def _record_order_placed(self) -> None:
        """Record that an order was placed (for rate limiting)."""
        self._order_times.append(datetime.now())
        self._daily_order_count += 1

    def _is_positions_cache_fresh(self) -> bool:
        """Check if positions cache is still fresh."""
        if self._positions_cache_time is None:
            return False

        age = (datetime.now() - self._positions_cache_time).total_seconds()
        return age < self.config.positions_cache_ttl

    # ==================== Cleanup ====================

    def __del__(self) -> None:
        """Cleanup when broker is destroyed."""
        self.stop_reconciliation()

    def close(self) -> None:
        """Explicitly close broker and cleanup resources."""
        logger.info("Closing KiteBroker...")
        self.stop_reconciliation()
        logger.info("KiteBroker closed")
