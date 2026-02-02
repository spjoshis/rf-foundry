"""Trading workflow orchestration."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from stable_baselines3 import PPO

from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.execution.base_broker import BaseBroker, OrderSide, Portfolio, Order
from tradebox.execution.paper_broker import PaperBroker
from tradebox.features.technical import TechnicalFeatures, FeatureConfig
from tradebox.monitoring import MetricsCollector, MetricsStore
from tradebox.monitoring.events import (
    PortfolioMetricsEvent,
    TradeMetricsEvent,
    ModelMetricsEvent,
    SystemMetricsEvent,
)
from tradebox.orchestration.config import OrchestrationConfig
from tradebox.orchestration.exceptions import WorkflowError
from tradebox.orchestration.state import StateManager
from tradebox.risk import RiskManager, RiskConfig


class TradingWorkflow:
    """
    Orchestrates the complete trading pipeline.

    Workflow stages:
    1. Load trained model
    2. Fetch latest market data for each symbol
    3. Extract technical features
    4. Generate prediction (action) from RL agent
    5. Validate action through risk managers
    6. Execute order via broker
    7. Update state and emit metrics

    Example:
        >>> from tradebox.orchestration import TradingWorkflow, OrchestrationConfig
        >>> config = OrchestrationConfig.from_yaml("configs/orchestration/paper_eod.yaml")
        >>> workflow = TradingWorkflow(config)
        >>> workflow.execute()  # Run once
    """

    def __init__(
        self,
        config: OrchestrationConfig,
        broker: Optional[BaseBroker] = None,
        risk_manager: Optional[RiskManager] = None,
        state_manager: Optional[StateManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize trading workflow.

        Args:
            config: Orchestration configuration
            broker: Broker instance (creates PaperBroker if None)
            risk_manager: Risk manager (creates default if None)
            state_manager: State manager (creates default if None)
            metrics_collector: Metrics collector (creates default if None)
        """
        self.config = config

        # Initialize broker
        if broker is None:
            if config.broker == "paper":
                broker = PaperBroker(initial_capital=config.initial_capital)
                logger.info(
                    f"Initialized PaperBroker with ₹{config.initial_capital:,.0f}"
                )
            else:
                # TODO: Initialize KiteBroker when live trading
                raise NotImplementedError("Live Kite broker not yet implemented")

        self.broker = broker

        # Initialize risk manager
        if risk_manager is None:
            risk_config = RiskConfig(
                max_position_size_pct=config.max_position_size_pct,
                max_stock_allocation_pct=config.max_stock_allocation_pct,
                max_daily_loss_pct=config.max_daily_loss_pct,
                max_sector_concentration_pct=config.max_sector_concentration_pct,
                min_daily_volume=config.min_daily_volume,
                max_leverage=config.max_leverage,
            )
            risk_manager = RiskManager(risk_config)

        self.risk_manager = risk_manager

        # Initialize state manager
        if state_manager is None:
            state_manager = StateManager(config.state_file)

        self.state_manager = state_manager

        # Initialize metrics collector
        if metrics_collector is None:
            metrics_store = MetricsStore(config.metrics_db)
            metrics_collector = MetricsCollector(metrics_store)

        self.metrics_collector = metrics_collector

        # Initialize data loader
        self.data_loader = YahooDataLoader()

        # Initialize feature extractor
        self.feature_config = FeatureConfig()
        self.feature_extractor = TechnicalFeatures(self.feature_config)

        # Agent will be loaded on demand
        self.agent: Optional[PPO] = None

        logger.info(
            f"TradingWorkflow initialized: mode={config.mode}, "
            f"broker={config.broker}, symbols={len(config.symbols)}"
        )

    def _load_model(self) -> PPO:
        """
        Load trained RL model.

        Returns:
            Loaded PPO agent

        Raises:
            WorkflowError: If model loading fails
        """
        if self.agent is not None:
            return self.agent

        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise WorkflowError(f"Model not found: {model_path}")

        try:
            logger.info(f"Loading model from {model_path}")
            self.agent = PPO.load(str(model_path))
            logger.info("Model loaded successfully")
            return self.agent

        except Exception as e:
            raise WorkflowError(f"Failed to load model: {e}")

    def _fetch_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch latest market data for symbol.

        Args:
            symbol: Stock symbol
            days: Number of days of history to fetch

        Returns:
            DataFrame with OHLCV data

        Raises:
            WorkflowError: If data fetching fails
        """
        try:
            logger.debug(f"Fetching data for {symbol} ({days} days)")
            df = self.data_loader.load_symbol(
                symbol=symbol,
                period=f"{days}d",
                interval="1d",
            )

            if df.empty:
                raise WorkflowError(f"No data retrieved for {symbol}")

            logger.debug(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            raise WorkflowError(f"Data fetch failed for {symbol}: {e}")

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical features from price data.

        Args:
            data: Raw OHLCV data

        Returns:
            DataFrame with technical features

        Raises:
            WorkflowError: If feature extraction fails
        """
        try:
            logger.debug("Extracting technical features")
            features = self.feature_extractor.extract(data)

            if features.empty:
                raise WorkflowError("Feature extraction returned empty DataFrame")

            logger.debug(f"Extracted {len(features.columns)} features")
            return features

        except Exception as e:
            raise WorkflowError(f"Feature extraction failed: {e}")

    def _predict_action(self, features: pd.DataFrame, symbol: str) -> int:
        """
        Generate trading action from RL agent.

        Args:
            features: Feature DataFrame
            symbol: Stock symbol

        Returns:
            Action (0=hold, 1=buy, 2=sell)

        Raises:
            WorkflowError: If prediction fails
        """
        try:
            # Get latest observation (last row)
            latest = features.iloc[-1]

            # Prepare observation (exclude non-feature columns)
            exclude_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            obs_cols = [col for col in features.columns if col not in exclude_cols]
            observation = latest[obs_cols].values

            # Predict action
            action, _states = self.agent.predict(observation, deterministic=True)
            action = int(action)

            logger.info(f"Predicted action for {symbol}: {action} (0=hold, 1=buy, 2=sell)")

            # Emit model metrics
            self._emit_model_metrics(symbol, action, observation)

            return action

        except Exception as e:
            raise WorkflowError(f"Prediction failed for {symbol}: {e}")

    def _validate_action(
        self,
        symbol: str,
        action: int,
        current_price: float,
        portfolio: Portfolio,
    ) -> bool:
        """
        Validate trading action through risk manager.

        Args:
            symbol: Stock symbol
            action: Trading action (0=hold, 1=buy, 2=sell)
            current_price: Current market price
            portfolio: Current portfolio state

        Returns:
            True if action is valid, False otherwise
        """
        # Hold action always valid
        if action == 0:
            return True

        # Determine order side and quantity
        if action == 1:  # Buy
            side = OrderSide.BUY
            # Calculate quantity based on position sizing
            position_value = portfolio.total_value * 0.10  # 10% of portfolio
            quantity = int(position_value / current_price)

            if quantity == 0:
                logger.warning(f"Calculated quantity is 0 for {symbol}, skipping")
                return False

        elif action == 2:  # Sell
            # Check if we have position to sell
            position = self.broker.get_position(symbol)
            if position is None or position.quantity == 0:
                logger.warning(f"No position to sell for {symbol}")
                return False

            side = OrderSide.SELL
            quantity = position.quantity

        else:
            logger.warning(f"Invalid action: {action}")
            return False

        # Create temporary order for validation
        from tradebox.execution.base_broker import Order

        order = Order(
            order_id=f"temp_{symbol}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=current_price,
        )

        # Validate through risk manager
        valid, reason = self.risk_manager.validate_order(order, portfolio)

        if not valid:
            logger.warning(f"Action rejected by risk manager: {reason}")
            return False

        logger.info(f"Action validated: {side.value} {quantity} {symbol} @ ₹{current_price:.2f}")
        return True

    def _execute_action(
        self,
        symbol: str,
        action: int,
        current_price: float,
    ) -> None:
        """
        Execute trading action via broker.

        Args:
            symbol: Stock symbol
            action: Trading action (0=hold, 1=buy, 2=sell)
            current_price: Current market price

        Raises:
            WorkflowError: If execution fails
        """
        if action == 0:
            logger.debug(f"Holding position for {symbol}")
            return

        try:
            start_time = datetime.now()

            if action == 1:  # Buy
                portfolio = self.broker.get_portfolio()
                position_value = portfolio.total_value * 0.10
                quantity = int(position_value / current_price)

                order = self.broker.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    price=None,  # Market order
                )

                # Calculate latency
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                logger.info(
                    f"✅ BUY executed: {order.filled_quantity} {symbol} @ "
                    f"₹{order.filled_price:.2f} (Order ID: {order.order_id})"
                )

                # Emit trade metrics
                self._emit_trade_metrics(order, current_price, latency_ms)

                # Increment trade count
                self.state_manager.increment_trade_count()

            elif action == 2:  # Sell
                position = self.broker.get_position(symbol)
                if position is None:
                    logger.warning(f"No position to sell for {symbol}")
                    return

                order = self.broker.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    price=None,  # Market order
                )

                # Calculate latency
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                logger.info(
                    f"✅ SELL executed: {order.filled_quantity} {symbol} @ "
                    f"₹{order.filled_price:.2f} (Order ID: {order.order_id})"
                )

                # Emit trade metrics
                self._emit_trade_metrics(order, current_price, latency_ms)

                # Increment trade count
                self.state_manager.increment_trade_count()

        except Exception as e:
            # Emit error metric
            self._emit_system_error("broker", str(e))
            raise WorkflowError(f"Execution failed for {symbol}: {e}")

    def _process_symbol(self, symbol: str, agent: PPO) -> None:
        """
        Process trading for a single symbol.

        This executes the complete workflow for one symbol:
        data → features → prediction → validation → execution

        Args:
            symbol: Stock symbol
            agent: Trained RL agent

        Raises:
            WorkflowError: If processing fails
        """
        logger.info(f"Processing symbol: {symbol}")

        # 1. Fetch data
        data = self._fetch_data(symbol)

        # 2. Extract features
        features = self._extract_features(data)

        # 3. Get current price
        current_price = float(data["Close"].iloc[-1])
        logger.info(f"Current price for {symbol}: ₹{current_price:.2f}")

        # 4. Predict action
        action = self._predict_action(features, symbol)

        # 5. Get portfolio state
        portfolio = self.broker.get_portfolio()

        # 6. Validate action
        if not self._validate_action(symbol, action, current_price, portfolio):
            logger.info(f"Action not validated for {symbol}, skipping execution")
            return

        # 7. Execute action
        self._execute_action(symbol, action, current_price)

    def _emit_portfolio_metrics(self, portfolio: Portfolio) -> None:
        """
        Emit portfolio metrics event.

        Args:
            portfolio: Current portfolio state
        """
        try:
            event = PortfolioMetricsEvent(
                timestamp=datetime.now(),
                total_value=portfolio.total_value,
                cash=portfolio.cash,
                positions_value=sum(p.market_value for p in portfolio.positions.values()),
                unrealized_pnl=sum(p.unrealized_pnl for p in portfolio.positions.values()),
                realized_pnl=sum(p.realized_pnl for p in portfolio.positions.values()),
            )
            self.metrics_collector.record_portfolio_snapshot(event)
        except Exception as e:
            logger.error(f"Failed to emit portfolio metrics: {e}")

    def _emit_trade_metrics(self, order: Order, intended_price: float, latency_ms: float) -> None:
        """
        Emit trade execution metrics event.

        Args:
            order: Executed order
            intended_price: Intended execution price
            latency_ms: Execution latency in milliseconds
        """
        try:
            # Calculate slippage
            slippage_pct = ((order.filled_price - intended_price) / intended_price) * 100

            event = TradeMetricsEvent(
                trade_id=order.order_id,
                timestamp=order.timestamp,
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.filled_quantity,
                intended_price=intended_price,
                filled_price=order.filled_price,
                slippage_pct=slippage_pct,
                commission=order.commission,
                latency_ms=latency_ms,
                order_status=order.status.value,
            )
            self.metrics_collector.record_trade(event)
        except Exception as e:
            logger.error(f"Failed to emit trade metrics: {e}")

    def _emit_model_metrics(self, symbol: str, action: int, observation) -> None:
        """
        Emit model prediction metrics event.

        Args:
            symbol: Stock symbol
            action: Predicted action
            observation: Feature observation (optional)
        """
        try:
            event = ModelMetricsEvent(
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                confidence=None,  # TODO: Extract confidence from model
                observation=None,  # Skip serializing for now
            )
            self.metrics_collector.record_model_prediction(event)
        except Exception as e:
            logger.error(f"Failed to emit model metrics: {e}")

    def _emit_system_error(self, component: str, error_message: str) -> None:
        """
        Emit system error metric.

        Args:
            component: Component name (data/agent/broker/risk)
            error_message: Error message
        """
        try:
            event = SystemMetricsEvent(
                timestamp=datetime.now(),
                metric_name="error",
                metric_value=1.0,
                metric_type="error",
                component=component,
                message=error_message,
            )
            self.metrics_collector.record_system_metric(event)
        except Exception as e:
            logger.error(f"Failed to emit system error metric: {e}")

    def _generate_report(self) -> None:
        """Generate end-of-day report."""
        logger.info("=" * 70)
        logger.info("END-OF-DAY REPORT")
        logger.info("=" * 70)

        # Get portfolio
        portfolio = self.broker.get_portfolio()
        portfolio.update_total_value()

        # Emit portfolio metrics
        self._emit_portfolio_metrics(portfolio)

        logger.info(f"Portfolio Value: ₹{portfolio.total_value:,.2f}")
        logger.info(f"Cash: ₹{portfolio.cash:,.2f}")
        logger.info(f"Positions Value: ₹{sum(p.market_value for p in portfolio.positions.values()):,.2f}")

        # Active positions
        if portfolio.positions:
            logger.info("\nActive Positions:")
            for symbol, pos in portfolio.positions.items():
                logger.info(
                    f"  {symbol}: {pos.quantity} @ ₹{pos.avg_price:.2f} "
                    f"(Current: ₹{pos.current_price:.2f}, "
                    f"P&L: ₹{pos.unrealized_pnl:+,.2f})"
                )
        else:
            logger.info("\nNo active positions")

        # State
        state = self.state_manager.get_state()
        logger.info(f"\nTrades Today: {state.total_trades_today}")
        logger.info(f"Daily P&L: ₹{state.daily_pnl:+,.2f}")

        if state.circuit_breaker_active:
            logger.warning("⚠️ CIRCUIT BREAKER ACTIVE")

        logger.info("=" * 70)

    def execute(self) -> None:
        """
        Execute complete trading workflow.

        This is the main entry point that orchestrates all stages:
        1. Check circuit breaker
        2. Load model
        3. Process each symbol
        4. Generate report
        5. Update state

        Raises:
            WorkflowError: If workflow execution fails
        """
        logger.info("=" * 70)
        logger.info("TRADING WORKFLOW EXECUTION")
        logger.info("=" * 70)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Mode: {self.config.mode}")
        logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        logger.info("=" * 70)

        try:
            # 1. Check circuit breaker
            state = self.state_manager.get_state()
            if state.circuit_breaker_active:
                logger.error("⚠️ Circuit breaker is ACTIVE. Trading halted.")
                logger.error("Reset circuit breaker manually to resume trading.")
                return

            # 2. Load model
            agent = self._load_model()

            # 3. Reset daily tracking if new day
            portfolio = self.broker.get_portfolio()
            portfolio.update_total_value()

            today = datetime.now().date()
            last_run = state.last_run_datetime
            if last_run is None or last_run.date() != today:
                logger.info("New trading day, resetting daily tracking")
                self.state_manager.reset_daily(portfolio.total_value)
                self.risk_manager.reset_daily(portfolio.total_value)

            # 4. Process each symbol
            for symbol in self.config.symbols:
                try:
                    self._process_symbol(symbol, agent)
                except WorkflowError as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    self.state_manager.record_error(f"{symbol}: {e}")
                    # Continue with next symbol

            # 5. Update state
            self.state_manager.update_last_run()

            # 6. Generate report
            self._generate_report()

            logger.info("✅ Workflow execution completed successfully")

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.state_manager.record_error(str(e))
            raise WorkflowError(f"Workflow execution failed: {e}")
