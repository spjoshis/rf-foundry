#!/usr/bin/env python3
"""
Quick test script to validate paper trading deployment setup.

This script tests the core components without running a full trading loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from tradebox.agents import PPOAgent
from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.execution import PaperBroker, OrderSide
from tradebox.features.technical import FeatureConfig, TechnicalFeatures


def test_model_loading(model_path: str):
    """Test model loading."""
    logger.info(f"Testing model loading: {model_path}")
    try:
        agent = PPOAgent.load(model_path)
        logger.success("✓ Model loaded successfully")
        return agent
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return None


def test_paper_broker(initial_capital: float = 100000):
    """Test paper broker initialization."""
    logger.info(f"Testing paper broker with ₹{initial_capital:,.0f}")
    try:
        broker = PaperBroker(initial_capital=initial_capital)
        portfolio = broker.get_portfolio()
        assert portfolio.cash == initial_capital
        assert portfolio.total_value == initial_capital
        logger.success(f"✓ Paper broker initialized (Value: ₹{portfolio.total_value:,.0f})")
        return broker
    except Exception as e:
        logger.error(f"✗ Paper broker test failed: {e}")
        return None


def test_data_fetch(symbol: str = "^NSEI", interval: str = "5m"):
    """Test market data fetching."""
    logger.info(f"Testing data fetch for {symbol} ({interval})")
    try:
        loader = YahooDataLoader(cache_dir="cache", use_cache=False)
        data = loader.download_intraday(symbol=symbol, period="5d", interval=interval)
        logger.success(f"✓ Fetched {len(data)} bars for {symbol}")
        return data
    except Exception as e:
        logger.error(f"✗ Data fetch failed: {e}")
        return None


def test_feature_extraction(data):
    """Test feature extraction."""
    logger.info("Testing feature extraction")
    try:
        extractor = TechnicalFeatures(FeatureConfig(timeframe="intraday"))
        features = extractor.extract(data, fit_normalize=True)
        logger.success(f"✓ Extracted {len(features.columns)} features")
        return features
    except Exception as e:
        logger.error(f"✗ Feature extraction failed: {e}")
        return None


def test_agent_inference(agent, data, features):
    """Test agent inference."""
    logger.info("Testing agent inference")
    try:
        import numpy as np

        # Extract observation directly (same as deployment script)
        numeric_features = features.select_dtypes(include=[np.number])

        # Get lookback window (60 bars)
        lookback = 60
        if len(numeric_features) < lookback:
            # Pad with zeros if needed
            padding = np.zeros((lookback - len(numeric_features), len(numeric_features.columns)))
            feature_window = np.vstack([padding, numeric_features.values])
        else:
            feature_window = numeric_features.tail(lookback).values

        # Flatten technical features
        technical_flat = feature_window.flatten()

        # Create portfolio state
        portfolio_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        # Combine observation
        obs = np.concatenate([technical_flat, portfolio_state]).astype(np.float32)

        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)
        actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
        logger.success(f"✓ Agent predicted action: {actions.get(int(action), action)}")
        return action
    except Exception as e:
        logger.error(f"✗ Agent inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_paper_trading(broker, symbol: str = "^NSEI"):
    """Test paper trading operations."""
    logger.info("Testing paper trading operations")
    try:
        # Calculate affordable quantity (use 20% of capital)
        portfolio = broker.get_portfolio()
        # Get current price by simulating a check
        from tradebox.data.loaders.yahoo_loader import YahooDataLoader
        loader = YahooDataLoader(cache_dir="cache", use_cache=False)
        data = loader.download_intraday(symbol, period="1d", interval="5m")
        current_price = data.iloc[-1]["Close"]

        max_quantity = int((portfolio.total_value * 0.2) / current_price)
        quantity = max(1, max_quantity)  # At least 1 share

        logger.info(f"Testing with {quantity} shares @ ₹{current_price:.2f}")

        # Test buy market order (price=None for immediate execution)
        order = broker.place_order(symbol, OrderSide.BUY, quantity, price=None)
        logger.success(f"✓ Buy order placed: {order.order_id} (Status: {order.status.value})")

        # Check portfolio
        portfolio = broker.get_portfolio()
        assert symbol in broker.positions, f"Position not created for {symbol}"
        logger.success(f"✓ Position created: {broker.positions[symbol].quantity} shares")

        # Test sell market order
        actual_quantity = broker.positions[symbol].quantity
        sell_order = broker.place_order(symbol, OrderSide.SELL, actual_quantity, price=None)
        logger.success(f"✓ Sell order placed: {sell_order.order_id} (Status: {sell_order.status.value})")

        # Check portfolio closed
        portfolio = broker.get_portfolio()
        assert symbol not in broker.positions, f"Position not closed for {symbol}"
        logger.success("✓ Position closed successfully")

        return True
    except Exception as e:
        logger.error(f"✗ Paper trading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<level>{level: <8}</level> | {message}",
    )

    logger.info("=" * 70)
    logger.info("Paper Trading Deployment Validation")
    logger.info("=" * 70)
    logger.info("")

    # Test 1: Model loading
    model_path = "models/exp005_21_per_return_intra/best/best_model.zip"
    agent = test_model_loading(model_path)
    if not agent:
        logger.error("Cannot proceed without model")
        return 1

    logger.info("")

    # Test 2: Paper broker
    broker = test_paper_broker()
    if not broker:
        return 1

    logger.info("")

    # Test 3: Data fetch
    data = test_data_fetch()
    if data is None:
        return 1

    logger.info("")

    # Test 4: Feature extraction
    features = test_feature_extraction(data)
    if features is None:
        return 1

    logger.info("")

    # Test 5: Agent inference
    action = test_agent_inference(agent, data, features)
    if action is None:
        return 1

    logger.info("")

    # Test 6: Paper trading
    success = test_paper_trading(broker)
    if not success:
        return 1

    logger.info("")
    logger.info("=" * 70)
    logger.success("All tests passed! ✓")
    logger.info("=" * 70)
    logger.info("")
    logger.info("You can now deploy to paper trading:")
    logger.info("  poetry run python scripts/deploy_paper_trading.py \\")
    logger.info("      --model models/exp005_21_per_return_intra/best/best_model.zip \\")
    logger.info("      --symbols ^NSEI \\")
    logger.info("      --interval 5m \\")
    logger.info("      --verbose")

    return 0


if __name__ == "__main__":
    sys.exit(main())
