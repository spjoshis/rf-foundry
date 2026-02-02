"""
Quick Validation Script for CNN Implementation.

This script performs rapid sanity checks to verify CNN integration
works correctly before running full training experiments.

Usage:
    python scripts/quick_validate_cnn.py
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradebox.agents.config import PPOConfig
from tradebox.agents.ppo_agent import PPOAgent
from tradebox.data.loaders.yahoo_loader import YahooDataLoader
from tradebox.env.trading_env import EnvConfig, TradingEnv
from tradebox.features.extractor import FeatureExtractor, FeatureExtractorConfig


def test_environment_dict_observations():
    """Test that environment returns Dict observations correctly."""
    logger.info("=" * 60)
    logger.info("Test 1: Environment with Dict Observations")
    logger.info("=" * 60)

    try:
        # Load minimal data
        loader = YahooDataLoader(Path("data/eod"))
        logger.info("Loading RELIANCE.NS data (2020-2021, ~500 bars)...")
        data = loader.download("RELIANCE.NS", "2020-01-01", "2021-12-31")[:200]
        logger.info(f"  Loaded {len(data)} bars")

        # Extract features (technical only for quick test)
        logger.info("Extracting features...")
        config = FeatureExtractorConfig(fundamental={"enabled": False})
        extractor = FeatureExtractor(config)
        features = extractor.extract("RELIANCE", data, fit_normalize=True)
        logger.info(f"  Extracted {len(features.columns)} features")

        # Create env
        logger.info("Creating TradingEnv with Dict observation space...")
        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(data, features, config)

        # Verify observation space type
        from gymnasium import spaces

        assert isinstance(
            env.observation_space, spaces.Dict
        ), f"Expected Dict space, got {type(env.observation_space)}"
        logger.info(f"  ✓ Observation space is Dict")

        # Reset and get observation
        logger.info("Resetting environment...")
        obs, info = env.reset()

        # Verify observation structure
        assert isinstance(obs, dict), f"Expected dict observation, got {type(obs)}"
        logger.info(f"  ✓ Observation is dict with keys: {list(obs.keys())}")

        # Check required keys
        required_keys = {"price", "indicators", "portfolio"}
        assert required_keys.issubset(
            obs.keys()
        ), f"Missing required keys: {required_keys - obs.keys()}"
        logger.info(f"  ✓ All required keys present")

        # Check shapes
        logger.info(f"  Observation shapes:")
        logger.info(f"    - price: {obs['price'].shape} (expected: (60, 5))")
        logger.info(f"    - indicators: {obs['indicators'].shape}")
        logger.info(f"    - portfolio: {obs['portfolio'].shape} (expected: (4,))")
        if "fundamentals" in obs:
            logger.info(
                f"    - fundamentals: {obs['fundamentals'].shape} (optional, EOD only)"
            )

        # Verify price shape
        assert obs["price"].shape == (
            60,
            5,
        ), f"Price shape mismatch: {obs['price'].shape}"
        assert obs["portfolio"].shape == (
            4,
        ), f"Portfolio shape mismatch: {obs['portfolio'].shape}"
        logger.info(f"  ✓ Shapes are correct")

        # Verify data types
        assert obs["price"].dtype == np.float32, f"Price dtype: {obs['price'].dtype}"
        assert (
            obs["indicators"].dtype == np.float32
        ), f"Indicators dtype: {obs['indicators'].dtype}"
        logger.info(f"  ✓ Data types are float32")

        # Verify no NaN values
        assert not np.isnan(
            obs["price"]
        ).any(), "Price contains NaN values"
        assert not np.isnan(obs["indicators"]).any(), "Indicators contain NaN values"
        assert not np.isnan(
            obs["portfolio"]
        ).any(), "Portfolio contains NaN values"
        logger.info(f"  ✓ No NaN values in observations")

        # Test step
        logger.info("Testing step function...")
        action = 1  # Buy
        obs2, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs2, dict), "Step should return dict observation"
        logger.info(f"  ✓ Step works: action={action}, reward={reward:.4f}")

        logger.info("✅ Test 1 PASSED: Environment Dict observations work correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cnn_agent_creation():
    """Test that CNN agent can be created and used."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 2: CNN Agent Creation and Inference")
    logger.info("=" * 60)

    try:
        # Load minimal data (reuse from test 1)
        loader = YahooDataLoader(Path("data/eod"))
        logger.info("Loading RELIANCE.NS data...")
        data = loader.download("RELIANCE.NS", "2020-01-01", "2021-12-31")[:200]

        feature_config = FeatureExtractorConfig(fundamental={"enabled": False})
        extractor = FeatureExtractor(feature_config)
        features = extractor.extract("RELIANCE", data, fit_normalize=True)

        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(data, features, config)

        # Create CNN agent
        logger.info("Creating PPOAgent with CNN extractor...")
        ppo_config = PPOConfig(
            use_cnn_extractor=True, extractor_type="trading", use_attention=True
        )
        agent = PPOAgent(env, ppo_config)
        logger.info(f"  ✓ Agent created")

        # Verify policy type
        policy_type = type(agent.agent.policy).__name__
        logger.info(f"  Policy type: {policy_type}")
        assert "MultiInput" in policy_type, f"Expected MultiInputPolicy, got {policy_type}"
        logger.info(f"  ✓ Using MultiInputPolicy (correct for Dict obs)")

        # Verify features extractor
        extractor_type = type(agent.agent.policy.features_extractor).__name__
        logger.info(f"  Features extractor: {extractor_type}")
        assert (
            extractor_type == "TradingCNNExtractor"
        ), f"Expected TradingCNNExtractor, got {extractor_type}"
        logger.info(f"  ✓ Using TradingCNNExtractor")

        # Test prediction
        logger.info("Testing prediction...")
        obs, info = env.reset()
        action, _states = agent.predict(obs, deterministic=True)
        logger.info(f"  ✓ Prediction works: action={action} ({['Hold', 'Buy', 'Sell'][action]})")

        # Test multiple steps
        logger.info("Testing multiple steps...")
        total_reward = 0
        for step in range(10):
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                logger.info(f"    Episode ended at step {step+1}")
                obs, info = env.reset()
        logger.info(f"  ✓ Multiple steps work: total_reward={total_reward:.4f}")

        logger.info("✅ Test 2 PASSED: CNN Agent creation and inference work correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Test 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mlp_agent_creation():
    """Test that MLP baseline agent can be created (with CNN disabled)."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test 3: MLP Baseline Agent (CNN Disabled)")
    logger.info("=" * 60)

    try:
        # Load minimal data
        loader = YahooDataLoader(Path("data/eod"))
        logger.info("Loading RELIANCE.NS data...")
        data = loader.download("RELIANCE.NS", "2020-01-01", "2021-12-31")[:200]

        feature_config = FeatureExtractorConfig(fundamental={"enabled": False})
        extractor = FeatureExtractor(feature_config)
        features = extractor.extract("RELIANCE", data, fit_normalize=True)

        config = EnvConfig(lookback_window=60, max_episode_steps=100)
        env = TradingEnv(data, features, config)

        # Create MLP agent (CNN disabled)
        logger.info("Creating PPOAgent with CNN DISABLED (MLP baseline)...")
        ppo_config = PPOConfig(use_cnn_extractor=False)  # Explicit: use MLP
        agent = PPOAgent(env, ppo_config)
        logger.info(f"  ✓ Agent created")

        # Verify policy type (should still be MultiInput for Dict obs)
        policy_type = type(agent.agent.policy).__name__
        logger.info(f"  Policy type: {policy_type}")

        # Verify features extractor (should NOT be CNN)
        extractor_type = type(agent.agent.policy.features_extractor).__name__
        logger.info(f"  Features extractor: {extractor_type}")
        assert (
            extractor_type != "TradingCNNExtractor"
        ), f"Should not use TradingCNNExtractor when use_cnn_extractor=False"
        logger.info(f"  ✓ NOT using TradingCNNExtractor (correct for MLP baseline)")

        # Test prediction
        logger.info("Testing prediction...")
        obs, info = env.reset()
        action, _states = agent.predict(obs, deterministic=True)
        logger.info(f"  ✓ Prediction works: action={action} ({['Hold', 'Buy', 'Sell'][action]})")

        logger.info("✅ Test 3 PASSED: MLP baseline agent works correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  CNN Implementation Quick Validation  ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")

    results = []

    # Test 1: Environment with Dict observations
    results.append(("Environment Dict Observations", test_environment_dict_observations()))

    # Test 2: CNN Agent creation and inference
    results.append(("CNN Agent Creation", test_cnn_agent_creation()))

    # Test 3: MLP Agent creation (CNN disabled)
    results.append(("MLP Baseline Agent", test_mlp_agent_creation()))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    logger.info("=" * 60)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED - Ready to start training experiments!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Train MLP baseline:")
        logger.info("     python scripts/train.py --config configs/experiments/exp_mlp_baseline_quick.yaml")
        logger.info("  2. Train CNN model:")
        logger.info("     python scripts/train.py --config configs/experiments/exp_cnn_eod_quick.yaml")
        logger.info("  3. Compare results in TensorBoard:")
        logger.info("     tensorboard --logdir logs/tensorboard")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED - Please fix issues before training")
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Check that all files exist:")
        logger.error("     ls src/tradebox/models/trading_cnn_extractor.py")
        logger.error("  2. Verify imports:")
        logger.error("     python -c 'from tradebox.models import TradingCNNExtractor'")
        logger.error("  3. Check observation space:")
        logger.error("     python -c 'from tradebox.env import TradingEnv; print(TradingEnv.__doc__)'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
