"""Unit tests for regime-conditioned action masking."""

import numpy as np
import pytest

from tradebox.env.action_mask import ActionMaskConfig, RegimeActionMask


class TestActionMaskConfig:
    """Test suite for ActionMaskConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ActionMaskConfig()

        assert config.enabled is False
        assert config.regime_column == "regime_state"
        assert config.trend_bias_column == "trend_bias"
        assert config.ranging_state == 0
        assert config.transition_state == 1
        assert config.trending_state == 2
        assert config.allow_hold_always is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ActionMaskConfig(
            enabled=True,
            regime_column="custom_regime",
            trend_bias_column="custom_bias",
            ranging_state=10,
            transition_state=20,
            trending_state=30,
            allow_hold_always=False,
        )

        assert config.enabled is True
        assert config.regime_column == "custom_regime"
        assert config.trend_bias_column == "custom_bias"
        assert config.ranging_state == 10
        assert config.transition_state == 20
        assert config.trending_state == 30
        assert config.allow_hold_always is False

    def test_validation_ranging_equals_trending(self):
        """Test validation rejects when ranging_state equals trending_state."""
        with pytest.raises(ValueError, match="ranging_state.*trending_state.*must be different"):
            ActionMaskConfig(ranging_state=0, trending_state=0)

    def test_validation_ranging_equals_transition(self):
        """Test validation rejects when ranging_state equals transition_state."""
        with pytest.raises(ValueError, match="ranging_state.*transition_state.*must be different"):
            ActionMaskConfig(ranging_state=1, transition_state=1)

    def test_validation_trending_equals_transition(self):
        """Test validation rejects when trending_state equals transition_state."""
        with pytest.raises(ValueError, match="trending_state.*transition_state.*must be different"):
            ActionMaskConfig(transition_state=2, trending_state=2)


class TestRegimeActionMask:
    """Test suite for RegimeActionMask class."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        masker = RegimeActionMask()
        assert masker.config is not None
        assert masker.config.enabled is False

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)
        assert masker.config.enabled is True

    def test_mask_disabled_allows_all_actions(self):
        """Test that masking disabled allows all actions."""
        config = ActionMaskConfig(enabled=False)
        masker = RegimeActionMask(config)

        # Test all regime/bias combinations
        for regime in [0, 1, 2]:
            for bias in [-1, 0, 1]:
                mask = masker.get_mask(regime, bias)
                assert mask.shape == (3,)
                assert mask.dtype == bool
                np.testing.assert_array_equal(mask, [True, True, True])

    def test_mask_ranging_regime(self):
        """Test mask for ranging regime (only Hold allowed)."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Ranging regime, any bias
        for bias in [-1, 0, 1]:
            mask = masker.get_mask(regime_state=0, trend_bias=bias)
            expected = np.array([True, False, False])  # Only Hold
            np.testing.assert_array_equal(mask, expected)

    def test_mask_transition_regime(self):
        """Test mask for transition regime (all actions allowed)."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Transition regime, any bias
        for bias in [-1, 0, 1]:
            mask = masker.get_mask(regime_state=1, trend_bias=bias)
            expected = np.array([True, True, True])  # All actions
            np.testing.assert_array_equal(mask, expected)

    def test_mask_trending_uptrend(self):
        """Test mask for trending regime with uptrend (Hold + Buy)."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        mask = masker.get_mask(regime_state=2, trend_bias=1)
        expected = np.array([True, True, False])  # Hold + Buy
        np.testing.assert_array_equal(mask, expected)

    def test_mask_trending_downtrend(self):
        """Test mask for trending regime with downtrend (Hold + Sell)."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        mask = masker.get_mask(regime_state=2, trend_bias=-1)
        expected = np.array([True, False, True])  # Hold + Sell
        np.testing.assert_array_equal(mask, expected)

    def test_mask_trending_neutral(self):
        """Test mask for trending regime with neutral bias (only Hold)."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        mask = masker.get_mask(regime_state=2, trend_bias=0)
        expected = np.array([True, False, False])  # Only Hold
        np.testing.assert_array_equal(mask, expected)

    def test_mask_invalid_regime_state(self):
        """Test that invalid regime_state raises ValueError."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        with pytest.raises(ValueError, match="Invalid regime_state"):
            masker.get_mask(regime_state=999, trend_bias=0)

    def test_mask_invalid_trend_bias(self):
        """Test that invalid trend_bias raises ValueError."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        with pytest.raises(ValueError, match="Invalid trend_bias"):
            masker.get_mask(regime_state=0, trend_bias=5)

    def test_mask_custom_regime_states(self):
        """Test masking with custom regime state values."""
        config = ActionMaskConfig(
            enabled=True,
            ranging_state=10,
            transition_state=20,
            trending_state=30,
        )
        masker = RegimeActionMask(config)

        # Ranging (custom state 10)
        mask = masker.get_mask(regime_state=10, trend_bias=0)
        np.testing.assert_array_equal(mask, [True, False, False])

        # Transition (custom state 20)
        mask = masker.get_mask(regime_state=20, trend_bias=0)
        np.testing.assert_array_equal(mask, [True, True, True])

        # Trending uptrend (custom state 30)
        mask = masker.get_mask(regime_state=30, trend_bias=1)
        np.testing.assert_array_equal(mask, [True, True, False])

    def test_is_action_valid_hold(self):
        """Test is_action_valid for Hold action."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Hold should be valid in all regimes
        assert masker.is_action_valid(0, regime_state=0, trend_bias=0) is True
        assert masker.is_action_valid(0, regime_state=1, trend_bias=0) is True
        assert masker.is_action_valid(0, regime_state=2, trend_bias=1) is True

    def test_is_action_valid_buy(self):
        """Test is_action_valid for Buy action."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Buy should be invalid in ranging
        assert masker.is_action_valid(1, regime_state=0, trend_bias=0) is False

        # Buy should be valid in transition
        assert masker.is_action_valid(1, regime_state=1, trend_bias=0) is True

        # Buy should be valid in uptrend
        assert masker.is_action_valid(1, regime_state=2, trend_bias=1) is True

        # Buy should be invalid in downtrend
        assert masker.is_action_valid(1, regime_state=2, trend_bias=-1) is False

    def test_is_action_valid_sell(self):
        """Test is_action_valid for Sell action."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Sell should be invalid in ranging
        assert masker.is_action_valid(2, regime_state=0, trend_bias=0) is False

        # Sell should be valid in transition
        assert masker.is_action_valid(2, regime_state=1, trend_bias=0) is True

        # Sell should be invalid in uptrend
        assert masker.is_action_valid(2, regime_state=2, trend_bias=1) is False

        # Sell should be valid in downtrend
        assert masker.is_action_valid(2, regime_state=2, trend_bias=-1) is True

    def test_is_action_valid_invalid_action(self):
        """Test is_action_valid with invalid action."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        with pytest.raises(ValueError, match="Invalid action"):
            masker.is_action_valid(3, regime_state=0, trend_bias=0)

    def test_get_valid_actions_ranging(self):
        """Test get_valid_actions for ranging regime."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        actions = masker.get_valid_actions(regime_state=0, trend_bias=0)
        assert actions == [0]  # Only Hold

    def test_get_valid_actions_transition(self):
        """Test get_valid_actions for transition regime."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        actions = masker.get_valid_actions(regime_state=1, trend_bias=0)
        assert actions == [0, 1, 2]  # All actions

    def test_get_valid_actions_trending_uptrend(self):
        """Test get_valid_actions for trending uptrend."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        actions = masker.get_valid_actions(regime_state=2, trend_bias=1)
        assert actions == [0, 1]  # Hold + Buy

    def test_get_valid_actions_trending_downtrend(self):
        """Test get_valid_actions for trending downtrend."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        actions = masker.get_valid_actions(regime_state=2, trend_bias=-1)
        assert actions == [0, 2]  # Hold + Sell

    def test_get_mask_statistics_empty_sequence(self):
        """Test get_mask_statistics with empty arrays."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        regimes = np.array([])
        biases = np.array([])

        stats = masker.get_mask_statistics(regimes, biases)

        assert stats["hold_allowed_pct"] == 0.0
        assert stats["buy_allowed_pct"] == 0.0
        assert stats["sell_allowed_pct"] == 0.0
        assert stats["all_allowed_pct"] == 0.0
        assert stats["restricted_pct"] == 0.0

    def test_get_mask_statistics_single_timestep(self):
        """Test get_mask_statistics with single timestep."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Ranging regime: only Hold allowed
        regimes = np.array([0])
        biases = np.array([0])

        stats = masker.get_mask_statistics(regimes, biases)

        assert stats["hold_allowed_pct"] == 100.0
        assert stats["buy_allowed_pct"] == 0.0
        assert stats["sell_allowed_pct"] == 0.0
        assert stats["all_allowed_pct"] == 0.0
        assert stats["restricted_pct"] == 100.0

    def test_get_mask_statistics_mixed_regimes(self):
        """Test get_mask_statistics with mixed regime sequence."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # 4 timesteps: ranging, transition, trending up, trending down
        regimes = np.array([0, 1, 2, 2])
        biases = np.array([0, 0, 1, -1])

        stats = masker.get_mask_statistics(regimes, biases)

        # All timesteps allow Hold
        assert stats["hold_allowed_pct"] == 100.0

        # Only transition + trending up allow Buy (2 out of 4)
        assert stats["buy_allowed_pct"] == 50.0

        # Only transition + trending down allow Sell (2 out of 4)
        assert stats["sell_allowed_pct"] == 50.0

        # Only transition allows all actions (1 out of 4)
        assert stats["all_allowed_pct"] == 25.0

        # All except transition are restricted (3 out of 4)
        assert stats["restricted_pct"] == 75.0

    def test_get_mask_statistics_mismatched_lengths(self):
        """Test get_mask_statistics with mismatched array lengths."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        regimes = np.array([0, 1, 2])
        biases = np.array([0, 1])  # Different length

        with pytest.raises(ValueError, match="must have same length"):
            masker.get_mask_statistics(regimes, biases)

    def test_allow_hold_always_false(self):
        """Test masking with allow_hold_always=False."""
        config = ActionMaskConfig(enabled=True, allow_hold_always=False)
        masker = RegimeActionMask(config)

        # In ranging regime with allow_hold_always=False
        # All actions should be masked
        mask = masker.get_mask(regime_state=0, trend_bias=0)
        expected = np.array([False, False, False])
        np.testing.assert_array_equal(mask, expected)

        # In transition, all should still be allowed
        mask = masker.get_mask(regime_state=1, trend_bias=0)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(mask, expected)


class TestActionMaskIntegration:
    """Integration tests for action masking in realistic scenarios."""

    def test_full_regime_cycle(self):
        """Test action masking through a full regime cycle."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Simulate a market going through different regimes
        regimes = [
            0, 0, 0,  # Ranging
            1, 1,     # Transition to trend
            2, 2, 2, 2,  # Trending up
            1,        # Transition
            2, 2,     # Trending down
            1,        # Transition back
            0, 0,     # Ranging again
        ]
        biases = [
            0, 0, 0,    # Neutral
            0, 1,       # Starting to go up
            1, 1, 1, 1, # Strong uptrend
            0,          # Weakening
            -1, -1,     # Strong downtrend
            0,          # Weakening
            0, 0,       # Neutral
        ]

        # Track which actions are valid at each timestep
        valid_actions_history = []
        for regime, bias in zip(regimes, biases):
            valid_actions = masker.get_valid_actions(regime, bias)
            valid_actions_history.append(valid_actions)

        # Verify specific points
        assert valid_actions_history[0] == [0]  # Ranging: only Hold
        assert valid_actions_history[4] == [0, 1, 2]  # Transition: all
        assert valid_actions_history[5] == [0, 1]  # Trending up: Hold + Buy
        assert valid_actions_history[10] == [0, 2]  # Trending down: Hold + Sell
        assert valid_actions_history[-1] == [0]  # Ranging again: only Hold

    def test_statistics_realistic_market(self):
        """Test statistics on realistic market regime distribution."""
        config = ActionMaskConfig(enabled=True)
        masker = RegimeActionMask(config)

        # Simulate 1000 timesteps with realistic regime distribution
        np.random.seed(42)
        # 40% ranging, 20% transition, 40% trending
        regimes = np.random.choice([0, 1, 2], size=1000, p=[0.4, 0.2, 0.4])
        # Bias: 30% down, 40% neutral, 30% up
        biases = np.random.choice([-1, 0, 1], size=1000, p=[0.3, 0.4, 0.3])

        stats = masker.get_mask_statistics(regimes, biases)

        # Hold should always be allowed
        assert stats["hold_allowed_pct"] == 100.0

        # Buy and Sell should be allowed less frequently
        assert stats["buy_allowed_pct"] < 100.0
        assert stats["sell_allowed_pct"] < 100.0

        # All actions allowed should be relatively rare
        assert stats["all_allowed_pct"] < 50.0

        # Most timesteps should have some restriction
        assert stats["restricted_pct"] > 50.0
