"""Configuration loader for KiteBroker."""

import os
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from tradebox.execution.retry import RetryConfig


def load_kite_broker_config(
    config_path: Optional[str] = None,
) -> "KiteBrokerConfig":  # Forward reference
    """
    Load KiteBroker configuration from YAML file and environment.

    API credentials are loaded from environment variables for security:
    - KITE_API_KEY
    - KITE_API_SECRET
    - KITE_ACCESS_TOKEN (optional)

    Args:
        config_path: Path to configuration YAML file
                     (default: configs/execution/kite_broker.yaml)

    Returns:
        KiteBrokerConfig object

    Raises:
        ValueError: If required credentials are missing
        FileNotFoundError: If config file doesn't exist

    Example:
        >>> # Set environment variables first
        >>> os.environ["KITE_API_KEY"] = "your_key"
        >>> os.environ["KITE_API_SECRET"] = "your_secret"
        >>>
        >>> config = load_kite_broker_config()
        >>> print(config.api_key)
    """
    # Default config path
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "configs" / "execution" / "kite_broker.yaml"

    # Load YAML config
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path_obj, "r") as f:
        config_dict = yaml.safe_load(f)

    kite_config = config_dict.get("kite_broker", {})

    # Load API credentials from environment
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    access_token = os.getenv("KITE_ACCESS_TOKEN")  # Optional

    if not api_key:
        raise ValueError(
            "KITE_API_KEY environment variable not set. "
            "Set it with: export KITE_API_KEY='your_key'"
        )
    if not api_secret:
        raise ValueError(
            "KITE_API_SECRET environment variable not set. "
            "Set it with: export KITE_API_SECRET='your_secret'"
        )

    # Build retry config
    retry_dict = kite_config.get("retry", {})
    retry_config = RetryConfig(
        max_retries=retry_dict.get("max_retries", 3),
        initial_delay_seconds=retry_dict.get("initial_delay_seconds", 1.0),
        max_delay_seconds=retry_dict.get("max_delay_seconds", 30.0),
        backoff_multiplier=retry_dict.get("backoff_multiplier", 2.0),
        jitter=retry_dict.get("jitter", True),
    )

    # Build broker config
    reconciliation = kite_config.get("reconciliation", {})
    rate_limits = kite_config.get("rate_limits", {})
    cache = kite_config.get("cache", {})

    # Import here to avoid circular dependency
    from tradebox.execution.kite_broker import KiteBrokerConfig

    broker_config = KiteBrokerConfig(
        api_key=api_key,
        api_secret=api_secret,
        access_token=access_token,
        reconciliation_enabled=reconciliation.get("enabled", True),
        reconciliation_interval_seconds=reconciliation.get("interval_seconds", 30),
        max_orders_per_second=rate_limits.get("max_orders_per_second", 8),
        max_orders_per_minute=rate_limits.get("max_orders_per_minute", 180),
        max_orders_per_day=rate_limits.get("max_orders_per_day", 2500),
        retry_config=retry_config,
        cache_ttl_seconds=cache.get("ttl_seconds", 5),
        positions_cache_ttl=cache.get("positions_ttl_seconds", 10),
    )

    logger.info(f"KiteBroker config loaded from {config_path}")

    # Security check: Ensure access token is not logged
    if access_token:
        logger.debug("Access token provided (not logged for security)")
    else:
        logger.warning(
            "No access token provided. Call generate_session() after user login flow."
        )

    return broker_config
