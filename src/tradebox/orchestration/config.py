"""Configuration for trading orchestration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from loguru import logger


@dataclass
class OrchestrationConfig:
    """
    Configuration for trading orchestration.

    Attributes:
        mode: Trading mode ('paper' or 'live')
        schedule: Cron-like schedule string (e.g., "15:30" for 3:30 PM)
        timezone: Timezone for scheduling (e.g., "Asia/Kolkata")
        symbols: List of symbols to trade
        model_path: Path to trained model weights
        broker: Broker type ('paper' or 'kite')
        initial_capital: Initial capital for paper trading
        max_position_size_pct: Max position size as % of portfolio
        max_stock_allocation_pct: Max allocation per stock
        max_daily_loss_pct: Max daily loss allowed
        max_sector_concentration_pct: Max sector concentration
        min_daily_volume: Minimum daily volume required
        max_leverage: Maximum leverage allowed
        metrics_db: Path to metrics database
        alert_email: Email for alerts
        state_file: Path to state persistence file
        log_level: Logging level
        log_file: Path to log file

    Example:
        >>> config = OrchestrationConfig.from_yaml("configs/orchestration/paper_eod.yaml")
        >>> print(config.mode)
        'paper'
    """

    # Orchestration settings
    mode: str = "paper"  # paper | live
    schedule: str = "15:30"  # HH:MM format
    timezone: str = "Asia/Kolkata"
    symbols: List[str] = field(default_factory=lambda: ["^NSEI"])
    model_path: str = "models/best_model.zip"

    # Execution settings
    broker: str = "paper"  # paper | kite
    initial_capital: float = 100000.0

    # Risk management settings
    max_position_size_pct: float = 0.20
    max_stock_allocation_pct: float = 0.15
    max_daily_loss_pct: float = 0.02
    max_sector_concentration_pct: float = 0.30
    min_daily_volume: int = 100000
    max_leverage: float = 1.0

    # Monitoring settings
    metrics_db: str = "data/metrics.db"
    alert_email: Optional[str] = None

    # State management
    state_file: str = "data/orchestrator_state.json"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/orchestrator.log"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate mode
        if self.mode not in ["paper", "live"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'paper' or 'live'")

        # Validate broker
        if self.broker not in ["paper", "kite"]:
            raise ValueError(
                f"Invalid broker: {self.broker}. Must be 'paper' or 'kite'"
            )

        # Validate schedule format (HH:MM)
        try:
            hour, minute = map(int, self.schedule.split(":"))
            if not (0 <= hour < 24 and 0 <= minute < 60):
                raise ValueError("Invalid time range")
        except (ValueError, AttributeError):
            raise ValueError(
                f"Invalid schedule format: {self.schedule}. Must be HH:MM"
            )

        # Validate symbols
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")

        # Validate model path
        if not self.model_path:
            raise ValueError("model_path cannot be empty")

        # Validate risk parameters
        if not (0.0 < self.max_position_size_pct <= 1.0):
            raise ValueError("max_position_size_pct must be between 0 and 1")

        if not (0.0 < self.max_stock_allocation_pct <= 1.0):
            raise ValueError("max_stock_allocation_pct must be between 0 and 1")

        if not (0.0 < self.max_daily_loss_pct <= 1.0):
            raise ValueError("max_daily_loss_pct must be between 0 and 1")

        if self.max_leverage < 0.0:
            raise ValueError("max_leverage must be >= 0")

        # Validate initial capital
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        logger.info(f"OrchestrationConfig validated: mode={self.mode}, broker={self.broker}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OrchestrationConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            OrchestrationConfig instance

        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If YAML is invalid

        Example:
            >>> config = OrchestrationConfig.from_yaml("configs/orchestration/paper_eod.yaml")
        """
        yaml_file = Path(yaml_path)

        if not yaml_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        # Extract nested configuration
        orchestration = data.get("orchestration", {})
        execution = data.get("execution", {})
        risk = data.get("risk", {})
        monitoring = data.get("monitoring", {})
        logging_config = data.get("logging", {})

        # Build config object
        config = cls(
            # Orchestration
            mode=orchestration.get("mode", "paper"),
            schedule=orchestration.get("schedule", "15:30"),
            timezone=orchestration.get("timezone", "Asia/Kolkata"),
            symbols=orchestration.get("symbols", ["^NSEI"]),
            model_path=orchestration.get("model_path", "models/best_model.zip"),
            # Execution
            broker=execution.get("broker", "paper"),
            initial_capital=execution.get("initial_capital", 100000.0),
            # Risk
            max_position_size_pct=risk.get("max_position_size_pct", 0.20),
            max_stock_allocation_pct=risk.get("max_stock_allocation_pct", 0.15),
            max_daily_loss_pct=risk.get("max_daily_loss_pct", 0.02),
            max_sector_concentration_pct=risk.get("max_sector_concentration_pct", 0.30),
            min_daily_volume=risk.get("min_daily_volume", 100000),
            max_leverage=risk.get("max_leverage", 1.0),
            # Monitoring
            metrics_db=monitoring.get("metrics_db", "data/metrics.db"),
            alert_email=monitoring.get("alert_email"),
            # Logging
            log_level=logging_config.get("level", "INFO"),
            log_file=logging_config.get("file", "logs/orchestrator.log"),
        )

        logger.info(f"Loaded configuration from {yaml_path}")
        return config

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "orchestration": {
                "mode": self.mode,
                "schedule": self.schedule,
                "timezone": self.timezone,
                "symbols": self.symbols,
                "model_path": self.model_path,
            },
            "execution": {
                "broker": self.broker,
                "initial_capital": self.initial_capital,
            },
            "risk": {
                "max_position_size_pct": self.max_position_size_pct,
                "max_stock_allocation_pct": self.max_stock_allocation_pct,
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_sector_concentration_pct": self.max_sector_concentration_pct,
                "min_daily_volume": self.min_daily_volume,
                "max_leverage": self.max_leverage,
            },
            "monitoring": {
                "metrics_db": self.metrics_db,
                "alert_email": self.alert_email,
            },
            "logging": {
                "level": self.log_level,
                "file": self.log_file,
            },
        }
