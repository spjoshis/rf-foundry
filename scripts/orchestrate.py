#!/usr/bin/env python3
"""
Trading orchestrator CLI.

This script runs the trading workflow orchestrator, which schedules and executes
trading operations at specified times.

Usage:
    # Run once (manual execution)
    python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --once

    # Run continuously (daemon mode)
    python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml

    # Dry run (validate configuration without executing)
    python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --dry-run

Examples:
    # Paper trading with EOD strategy
    python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --once

    # Run continuously in background
    nohup python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml > orchestrator.log 2>&1 &

Author: TradeBox-RL
Date: 2025-12-15
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradebox.orchestration import (
    OrchestrationConfig,
    TradingScheduler,
    TradingWorkflow,
)


def setup_logging(config: OrchestrationConfig) -> None:
    """
    Configure logging.

    Args:
        config: Orchestration configuration
    """
    # Remove default logger
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=config.log_level,
        colorize=True,
    )

    # Add file handler
    log_file = Path(config.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        config.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=config.log_level,
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )


def validate_config(config: OrchestrationConfig) -> bool:
    """
    Validate configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    logger.info("Validating configuration...")

    # Check model exists
    model_path = Path(config.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False

    logger.info(f"✅ Model found: {model_path}")

    # Validate broker
    if config.broker not in ["paper", "kite"]:
        logger.error(f"Invalid broker: {config.broker}")
        return False

    logger.info(f"✅ Broker: {config.broker}")

    # Validate mode
    if config.mode not in ["paper", "live"]:
        logger.error(f"Invalid mode: {config.mode}")
        return False

    logger.info(f"✅ Mode: {config.mode}")

    # Validate symbols
    if not config.symbols:
        logger.error("No symbols configured")
        return False

    logger.info(f"✅ Symbols: {', '.join(config.symbols)}")

    # Validate schedule
    logger.info(f"✅ Schedule: {config.schedule} {config.timezone}")

    logger.info("Configuration validation successful")
    return True


def run_once(config: OrchestrationConfig) -> None:
    """
    Run workflow once (manual execution).

    Args:
        config: Orchestration configuration
    """
    logger.info("=" * 70)
    logger.info("MANUAL EXECUTION MODE")
    logger.info("=" * 70)

    workflow = TradingWorkflow(config)

    try:
        workflow.execute()
        logger.info("✅ Manual execution completed successfully")
    except Exception as e:
        logger.error(f"❌ Manual execution failed: {e}")
        sys.exit(1)


def run_continuous(config: OrchestrationConfig) -> None:
    """
    Run workflow continuously (daemon mode).

    Args:
        config: Orchestration configuration
    """
    logger.info("=" * 70)
    logger.info("CONTINUOUS EXECUTION MODE (DAEMON)")
    logger.info("=" * 70)
    logger.warning("Press Ctrl+C to stop")

    scheduler = TradingScheduler(config)
    workflow = TradingWorkflow(config)

    # Show next run time
    next_run = scheduler.get_next_run_time()
    logger.info(f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M %Z')}")

    try:
        scheduler.run_continuous(workflow)
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading orchestrator for TradeBox-RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once (manual)
  python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --once

  # Run continuously (daemon)
  python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml

  # Dry run (validate only)
  python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml --dry-run

  # Run in background
  nohup python scripts/orchestrate.py --config configs/orchestration/paper_eod.yaml > orchestrator.log 2>&1 &
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to orchestration configuration YAML file",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run workflow once and exit (manual execution)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing workflow",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = OrchestrationConfig.from_yaml(args.config)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    # Setup logging
    setup_logging(config)

    # Print header
    logger.info("=" * 70)
    logger.info("TRADEBOX-RL TRADING ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Broker: {config.broker}")
    logger.info(f"Schedule: {config.schedule} {config.timezone}")
    logger.info("=" * 70)

    # Validate configuration
    if not validate_config(config):
        logger.error("❌ Configuration validation failed")
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        logger.info("✅ Dry run successful - configuration is valid")
        sys.exit(0)

    # Execute
    if args.once:
        run_once(config)
    else:
        run_continuous(config)


if __name__ == "__main__":
    main()
