"""
Trading orchestration module.

This module provides the infrastructure for orchestrating live trading workflows,
including scheduling, state management, and workflow execution.

Key Components:
    - TradingScheduler: Schedules trading workflows at specified times
    - TradingWorkflow: Orchestrates data → prediction → execution pipeline
    - StateManager: Manages orchestrator state persistence
    - OrchestrationConfig: Configuration for orchestration settings

Example:
    >>> from tradebox.orchestration import TradingScheduler, TradingWorkflow
    >>> from tradebox.orchestration import OrchestrationConfig
    >>>
    >>> # Load configuration
    >>> config = OrchestrationConfig.from_yaml("configs/orchestration/paper_eod.yaml")
    >>>
    >>> # Initialize workflow
    >>> workflow = TradingWorkflow(config)
    >>>
    >>> # Run once
    >>> workflow.execute()
    >>>
    >>> # Or schedule continuous execution
    >>> scheduler = TradingScheduler(config)
    >>> scheduler.run_continuous(workflow)
"""

from tradebox.orchestration.config import OrchestrationConfig
from tradebox.orchestration.exceptions import (
    OrchestrationError,
    SchedulerError,
    WorkflowError,
    StateError,
)
from tradebox.orchestration.scheduler import TradingScheduler
from tradebox.orchestration.state import StateManager, OrchestratorState
from tradebox.orchestration.workflow import TradingWorkflow

__all__ = [
    "OrchestrationConfig",
    "OrchestrationError",
    "SchedulerError",
    "WorkflowError",
    "StateError",
    "TradingScheduler",
    "StateManager",
    "OrchestratorState",
    "TradingWorkflow",
]
