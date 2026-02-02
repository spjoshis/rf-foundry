"""Custom exceptions for orchestration module."""


class OrchestrationError(Exception):
    """Base exception for orchestration errors."""

    pass


class SchedulerError(OrchestrationError):
    """Exception raised for scheduler errors."""

    pass


class WorkflowError(OrchestrationError):
    """Exception raised for workflow execution errors."""

    pass


class StateError(OrchestrationError):
    """Exception raised for state management errors."""

    pass
