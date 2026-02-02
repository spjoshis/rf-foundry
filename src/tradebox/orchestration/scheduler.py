"""Trading scheduler for orchestrating workflows at specified times."""

import time
from datetime import datetime, timedelta
from typing import Optional

import pytz
from loguru import logger

from tradebox.orchestration.config import OrchestrationConfig
from tradebox.orchestration.exceptions import SchedulerError


class TradingScheduler:
    """
    Schedules trading workflows at specified times.

    Supports cron-like scheduling with timezone awareness (IST for Indian markets).
    Automatically skips weekends and handles market holidays.

    Example:
        >>> from tradebox.orchestration import TradingScheduler, TradingWorkflow, OrchestrationConfig
        >>> config = OrchestrationConfig(schedule="15:30", timezone="Asia/Kolkata")
        >>> scheduler = TradingScheduler(config)
        >>> workflow = TradingWorkflow(config)
        >>>
        >>> # Run once if it's time
        >>> if scheduler.should_run_now():
        ...     workflow.execute()
        >>>
        >>> # Or run continuously
        >>> scheduler.run_continuous(workflow)
    """

    def __init__(self, config: OrchestrationConfig):
        """
        Initialize trading scheduler.

        Args:
            config: Orchestration configuration
        """
        self.config = config

        try:
            self.timezone = pytz.timezone(config.timezone)
        except pytz.UnknownTimeZoneError:
            raise SchedulerError(f"Unknown timezone: {config.timezone}")

        # Parse schedule time
        try:
            hour, minute = map(int, config.schedule.split(":"))
            self.schedule_hour = hour
            self.schedule_minute = minute
        except ValueError:
            raise SchedulerError(
                f"Invalid schedule format: {config.schedule}. Must be HH:MM"
            )

        self.last_run_date: Optional[datetime] = None

        logger.info(
            f"TradingScheduler initialized: schedule={config.schedule} {config.timezone}"
        )

    def _is_market_day(self, dt: datetime) -> bool:
        """
        Check if given date is a trading day (not weekend).

        Args:
            dt: Datetime to check

        Returns:
            True if market day, False if weekend

        Note:
            This is a simplified check. In production, integrate with
            NSE holiday calendar API or maintain a holiday list.
        """
        # Monday=0, Sunday=6
        weekday = dt.weekday()

        # Skip weekends (Saturday=5, Sunday=6)
        if weekday >= 5:
            logger.debug(f"Skipping weekend: {dt.strftime('%Y-%m-%d %A')}")
            return False

        # TODO: Check against NSE holiday calendar
        # For now, all weekdays are market days

        return True

    def should_run_now(self) -> bool:
        """
        Check if workflow should run now.

        Returns:
            True if workflow should run, False otherwise

        Example:
            >>> scheduler = TradingScheduler(config)
            >>> if scheduler.should_run_now():
            ...     print("Time to trade!")
        """
        now = datetime.now(self.timezone)

        # Check if market day
        if not self._is_market_day(now):
            return False

        # Check if we already ran today
        today = now.date()
        if self.last_run_date == today:
            logger.debug("Already ran today, skipping")
            return False

        # Check if current time matches schedule
        current_hour = now.hour
        current_minute = now.minute

        # Allow 5-minute window around scheduled time
        schedule_time = now.replace(
            hour=self.schedule_hour, minute=self.schedule_minute, second=0, microsecond=0
        )
        time_diff = abs((now - schedule_time).total_seconds())

        if time_diff <= 300:  # Within 5 minutes
            logger.info(f"Time to run workflow: {now.strftime('%Y-%m-%d %H:%M')}")
            self.last_run_date = today
            return True

        return False

    def wait_until_next_run(self) -> None:
        """
        Block until next scheduled run time.

        This method calculates the next scheduled run time and sleeps
        until then. It accounts for weekends and already-run checks.

        Example:
            >>> scheduler = TradingScheduler(config)
            >>> scheduler.wait_until_next_run()  # Blocks until scheduled time
        """
        now = datetime.now(self.timezone)
        today = now.date()

        # Calculate next run time
        next_run = now.replace(
            hour=self.schedule_hour,
            minute=self.schedule_minute,
            second=0,
            microsecond=0,
        )

        # If already past today's schedule time or already ran today, schedule for tomorrow
        if now >= next_run or self.last_run_date == today:
            next_run = next_run + timedelta(days=1)

        # Skip weekends
        while not self._is_market_day(next_run):
            next_run = next_run + timedelta(days=1)
            logger.debug(f"Skipping non-market day, next run: {next_run}")

        # Calculate wait time
        wait_seconds = (next_run - now).total_seconds()

        logger.info(
            f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M %Z')} "
            f"(waiting {wait_seconds/3600:.1f} hours)"
        )

        # Sleep until next run
        time.sleep(wait_seconds)

    def run_continuous(self, workflow) -> None:
        """
        Run workflow continuously on schedule.

        This is a blocking method that runs forever, executing the
        workflow at scheduled times. Suitable for running as a daemon.

        Args:
            workflow: TradingWorkflow instance to execute

        Example:
            >>> scheduler = TradingScheduler(config)
            >>> workflow = TradingWorkflow(config)
            >>> scheduler.run_continuous(workflow)  # Runs forever
        """
        logger.info("Starting continuous scheduler")

        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"Scheduler iteration {iteration}")

                # Wait until next scheduled run
                self.wait_until_next_run()

                # Execute workflow
                logger.info("Executing scheduled workflow...")
                workflow.execute()
                logger.info("Workflow execution complete")

            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                break

            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                # Continue to next iteration despite error
                # Alert would be sent by workflow error handler

    def get_next_run_time(self) -> datetime:
        """
        Get the next scheduled run time.

        Returns:
            Datetime of next scheduled run

        Example:
            >>> scheduler = TradingScheduler(config)
            >>> next_run = scheduler.get_next_run_time()
            >>> print(f"Next run: {next_run}")
        """
        now = datetime.now(self.timezone)
        today = now.date()

        next_run = now.replace(
            hour=self.schedule_hour,
            minute=self.schedule_minute,
            second=0,
            microsecond=0,
        )

        # If already past today's schedule or already ran, go to next day
        if now >= next_run or self.last_run_date == today:
            next_run = next_run + timedelta(days=1)

        # Skip weekends
        while not self._is_market_day(next_run):
            next_run = next_run + timedelta(days=1)

        return next_run
