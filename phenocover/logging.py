#!/usr/bin/env python

"""
Logging Module for Wheat Phenology Estimator
Description:
Logging configuration and utilities.

This module provides a centralized logging configuration with support for:
- Console and file logging
- Rich formatting for enhanced console output
- Configurable log levels
"""

__author__ = "Joseph Gitahi"
__email__ = "joemureithi@live.com"
__maintainer__ = "Joseph Gitahi"
__year__ = "2025"

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

try:
    from rich import print as rich_print
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install
    RICH_AVAILABLE = True
except ImportError:
    rich_print = None
    Console = None
    RichHandler = None
    install = None
    RICH_AVAILABLE = False

# Ensure we always have print available
print = print  # Built-in print function


class Logger:
    """
    A comprehensive logging class for the phenocover package.

    Provides centralized logging configuration with support for both
    console and file logging, with optional rich formatting.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure only one logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the logger if not already initialized."""
        if not self._initialized:
            # Initialize console with safer settings for Windows
            if RICH_AVAILABLE and Console:
                try:
                    # Try to create console with safer width/height settings
                    self.console = Console(
                        # Limit width to prevent buffer issues
                        # width=min(120, 80),
                        # height=min(50, 30),  # Limit height
                        force_terminal=True,  # Let Rich detect terminal capabilities
                        legacy_windows=True   # Better Windows compatibility
                    )
                except Exception:
                    # Fallback to None if Rich console creation fails
                    self.console = None
            else:
                self.console = None

            self._loggers = {}
            self._log_dir = None
            self._setup_rich_traceback()
            Logger._initialized = True

    def _setup_rich_traceback(self):
        """Setup rich traceback handling if available."""
        if RICH_AVAILABLE and install and self.console:
            install(show_locals=True, console=self.console)

    def configure_logging(
        self,
        level: Union[str, int] = logging.INFO,
        log_dir: Optional[Union[str, Path]] = None,
        log_filename: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        use_rich: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_string: Optional[str] = None,
        suppress_third_party_debug: bool = True
    ) -> None:
        """
        Configure the global logging settings.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (defaults to ./logs)
            log_filename: Log file name (defaults to wheat-phenology-estimator_YYYY-MM-DD.log)
            enable_file_logging: Whether to enable file logging
            enable_console_logging: Whether to enable console logging
            use_rich: Whether to use rich formatting for console output
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup files to keep
            format_string: Custom format string for file logging
            suppress_third_party_debug: Whether to suppress DEBUG logs from third-party libraries
        """
        # Convert string level to int if necessary
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        # Setup log directory
        if log_dir is None:
            log_dir = Path('./logs')
        else:
            log_dir = Path(log_dir)

        self._log_dir = log_dir

        if enable_file_logging:
            log_dir.mkdir(exist_ok=True)

        # Setup log filename
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y-%m-%d')
            log_filename = f'phenocover_{timestamp}.log'

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Setup formatters
        if format_string is None:
            format_string = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(funcName)s:%(lineno)d - %(message)s'
            )

        file_formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Setup file logging
        if enable_file_logging:
            log_file = log_dir / log_filename
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Setup console logging
        if enable_console_logging:
            if use_rich and RICH_AVAILABLE and RichHandler and self.console:
                try:
                    console_handler = RichHandler(
                        console=self.console,
                        show_time=True,
                        show_path=True,
                        rich_tracebacks=True,
                        tracebacks_show_locals=True
                    )
                    console_handler.setLevel(level)
                    root_logger.addHandler(console_handler)
                except Exception:
                    # Fallback to standard console handler if Rich fails
                    console_handler = logging.StreamHandler(sys.stdout)
                    console_handler.setLevel(level)
                    console_formatter = logging.Formatter(
                        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%H:%M:%S'
                    )
                    console_handler.setFormatter(console_formatter)
                    root_logger.addHandler(console_handler)
            else:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
                root_logger.addHandler(console_handler)

        # Suppress noisy third-party debug logs if requested
        if suppress_third_party_debug:
            # Convert level to int if it's a string for comparison
            level_int = level if isinstance(
                level, int) else getattr(logging, level.upper())

            if level_int <= logging.DEBUG:
                # List of third-party loggers that are typically too verbose at DEBUG level
                noisy_loggers = [
                    'urllib3.connectionpool',
                    'requests.packages.urllib3.connectionpool',
                    'urllib3',
                    'requests',
                    'matplotlib',
                    'PIL',
                    'fiona',
                    'rasterio',
                    'boto3',
                    'botocore',
                    's3transfer'
                ]

                for logger_name in noisy_loggers:
                    third_party_logger = logging.getLogger(logger_name)
                    # Suppress DEBUG messages
                    third_party_logger.setLevel(logging.INFO)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for the given name.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]

    def log_error(self, error: Exception, context: Optional[str] = None, **kwargs) -> None:
        """
        Log errors with context information.

        Args:
            error: Exception that occurred
            context: Additional context description
            **kwargs: Additional context information
        """
        logger = self.get_logger('phenocover.error')
        message = f"{type(error).__name__}: {str(error)}"
        if context:
            message = f"{context} | {message}"

        extra_context = ' | '.join([f'{k}={v}' for k, v in kwargs.items()])
        if extra_context:
            message += f" | {extra_context}"

        logger.error(message, exc_info=True)

    def set_level(self, level: Union[str, int]) -> None:
        """
        Set the logging level for all handlers.

        Args:
            level: New logging level
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        for handler in root_logger.handlers:
            handler.setLevel(level)

    def get_log_files(self) -> list:
        """
        Get list of current log files.

        Returns:
            List of log file paths
        """
        if self._log_dir and self._log_dir.exists():
            return list(self._log_dir.glob('*.log*'))
        return []

    def cleanup_old_logs(self, max_age_days: int = 30) -> None:
        """
        Clean up old log files.

        Args:
            max_age_days: Maximum age of log files to keep
        """
        if not self._log_dir or not self._log_dir.exists():
            return

        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

        for log_file in self._log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.get_logger('phenocover.maintenance').info(
                        f"Removed old log file: {log_file.name}"
                    )
                except OSError as e:
                    self.get_logger('phenocover.maintenance').error(
                        f"Failed to remove log file {log_file.name}: {e}"
                    )


# Global logger instance
logger_instance = Logger()

# Convenience functions


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (defaults to calling module name)

    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get(
                '__name__', 'phenocover')
        else:
            name = 'phenocover'

    # Ensure name is not None at this point
    assert name is not None, "Logger name should not be None"
    return logger_instance.get_logger(name)


def configure_logging(**kwargs) -> None:
    """Configure global logging settings."""
    logger_instance.configure_logging(**kwargs)


def log_error(error: Exception, context: Optional[str] = None, **kwargs) -> None:
    """Log errors with context information."""
    logger_instance.log_error(error, context, **kwargs)


def set_log_level(level: Union[str, int]) -> None:
    """Set the logging level."""
    logger_instance.set_level(level)


def cleanup_old_logs(max_age_days: int = 30) -> None:
    """Clean up old log files."""
    logger_instance.cleanup_old_logs(max_age_days)


# Default logger for backward compatibility
LOGGER = get_logger(__name__)
