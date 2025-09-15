"""Enhanced logging configuration for CV Generator with component-specific loggers."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict
from enum import Enum

import sys
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config


class LogComponent(Enum):
    """Available logging components."""
    SCRAPING = "scraping"
    AGENTS = "agents"
    ERRORS = "errors"
    MAIN = "main"


# Global registry to track configured loggers
_configured_loggers: Dict[str, logging.Logger] = {}


def setup_component_logging(
    component: LogComponent,
    log_level: Optional[str] = None,
    logs_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up component-specific logging configuration.

    Args:
        component: The component to configure logging for
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logs_dir: Directory to store log files

    Returns:
        Configured logger instance for the component
    """
    # Use config values if not provided
    log_level = log_level or config.log_level
    logs_dir = logs_dir or config.logs_dir

    # Ensure logs directory exists
    logs_dir.mkdir(exist_ok=True)

    # Create component-specific logger name
    logger_name = f"cv_generator.{component.value}"

    # Return existing logger if already configured
    if logger_name in _configured_loggers:
        return _configured_loggers[logger_name]

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False  # Prevent duplicate messages

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (only for errors and main, with simple format)
    if component in [LogComponent.ERRORS, LogComponent.MAIN]:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if component == LogComponent.MAIN else logging.ERROR)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Component-specific file handler
    log_file = f"{component.value}.log"
    file_path = logs_dir / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Errors also get logged to a general error file
    if component != LogComponent.ERRORS:
        error_file_path = logs_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    # Store in registry
    _configured_loggers[logger_name] = logger

    # Log initial message
    logger.info(f"Component logging initialized - {component.value} - Level: {log_level}, File: {file_path}")

    return logger


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    logs_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up main application logging configuration.

    This function maintains backward compatibility while setting up the main logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Name of the log file (legacy parameter)
        logs_dir: Directory to store log files

    Returns:
        Configured main logger instance
    """
    return setup_component_logging(LogComponent.MAIN, log_level, logs_dir)


def get_scraping_logger() -> logging.Logger:
    """Get the scraping component logger."""
    return setup_component_logging(LogComponent.SCRAPING)


def get_agents_logger() -> logging.Logger:
    """Get the agents component logger."""
    return setup_component_logging(LogComponent.AGENTS)


def get_error_logger() -> logging.Logger:
    """Get the error component logger."""
    return setup_component_logging(LogComponent.ERRORS)


def get_logger(name: str = "cv_generator") -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: Logger name (defaults to main application logger)

    Returns:
        Logger instance
    """
    if name == "cv_generator":
        return setup_component_logging(LogComponent.MAIN)
    return logging.getLogger(name)


def get_component_logger(component: LogComponent) -> logging.Logger:
    """
    Get a component-specific logger.

    Args:
        component: The component to get logger for

    Returns:
        Component-specific logger instance
    """
    return setup_component_logging(component)