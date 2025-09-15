"""Logging configuration for CV Generator."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    logs_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Name of the log file
        logs_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    # Use config values if not provided
    log_level = log_level or config.log_level
    log_file = log_file or config.log_file
    logs_dir = logs_dir or config.logs_dir

    # Ensure logs directory exists
    logs_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("cv_generator")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_path = logs_dir / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log initial message
    logger.info(f"Logging initialized - Level: {log_level}, File: {file_path}")

    return logger


def get_logger(name: str = "cv_generator") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)