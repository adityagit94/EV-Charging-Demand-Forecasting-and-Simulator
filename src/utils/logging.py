"""Logging configuration and utilities."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import settings


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 month",
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration with loguru.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is used.
        rotation: Log file rotation policy
        retention: Log file retention policy
        format_string: Custom format string for log messages
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str):
    """Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize logging with settings
setup_logging(
    log_level=settings.logging.level,
    log_file=settings.logging.log_file,
    rotation=settings.logging.rotation,
    retention=settings.logging.retention,
    format_string=settings.logging.format,
)
