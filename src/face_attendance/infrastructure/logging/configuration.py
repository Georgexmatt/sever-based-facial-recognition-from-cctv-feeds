"""Loguru configuration for application startup."""

import sys
from pathlib import Path

from loguru import logger

from config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure Loguru sinks for console and optional file logging."""

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        serialize=settings.log_json,
        backtrace=settings.debug,
        diagnose=settings.debug,
    )

    if settings.log_to_file:
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level=settings.log_level.upper(),
            serialize=settings.log_json,
            rotation="10 MB",
            retention="14 days",
            compression="gz",
            backtrace=settings.debug,
            diagnose=settings.debug,
        )

    logger.debug("Loguru configured", level=settings.log_level, json=settings.log_json)
