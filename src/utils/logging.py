"""
AQQIP Structured Logging

Provides a configured structlog logger.
"""
from __future__ import annotations

import logging
import sys

import structlog

from config.settings import get_settings


def setup_logging() -> None:
    """Initialize structured logging."""
    settings = get_settings()
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True) if sys.stdout.isatty() else structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog logger bound to the given name."""
    return structlog.get_logger(name)
