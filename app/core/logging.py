"""Logging configuration."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single, plain-text root logger for the service."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (logging is configured once at startup)."""
    return logging.getLogger(name)
