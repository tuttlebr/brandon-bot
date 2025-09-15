"""
Centralized Logging Configuration

This module provides a centralized logging configuration that ensures
consistent formatting across the entire application. All modules should
use this configuration instead of setting up their own logging.
"""

import logging
import os
from typing import Optional


class LoggingConfig:
    """Centralized logging configuration manager"""

    # Standard logging format used across the entire application
    STANDARD_FORMAT = (
        "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
    )
    STANDARD_DATEFMT = "%Y-%m-%d %H:%M:%S"

    # Log directory
    LOG_DIR = "/tmp/chatbot_storage"

    _configured = False

    @classmethod
    def configure_logging(
        cls,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        force_reconfigure: bool = False,
    ) -> None:
        """
        Configure logging with the standard format

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path (defaults to chatbot.log in LOG_DIR)
            force_reconfigure: Force reconfiguration even if already configured
        """
        if cls._configured and not force_reconfigure:
            return

        # Ensure log directory exists
        os.makedirs(cls.LOG_DIR, exist_ok=True)

        # Set up log file path
        if log_file is None:
            log_file = os.path.join(cls.LOG_DIR, "chatbot.log")

        # Configure logging with standard format
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=cls.STANDARD_FORMAT,
            datefmt=cls.STANDARD_DATEFMT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode="a"),
            ],
            force=force_reconfigure,
        )

        cls._configured = True

        # Log the configuration
        logger = logging.getLogger(__name__)
        logger.info(
            "Logging configured with level=%s, format=%s, file=%s",
            log_level.upper(),
            cls.STANDARD_FORMAT,
            log_file,
        )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the standard configuration

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        # Ensure logging is configured
        if not cls._configured:
            cls.configure_logging()

        return logging.getLogger(name)

    @classmethod
    def is_configured(cls) -> bool:
        """Check if logging has been configured"""
        return cls._configured

    @classmethod
    def reset(cls) -> None:
        """Reset configuration state (for testing)"""
        cls._configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a configured logger

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return LoggingConfig.get_logger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    force_reconfigure: bool = False,
) -> None:
    """
    Convenience function to configure logging

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path (defaults to chatbot.log in LOG_DIR)
        force_reconfigure: Force reconfiguration even if already configured
    """
    LoggingConfig.configure_logging(log_level, log_file, force_reconfigure)
