"""
Startup Utilities

This module handles application startup tasks like configuring logging
and suppressing warnings based on configuration.
"""

import logging

logger = logging.getLogger(__name__)


def initialize_app():
    """
    Initialize application settings at startup

    This should be called early in the application lifecycle
    to set up proper configurations.
    """
    from utils.config import config

    # Initialize tools once at startup
    try:
        from tools.initialize_tools import initialize_all_tools

        initialize_all_tools()
        logger.info("Tools initialized at startup")
    except Exception as e:
        logger.error(f"Failed to initialize tools at startup: {e}")
        # Don't raise - let the app continue and try again later

    # Initialize LLM client service
    try:
        from models.chat_config import ChatConfig
        from services.llm_client_service import llm_client_service

        config_obj = ChatConfig.from_environment()
        llm_client_service.initialize(config_obj)
        logger.info("LLM client service initialized at startup")
    except Exception as e:
        logger.error(
            f"Failed to initialize LLM client service at startup: {e}"
        )
        # Don't raise - let the app continue and try again later

    # Suppress Streamlit warnings if configured
    if config.system.SUPPRESS_STREAMLIT_WARNINGS:
        try:
            from utils.streamlit_context import suppress_streamlit_warnings

            suppress_streamlit_warnings()
            logger.info("Streamlit context warnings suppressed")
        except Exception as e:
            logger.debug(f"Could not suppress Streamlit warnings: {e}")
