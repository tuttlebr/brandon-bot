"""
Model Schema Helper

This module provides helper functions for easily configuring and managing
LLM model schemas via environment variables and runtime configuration.
"""

import os
from typing import Dict, Optional

from utils.llm_schema_manager import LLMSchema, schema_manager
from utils.logging_config import get_logger

logger = get_logger(__name__)


def configure_model_schema_from_env(model_name: str) -> bool:
    """
    Configure a model schema from environment variables

    Environment variables expected:
    - {MODEL_NAME}_THINKING_START: Thinking start tag
    - {MODEL_NAME}_THINKING_STOP: Thinking stop tag
    - {MODEL_NAME}_TOOL_START: Tool call start tag
    - {MODEL_NAME}_TOOL_STOP: Tool call stop tag
    - {MODEL_NAME}_ANALYSIS_START: Analysis block start tag (optional)
    - {MODEL_NAME}_ANALYSIS_STOP: Analysis block stop tag (optional)

    Args:
        model_name: Name of the model to configure

    Returns:
        True if any schema configuration was found and applied
    """
    # Normalize model name for environment variable naming
    env_prefix = (
        model_name.upper()
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )

    # Check for environment variables
    thinking_start = os.getenv(f"{env_prefix}_THINKING_START")
    thinking_stop = os.getenv(f"{env_prefix}_THINKING_STOP")
    tool_start = os.getenv(f"{env_prefix}_TOOL_START")
    tool_stop = os.getenv(f"{env_prefix}_TOOL_STOP")
    analysis_start = os.getenv(f"{env_prefix}_ANALYSIS_START")
    analysis_stop = os.getenv(f"{env_prefix}_ANALYSIS_STOP")

    # Check if any schema configuration was found
    schema_config_found = any(
        [
            thinking_start,
            thinking_stop,
            tool_start,
            tool_stop,
            analysis_start,
            analysis_stop,
        ]
    )

    if not schema_config_found:
        return False

    # Get current default schema
    current_schema = schema_manager.get_schema(model_name)

    # Create new schema with environment overrides
    new_schema = LLMSchema(
        thinking_start=thinking_start or current_schema.thinking_start,
        thinking_stop=thinking_stop or current_schema.thinking_stop,
        tool_start=tool_start or current_schema.tool_start,
        tool_stop=tool_stop or current_schema.tool_stop,
        analysis_start=analysis_start or current_schema.analysis_start,
        analysis_stop=analysis_stop or current_schema.analysis_stop,
    )

    # Register the schema
    schema_manager.register_model_schema(model_name, new_schema)

    logger.info(
        "Configured schema for model '%s' from environment variables",
        model_name,
    )
    logger.debug(
        "Schema details: thinking=(%s, %s), tools=(%s, %s), analysis=(%s, %s)",
        new_schema.thinking_start,
        new_schema.thinking_stop,
        new_schema.tool_start,
        new_schema.tool_stop,
        new_schema.analysis_start,
        new_schema.analysis_stop,
    )

    return True


def configure_model_schema(
    model_name: str,
    thinking_start: str = None,
    thinking_stop: str = None,
    tool_start: str = None,
    tool_stop: str = None,
    analysis_start: str = None,
    analysis_stop: str = None,
) -> None:
    """
    Configure a model schema programmatically

    Args:
        model_name: Name of the model
        thinking_start: Thinking start tag
        thinking_stop: Thinking stop tag
        tool_start: Tool call start tag
        tool_stop: Tool call stop tag
        analysis_start: Analysis block start tag (optional)
        analysis_stop: Analysis block stop tag (optional)
    """
    # Get current default schema
    current_schema = schema_manager.get_schema(model_name)

    # Create new schema with provided overrides
    new_schema = LLMSchema(
        thinking_start=thinking_start or current_schema.thinking_start,
        thinking_stop=thinking_stop or current_schema.thinking_stop,
        tool_start=tool_start or current_schema.tool_start,
        tool_stop=tool_stop or current_schema.tool_stop,
        analysis_start=analysis_start or current_schema.analysis_start,
        analysis_stop=analysis_stop or current_schema.analysis_stop,
    )

    # Register the schema
    schema_manager.register_model_schema(model_name, new_schema)

    logger.info(
        "Configured schema for model '%s' programmatically", model_name
    )


def get_model_schema_info(model_name: str) -> Dict[str, Optional[str]]:
    """
    Get schema information for a model

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with schema information
    """
    schema = schema_manager.get_schema(model_name)

    return {
        "thinking_start": schema.thinking_start,
        "thinking_stop": schema.thinking_stop,
        "tool_start": schema.tool_start,
        "tool_stop": schema.tool_stop,
        "analysis_start": schema.analysis_start,
        "analysis_stop": schema.analysis_stop,
    }


def list_all_model_schemas() -> Dict[str, Dict[str, Optional[str]]]:
    """
    List all configured model schemas

    Returns:
        Dictionary mapping model names to their schema configurations
    """
    all_schemas = schema_manager.list_configured_models()

    result = {}
    for model_name, schema in all_schemas.items():
        result[model_name] = {
            "thinking_start": schema.thinking_start,
            "thinking_stop": schema.thinking_stop,
            "tool_start": schema.tool_start,
            "tool_stop": schema.tool_stop,
            "analysis_start": schema.analysis_start,
            "analysis_stop": schema.analysis_stop,
        }

    return result


def auto_configure_common_models():
    """
    Auto-configure schemas for commonly used models based on known patterns

    Note: This function is intentionally empty to avoid overriding user configurations.
    Users should configure their models via JSON file or environment variables.
    """
    # No automatic configurations to avoid overriding user settings
    # Users should use model_schemas.json or environment variables instead


# Note: Model schemas are now loaded exclusively from JSON file or environment variables
# No automatic configurations are applied to respect user settings
