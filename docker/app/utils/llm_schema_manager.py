"""
LLM Schema Manager

This module provides dynamic schema support for different LLM models,
allowing each model to have its own thinking and tool call formats.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from utils.config import config
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMSchema:
    """Schema definition for an LLM model"""

    # Thinking/reasoning tags
    thinking_start: str = "<think>"
    thinking_stop: str = "</think>"

    # Tool call tags
    tool_start: str = "<tool_call>"
    tool_stop: str = "</tool_call>"

    # Additional analysis block tags (for backward compatibility)
    analysis_start: Optional[str] = "analysis"
    analysis_stop: Optional[str] = "assistantfinal"


class LLMSchemaManager:
    """Manages LLM schemas for different models"""

    _instance: Optional["LLMSchemaManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._schemas: Dict[str, LLMSchema] = {}
        self._default_schema = LLMSchema(
            thinking_start=config.llm.DEFAULT_THINKING_START,
            thinking_stop=config.llm.DEFAULT_THINKING_STOP,
            tool_start=config.llm.DEFAULT_TOOL_START,
            tool_stop=config.llm.DEFAULT_TOOL_STOP,
        )

        # Load model-specific schemas from config
        self._load_model_schemas()
        self._initialized = True

        logger.info(
            "LLM Schema Manager initialized with %d model-specific schemas",
            len(self._schemas),
        )

    def _load_model_schemas(self):
        """Load model-specific schemas from configuration and JSON file"""
        # First try to load from JSON file
        self._load_schemas_from_file()

        # Then load from environment variable (this can override file settings)
        self._load_schemas_from_env()

    def _load_schemas_from_file(self):
        """Load schemas from JSON file"""
        try:
            schemas_file = config.llm.MODEL_SCHEMAS_FILE
            if not schemas_file:
                logger.debug("No MODEL_SCHEMAS_FILE configured")
                return

            # Try absolute path first, then relative to app directory
            import os

            file_paths = [
                schemas_file,
            ]

            schemas_data = None
            used_path = None

            for file_path in file_paths:
                logger.info("Checking for schema file at: %s", file_path)
                if os.path.exists(file_path):
                    logger.info("Found schema file at: %s", file_path)
                    with open(file_path, "r") as f:
                        schemas_data = json.load(f)
                    logger.info("Loaded JSON data: %s", schemas_data)
                    used_path = file_path
                    break
                else:
                    logger.debug("Schema file not found at: %s", file_path)

            if schemas_data is None:
                logger.debug(
                    "Model schemas file not found at any of: %s", file_paths
                )
                return

            logger.info("Loading model schemas from file: %s", used_path)

            for model_name, schema_config in schemas_data.items():
                schema = self._create_schema_from_config(schema_config)
                self._schemas[model_name] = schema
                logger.info(
                    "Loaded schema from file for model '%s': thinking=(%s,"
                    " %s), tools=(%s, %s)",
                    model_name,
                    schema.thinking_start,
                    schema.thinking_stop,
                    schema.tool_start,
                    schema.tool_stop,
                )

        except Exception as e:
            logger.error("Error loading model schemas from file: %s", e)

    def _load_schemas_from_env(self):
        """Load schemas from environment variable"""
        try:
            model_schemas = config.llm.MODEL_SCHEMAS

            for model_name, schema_config in model_schemas.items():
                schema = self._create_schema_from_config(schema_config)
                self._schemas[model_name] = schema
                logger.info(
                    "Loaded schema from env for model '%s': thinking=(%s, %s),"
                    " tools=(%s, %s)",
                    model_name,
                    schema.thinking_start,
                    schema.thinking_stop,
                    schema.tool_start,
                    schema.tool_stop,
                )

        except Exception as e:
            logger.error("Error loading model schemas from environment: %s", e)

    def _create_schema_from_config(
        self, schema_config: Dict[str, Any]
    ) -> LLMSchema:
        """Create LLMSchema from configuration dict, handling null values"""

        # Helper function to handle null values properly
        def get_value(key: str, default_value: str) -> str:
            if key in schema_config:
                value = schema_config[key]
                # If explicitly set to null, return empty string (disables filtering)
                if value is None:
                    logger.info(
                        "Schema config key '%s' is null, returning empty"
                        " string",
                        key,
                    )
                    return ""
                # If set to empty string, keep it empty
                elif value == "":
                    logger.info("Schema config key '%s' is empty string", key)
                    return ""
                # Otherwise use the provided value
                else:
                    logger.info(
                        "Schema config key '%s' has value: %s",
                        key,
                        repr(value),
                    )
                    return value
            # If not specified, use default
            else:
                logger.debug(
                    "Schema config key '%s' not found, using default: %s",
                    key,
                    repr(default_value),
                )
                return default_value

        return LLMSchema(
            thinking_start=get_value(
                "thinking_start", self._default_schema.thinking_start
            ),
            thinking_stop=get_value(
                "thinking_stop", self._default_schema.thinking_stop
            ),
            tool_start=get_value(
                "tool_start", self._default_schema.tool_start
            ),
            tool_stop=get_value("tool_stop", self._default_schema.tool_stop),
            analysis_start=get_value(
                "analysis_start", self._default_schema.analysis_start
            ),
            analysis_stop=get_value(
                "analysis_stop", self._default_schema.analysis_stop
            ),
        )

    def get_schema(self, model_name: str) -> LLMSchema:
        """
        Get schema for a specific model

        Args:
            model_name: Name of the model

        Returns:
            LLMSchema for the model, or default schema if not found
        """
        # Try exact match first
        if model_name in self._schemas:
            return self._schemas[model_name]

        # Try partial matches for model families
        for schema_model, schema in self._schemas.items():
            if (
                schema_model.lower() in model_name.lower()
                or model_name.lower() in schema_model.lower()
            ):
                logger.debug(
                    "Using partial match schema '%s' for model '%s'",
                    schema_model,
                    model_name,
                )
                return schema

        # Return default schema
        logger.debug("Using default schema for model '%s'", model_name)
        return self._default_schema

    def get_thinking_tags(self, model_name: str) -> Tuple[str, str]:
        """
        Get thinking start and stop tags for a model

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (start_tag, stop_tag)
        """
        schema = self.get_schema(model_name)
        return schema.thinking_start, schema.thinking_stop

    def get_tool_tags(self, model_name: str) -> Tuple[str, str]:
        """
        Get tool call start and stop tags for a model

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (start_tag, stop_tag)
        """
        schema = self.get_schema(model_name)
        return schema.tool_start, schema.tool_stop

    def get_analysis_tags(
        self, model_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get analysis block start and stop tags for a model

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (start_tag, stop_tag) or (None, None) if not configured
        """
        schema = self.get_schema(model_name)
        return schema.analysis_start, schema.analysis_stop

    def register_model_schema(self, model_name: str, schema: LLMSchema):
        """
        Register a schema for a specific model at runtime

        Args:
            model_name: Name of the model
            schema: Schema configuration
        """
        self._schemas[model_name] = schema
        logger.info("Registered runtime schema for model '%s'", model_name)

    def list_configured_models(self) -> Dict[str, LLMSchema]:
        """
        Get all configured model schemas

        Returns:
            Dictionary of model name to schema mappings
        """
        return self._schemas.copy()


# Global instance for easy access
schema_manager = LLMSchemaManager()
