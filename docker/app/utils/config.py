"""
Centralized Configuration Management System

This module provides a unified approach to managing all configuration values,
environment variables, and constants across the application. It replaces
scattered hardcoded values and environment variable access throughout the codebase.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UIConfig:
    """UI-related configuration constants"""

    # Spinner and loading indicators
    SPINNER_ICONS: List[str] = field(default_factory=lambda: ["🤖", "🧠", "🤔", "🤓", "⚡"])

    # Display limits and formatting
    MAX_PROMPT_DISPLAY_LENGTH: int = 2048
    MAX_CONTENT_PREVIEW_LENGTH: int = 100

    # Pagination and display
    MESSAGES_PER_PAGE: int = 25
    CURRENT_PAGE_DEFAULT: int = 0

    # Colors and styling
    BRAND_COLOR: str = "#76b900"

    # Asset paths
    USER_AVATAR_PATH: str = "/app/assets/user.png"
    ASSISTANT_AVATAR_PATH: str = "/app/assets/nvidia.png"


@dataclass
class SessionConfig:
    """Session state management configuration"""

    # Image storage limits
    MAX_IMAGES_IN_SESSION: int = 50

    # Session cleanup thresholds
    CLEANUP_INTERVAL_MESSAGES: int = 100

    # Image ID generation
    IMAGE_ID_PREFIX: str = "img_"


@dataclass
class FileProcessingConfig:
    """File processing configuration"""

    # PDF processing
    PDF_PROCESSING_TIMEOUT: int = 60
    PDF_TEMP_FILE_SUFFIX: str = ".pdf"

    # File size limits (in bytes)
    MAX_PDF_SIZE: int = 50 * 1024 * 1024  # 50MB

    # Supported file types
    SUPPORTED_PDF_TYPES: List[str] = field(default_factory=lambda: ['pdf'])


@dataclass
class ToolContextConfig:
    """Tool context extraction configuration"""

    # Context display limits
    MAX_PAGES_IN_CONTEXT: int = 3
    PREVIEW_TEXT_LENGTH: int = 500

    # Context formatting
    CONTEXT_SEPARATOR: str = "\n\n---\n\n"
    CONTEXT_TRUNCATION_SUFFIX: str = "..."


@dataclass
class LLMConfig:
    """LLM service configuration"""

    # Default model parameters
    DEFAULT_TEMPERATURE: float = 0.6
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_FREQUENCY_PENALTY: float = 0.0
    DEFAULT_PRESENCE_PENALTY: float = 0.0

    # Context and token limits
    MAX_CONTEXT_TURNS: int = 10
    SLIDING_WINDOW_MAX_TURNS: int = 6

    # Streaming and response
    STREAM_CHUNK_SIZE: int = 1024
    RESPONSE_TIMEOUT: int = 300  # 5 minutes


@dataclass
class ImageGenerationConfig:
    """Image generation configuration"""

    # Keywords for detecting image generation responses
    DETECTION_KEYWORDS: List[str] = field(
        default_factory=lambda: [
            "image_data",
            "enhanced_prompt",
            "original_prompt",
            "cfg_scale",
            "dimensions",
            "successfully generated",
            "image with cfg_scale",
        ]
    )

    # Image generation parameters
    DEFAULT_CFG_SCALE: float = 7.5
    DEFAULT_DIMENSIONS: str = "1024x1024"

    # Image storage
    IMAGE_FORMAT: str = "PNG"
    CAPTION_MAX_LENGTH: int = 200


@dataclass
class APIConfig:
    """API configuration and timeouts"""

    # Request timeouts
    DEFAULT_REQUEST_TIMEOUT: int = 30
    LLM_REQUEST_TIMEOUT: int = 300
    IMAGE_REQUEST_TIMEOUT: int = 120

    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    # Rate limiting
    REQUESTS_PER_MINUTE: int = 60


@dataclass
class DatabaseConfig:
    """Database and vector store configuration"""

    # Default database settings
    DEFAULT_COLLECTION_NAME: str = "milvus"
    DEFAULT_PARTITION_NAME: str = "milvus"
    DEFAULT_DB_NAME: str = "milvus"

    # Vector search parameters
    DEFAULT_TOP_K: int = 10
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.7


@dataclass
class EnvironmentConfig:
    """Environment variable configuration with defaults"""

    # Bot configuration
    BOT_TITLE: str = field(default_factory=lambda: os.getenv("BOT_TITLE", "Nano"))
    META_USER: str = field(default_factory=lambda: os.getenv("META_USER", "Brandon"))
    AUTH_USERNAME: str = field(default_factory=lambda: os.getenv("AUTH_USERNAME", "Brandon"))
    AUTH_KEY: str = field(default_factory=lambda: os.getenv("AUTH_KEY", "Brandon"))

    # Model endpoints and names
    FAST_LLM_MODEL_NAME: Optional[str] = field(default_factory=lambda: os.getenv("FAST_LLM_MODEL_NAME"))
    FAST_LLM_ENDPOINT: Optional[str] = field(default_factory=lambda: os.getenv("FAST_LLM_ENDPOINT"))
    LLM_MODEL_NAME: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME"))
    LLM_ENDPOINT: Optional[str] = field(default_factory=lambda: os.getenv("LLM_ENDPOINT"))
    INTELLIGENT_LLM_ENDPOINT: Optional[str] = field(default_factory=lambda: os.getenv("INTELLIGENT_LLM_ENDPOINT"))
    INTELLIGENT_LLM_MODEL_NAME: Optional[str] = field(default_factory=lambda: os.getenv("INTELLIGENT_LLM_MODEL_NAME"))

    # API keys
    NVIDIA_API_KEY: str = field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", "None"))
    TAVILY_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    WEATHER_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("WEATHER_API_KEY"))
    OPENAI_API_TYPE: str = field(default_factory=lambda: os.getenv("OPENAI_API_TYPE", "openai"))

    # Embedding configuration
    EMBEDDING_ENDPOINT: Optional[str] = field(default_factory=lambda: os.getenv("EMBEDDING_ENDPOINT"))
    EMBEDDING_MODEL: Optional[str] = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL"))

    # Database configuration
    DATABASE_URL: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    COLLECTION_NAME: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "milvus"))
    PARTITION_NAME: str = field(default_factory=lambda: os.getenv("PARTITION_NAME", "milvus"))
    DEFAULT_DB: str = field(default_factory=lambda: os.getenv("DEFAULT_DB", "milvus"))

    # Reranker configuration
    RERANKER_ENDPOINT: Optional[str] = field(default_factory=lambda: os.getenv("RERANKER_ENDPOINT"))
    RERANKER_MODEL: Optional[str] = field(default_factory=lambda: os.getenv("RERANKER_MODEL"))

    # Image generation
    IMAGE_ENDPOINT: Optional[str] = field(default_factory=lambda: os.getenv("IMAGE_ENDPOINT"))

    # PDF processing
    NVINGEST_ENDPOINT: Optional[str] = field(default_factory=lambda: os.getenv("NVINGEST_ENDPOINT"))

    def validate_required_env_vars(self) -> List[str]:
        """
        Validate that required environment variables are set

        Returns:
            List of missing required environment variables
        """
        required_vars = [
            ('FAST_LLM_MODEL_NAME', self.FAST_LLM_MODEL_NAME),
            ('FAST_LLM_ENDPOINT', self.FAST_LLM_ENDPOINT),
            ('NVIDIA_API_KEY', self.NVIDIA_API_KEY),
        ]

        missing = []
        for var_name, var_value in required_vars:
            if not var_value or var_value == "None":
                missing.append(var_name)

        return missing


class AppConfig:
    """
    Centralized application configuration

    This class provides a single point of access to all configuration values
    across the application, replacing scattered constants and environment variables.
    """

    def __init__(self):
        """Initialize all configuration sections"""
        self.ui = UIConfig()
        self.session = SessionConfig()
        self.file_processing = FileProcessingConfig()
        self.tool_context = ToolContextConfig()
        self.llm = LLMConfig()
        self.image_generation = ImageGenerationConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.env = EnvironmentConfig()

        # Validate environment variables
        missing_vars = self.env.validate_required_env_vars()
        if missing_vars:
            logging.warning(f"Missing required environment variables: {', '.join(missing_vars)}")

    def get_llm_parameters(self) -> Dict[str, Any]:
        """
        Get standard LLM parameters for API calls

        Returns:
            Dictionary of LLM parameters
        """
        return {
            "temperature": self.llm.DEFAULT_TEMPERATURE,
            "top_p": self.llm.DEFAULT_TOP_P,
            "frequency_penalty": self.llm.DEFAULT_FREQUENCY_PENALTY,
            "presence_penalty": self.llm.DEFAULT_PRESENCE_PENALTY,
        }

    def get_avatar_config(self) -> Dict[str, str]:
        """
        Get avatar configuration for UI

        Returns:
            Dictionary with user and assistant avatar paths
        """
        return {
            "user_avatar": self.ui.USER_AVATAR_PATH,
            "assistant_avatar": self.ui.ASSISTANT_AVATAR_PATH,
        }

    def get_api_timeout(self, endpoint_type: str = "default") -> int:
        """
        Get appropriate timeout for different API endpoints

        Args:
            endpoint_type: Type of endpoint ('llm', 'image', 'pdf', 'default')

        Returns:
            Timeout in seconds
        """
        timeout_map = {
            "llm": self.api.LLM_REQUEST_TIMEOUT,
            "image": self.api.IMAGE_REQUEST_TIMEOUT,
            "pdf": self.file_processing.PDF_PROCESSING_TIMEOUT,
            "default": self.api.DEFAULT_REQUEST_TIMEOUT,
        }
        return timeout_map.get(endpoint_type, self.api.DEFAULT_REQUEST_TIMEOUT)

    def get_file_processing_config(self) -> Dict[str, Any]:
        """
        Get file processing configuration

        Returns:
            Dictionary of file processing settings
        """
        return {
            "pdf_timeout": self.file_processing.PDF_PROCESSING_TIMEOUT,
            "max_pdf_size": self.file_processing.MAX_PDF_SIZE,
            "supported_types": self.file_processing.SUPPORTED_PDF_TYPES,
            "temp_suffix": self.file_processing.PDF_TEMP_FILE_SUFFIX,
        }

    def get_context_config(self) -> Dict[str, Any]:
        """
        Get tool context configuration

        Returns:
            Dictionary of context settings
        """
        return {
            "max_pages": self.tool_context.MAX_PAGES_IN_CONTEXT,
            "preview_length": self.tool_context.PREVIEW_TEXT_LENGTH,
            "separator": self.tool_context.CONTEXT_SEPARATOR,
            "truncation_suffix": self.tool_context.CONTEXT_TRUNCATION_SUFFIX,
        }

    def validate_environment(self) -> None:
        """
        Validate that critical environment variables are properly configured

        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        missing_vars = self.env.validate_required_env_vars()
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Additional validation for specific endpoints
        if self.env.NVINGEST_ENDPOINT and not self.env.NVINGEST_ENDPOINT.startswith(('http://', 'https://')):
            raise ValueError("NVINGEST_ENDPOINT must be a valid HTTP/HTTPS URL")

        logging.info("Environment configuration validation completed successfully")


# Global configuration instance
# Use this throughout the application instead of scattered constants
config = AppConfig()


# Legacy compatibility - maintain existing interface
# TODO: Remove these after migration is complete
def get_config():
    """Legacy function for backward compatibility"""
    return config


# Export commonly used configurations for easy access
UI_CONFIG = config.ui
SESSION_CONFIG = config.session
FILE_CONFIG = config.file_processing
LLM_CONFIG = config.llm
IMAGE_CONFIG = config.image_generation
API_CONFIG = config.api
ENV_CONFIG = config.env


# Configure logging with centralized format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
