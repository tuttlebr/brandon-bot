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
    SPINNER_ICONS: List[str] = field(
        default_factory=lambda: ["🤖", "🧠", "🤔", "🤓", "⚡"]
    )

    # Display limits and formatting
    MAX_PROMPT_DISPLAY_LENGTH: int = 4096

    # Pagination and display
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

    # PDF storage limits
    MAX_PDFS_IN_SESSION: int = 3

    # Image ID generation
    IMAGE_ID_PREFIX: str = "img_"


@dataclass
class FileProcessingConfig:
    """File processing configuration"""

    # PDF processing
    PDF_PROCESSING_TIMEOUT: int = 6000
    PDF_TEMP_FILE_SUFFIX: str = ".pdf"

    # File size limits (in bytes)
    MAX_PDF_SIZE: int = 16 * 1024 * 1024  # 16MB
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB

    # Supported file types
    SUPPORTED_PDF_TYPES: List[str] = field(default_factory=lambda: ['pdf'])
    SUPPORTED_IMAGE_TYPES: List[str] = field(
        default_factory=lambda: ['png', 'jpg', 'jpeg']
    )

    # PDF Summarization settings
    PDF_SUMMARIZATION_THRESHOLD: int = 10  # Number of pages to trigger summarization
    PDF_SUMMARIZATION_BATCH_SIZE: int = (
        10  # Pages per batch for summarization (increased from 5)
    )
    PDF_SUMMARY_MAX_LENGTH: int = 800  # Max words per page summary (increased from 500)
    PDF_SUMMARIZATION_ENABLED: bool = (
        True  # Enabled - summarization should be user-driven
    )
    PDF_SUMMARIZATION_USE_ASYNC: bool = (
        True  # Use async (True) or sync (False) processing
    )

    # PDF Batch Processing settings
    PDF_BATCH_PROCESSING_THRESHOLD: int = (
        50  # Number of pages to trigger batch processing
    )
    PDF_PAGES_PER_BATCH: int = (
        50  # Maximum pages to process per batch (increased from 20)
    )
    PDF_CONTEXT_MAX_PAGES: int = (
        100  # Maximum pages to include in context at once (increased from 30)
    )
    PDF_CONTEXT_MAX_CHARS: int = (
        100000  # Maximum characters per context injection (increased from 100000)
    )


@dataclass
class ToolContextConfig:
    """Tool context extraction configuration"""

    # Context display limits
    MAX_PAGES_IN_CONTEXT: int = 3
    PREVIEW_TEXT_LENGTH: int = 4096

    # Context formatting
    CONTEXT_SEPARATOR: str = "\n\n---\n\n"
    CONTEXT_TRUNCATION_SUFFIX: str = "..."


@dataclass
class LLMConfig:
    """LLM service configuration"""

    # Default model parameters
    DEFAULT_TEMPERATURE: float = 0.4
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_FREQUENCY_PENALTY: float = 0.0
    DEFAULT_PRESENCE_PENALTY: float = 0.0
    DEFAULT_MAX_TOKENS: int = 4096

    # Context and token limits
    SLIDING_WINDOW_MAX_TURNS: int = 20  # Increased from 6 to prevent context loss
    MAX_CONTEXT_TOKENS: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "128000"))
    )  # Maximum context length for LLM (tokens)

    # Conversation context injection
    AUTO_INJECT_CONVERSATION_CONTEXT: bool = (
        True  # Automatically inject conversation context
    )
    MIN_TURNS_FOR_CONTEXT_INJECTION: int = 1  # Minimum turns before injecting context


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


@dataclass
class APIConfig:
    """API configuration and timeouts"""

    # Request timeouts
    DEFAULT_REQUEST_TIMEOUT: int = 3600
    LLM_REQUEST_TIMEOUT: int = 3600
    IMAGE_REQUEST_TIMEOUT: int = 3600


@dataclass
class SystemConfig:
    """System-level configuration"""

    # Logging configuration
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    SUPPRESS_STREAMLIT_WARNINGS: bool = field(
        default_factory=lambda: os.getenv("SUPPRESS_STREAMLIT_WARNINGS", "true").lower()
        == "true"
    )


@dataclass
class EnvironmentConfig:
    """Environment variable configuration with defaults"""

    # Bot configuration
    BOT_TITLE: str = field(default_factory=lambda: os.getenv("BOT_TITLE", "Nano"))
    META_USER: str = field(default_factory=lambda: os.getenv("META_USER", "Human"))

    # Model endpoints and names
    FAST_LLM_MODEL_NAME: Optional[str] = field(
        default_factory=lambda: os.getenv("FAST_LLM_MODEL_NAME")
    )
    FAST_LLM_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("FAST_LLM_ENDPOINT")
    )
    LLM_MODEL_NAME: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME")
    )
    LLM_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_ENDPOINT")
    )
    INTELLIGENT_LLM_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("INTELLIGENT_LLM_ENDPOINT")
    )
    INTELLIGENT_LLM_MODEL_NAME: Optional[str] = field(
        default_factory=lambda: os.getenv("INTELLIGENT_LLM_MODEL_NAME")
    )
    VLM_MODEL_NAME: Optional[str] = field(
        default_factory=lambda: os.getenv("VLM_MODEL_NAME")
    )
    VLM_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("VLM_ENDPOINT")
    )

    # API keys
    NVIDIA_API_KEY: str = field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY", "None")
    )
    TAVILY_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    # Embedding configuration
    EMBEDDING_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_ENDPOINT")
    )
    EMBEDDING_MODEL: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL")
    )

    # Database configuration
    DATABASE_URL: Optional[str] = field(
        default_factory=lambda: os.getenv("DATABASE_URL")
    )
    COLLECTION_NAME: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "milvus")
    )
    PARTITION_NAME: str = field(
        default_factory=lambda: os.getenv("PARTITION_NAME", "milvus")
    )
    DEFAULT_DB: str = field(default_factory=lambda: os.getenv("DEFAULT_DB", "milvus"))

    # Reranker configuration
    RERANKER_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("RERANKER_ENDPOINT")
    )
    RERANKER_MODEL: Optional[str] = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL")
    )

    # Image generation
    IMAGE_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("IMAGE_ENDPOINT")
    )

    # PDF processing
    NVINGEST_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("NVINGEST_ENDPOINT")
    )

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
        self.system = SystemConfig()
        self.env = EnvironmentConfig()

        # Validate environment variables
        missing_vars = self.env.validate_required_env_vars()
        if missing_vars:
            logging.warning(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

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
            "max_tokens": self.llm.DEFAULT_MAX_TOKENS,
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

    def validate_environment(self) -> None:
        """
        Validate that critical environment variables are properly configured

        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        missing_vars = self.env.validate_required_env_vars()
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Additional validation for specific endpoints
        if self.env.NVINGEST_ENDPOINT and not self.env.NVINGEST_ENDPOINT.startswith(
            ('http://', 'https://')
        ):
            raise ValueError("NVINGEST_ENDPOINT must be a valid HTTP/HTTPS URL")

        logging.info("Environment configuration validation completed successfully")


# Global configuration instance
# Use this throughout the application instead of scattered constants
config = AppConfig()

# Ensure log directory exists before configuring logging
import os

log_dir = "/tmp/chatbot_storage"
os.makedirs(log_dir, exist_ok=True)

# Configure logging with centralized format
logging.basicConfig(
    level=getattr(logging, config.system.LOG_LEVEL.upper()),
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "chatbot.log"), mode="a"),
    ],
)
