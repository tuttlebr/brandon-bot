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

    # Display limits and formatting
    MAX_PROMPT_DISPLAY_LENGTH: int = 4096

    # Pagination and display
    CURRENT_PAGE_DEFAULT: int = 0

    # Colors and styling
    BRAND_COLOR: str = "#76b900"

    # Asset paths
    USER_AVATAR_PATH: str = "/app/assets/nvidia.png"
    ASSISTANT_AVATAR_PATH: str = "/app/assets/user.png"


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
    MAX_PDF_SIZE: int = 100 * 1024 * 1024
    MAX_IMAGE_SIZE: int = 20 * 1024 * 1024

    # Supported file types
    SUPPORTED_PDF_TYPES: List[str] = field(default_factory=lambda: ["pdf"])
    SUPPORTED_IMAGE_TYPES: List[str] = field(
        default_factory=lambda: ["png", "jpg", "jpeg"]
    )

    # PDF Summarization settings
    PDF_SUMMARIZATION_THRESHOLD: int = (
        10  # Number of pages to trigger summarization
    )
    PDF_SUMMARIZATION_BATCH_SIZE: int = (
        10  # Pages per batch for summarization (increased from 5)
    )
    PDF_SUMMARY_MAX_LENGTH: int = 800
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

    # PDF Chunking settings
    PDF_CHUNK_SIZE: int = 4096  # Target chunk size in characters
    PDF_CHUNK_OVERLAP: int = 0  # Overlap between chunks
    PDF_MIN_CHUNK_SIZE: int = 500  # Minimum chunk size
    PDF_MAX_CHUNKS_PER_QUERY: int = 10  # Maximum chunks to retrieve per query
    PDF_SIMILARITY_THRESHOLD: float = (
        3.0  # L2 distance threshold for similarity
    )

    # PDF Upload behavior
    PDF_REUPLOAD_EXISTING: bool = field(
        default_factory=lambda: os.getenv(
            "PDF_REUPLOAD_EXISTING", "true"
        ).lower()
        == "true"
    )  # Whether to re-upload existing PDFs (True=delete & re-upload, False=skip existing)
    # Set PDF_REUPLOAD_EXISTING=false to skip re-uploading existing PDFs and use existing chunks


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
    DEFAULT_TEMPERATURE: float = 0.25
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_FREQUENCY_PENALTY: float = 0.002
    DEFAULT_PRESENCE_PENALTY: float = 0.9
    DEFAULT_MAX_TOKENS: int = 65536

    # Context and token limits
    SLIDING_WINDOW_MAX_TURNS: int = 20
    MAX_CONTEXT_TOKENS: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "116000"))
    )  # Maximum context length for LLM (tokens)
    MAX_TOOL_RESPONSE_TOKENS: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_TOOL_RESPONSE_TOKENS", "16000")
        )
    )  # Maximum tokens for individual tool responses

    # Conversation context injection
    AUTO_INJECT_CONVERSATION_CONTEXT: bool = (
        True  # Automatically inject conversation context
    )
    MIN_TURNS_FOR_CONTEXT_INJECTION: int = (
        1  # Minimum turns before injecting context
    )


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
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    SUPPRESS_STREAMLIT_WARNINGS: bool = field(
        default_factory=lambda: os.getenv(
            "SUPPRESS_STREAMLIT_WARNINGS", "true"
        ).lower()
        == "true"
    )


@dataclass
class ToolConfig:
    """Tool availability configuration for A/B testing and feature toggling

    Tools not listed here or set to False will not be available to the LLM.
    This allows easy enabling/disabling of tools for testing purposes.
    """

    # Tool availability mapping - tool_name: enabled
    # Default configuration enables all standard tools
    ENABLED_TOOLS: Dict[str, bool] = field(
        default_factory=lambda: {
            "text_assistant": True,
            "conversation_context": True,
            "extract_web_content": True,
            "serpapi_internet_search": True,
            "serpapi_news_search": True,
            "retrieval_search": True,
            "pdf_assistant": True,
            "analyze_image": True,
            "generate_image": True,
            "context_generation": True,
            "get_weather": True,
            "generalist_conversation": True,
            "deep_researcher": True,
        }
    )

    # Use enhanced tool descriptions from centralized source
    USE_ENHANCED_DESCRIPTIONS: bool = field(
        default_factory=lambda: os.getenv(
            "USE_ENHANCED_TOOL_DESCRIPTIONS", "true"
        ).lower()
        == "true"
    )

    def __post_init__(self):
        """Load tool configuration from environment or config file"""
        # Allow environment variable override for each tool
        # Format: TOOL_ENABLE_<TOOL_NAME>=true/false
        for tool_name in list(self.ENABLED_TOOLS.keys()):
            env_var = f"TOOL_ENABLE_{tool_name.upper()}"
            env_value = os.getenv(env_var)
            if env_value is not None:
                self.ENABLED_TOOLS[tool_name] = env_value.lower() == "true"
                logging.info(
                    "Tool '%s' enabled: %s (from %s)",
                    tool_name,
                    self.ENABLED_TOOLS[tool_name],
                    env_var,
                )

        # Load from external config file if provided
        config_file = os.getenv("TOOL_CONFIG_FILE")
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)

    def _load_from_file(self, config_file: str):
        """Load tool configuration from JSON or YAML file"""
        import json
        import yaml

        try:
            with open(config_file, "r") as f:
                if config_file.endswith(".json"):
                    config_data = json.load(f)
                elif config_file.endswith((".yaml", ".yml")):
                    config_data = yaml.safe_load(f)
                else:
                    logging.warning(
                        "Unsupported config file format: %s", config_file
                    )
                    return

                # Update enabled tools from file
                if "enabled_tools" in config_data:
                    for tool_name, enabled in config_data[
                        "enabled_tools"
                    ].items():
                        self.ENABLED_TOOLS[tool_name] = bool(enabled)
                        logging.info(
                            "Tool '%s' enabled: %s (from %s)",
                            tool_name,
                            enabled,
                            config_file,
                        )

        except Exception as e:
            logging.error(
                "Failed to load tool config from %s: %s", config_file, e
            )

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled"""
        return self.ENABLED_TOOLS.get(tool_name, False)

    def get_enabled_tools(self) -> List[str]:
        """Get list of all enabled tools"""
        return [
            name for name, enabled in self.ENABLED_TOOLS.items() if enabled
        ]

    def set_tool_enabled(self, tool_name: str, enabled: bool):
        """Enable or disable a tool at runtime"""
        self.ENABLED_TOOLS[tool_name] = enabled
        logging.info("Tool '%s' dynamically set to: %s", tool_name, enabled)


@dataclass
class EnvironmentConfig:
    """Environment variable configuration with defaults"""

    # Bot configuration
    BOT_TITLE: str = field(
        default_factory=lambda: os.getenv("BOT_TITLE", "Nano")
    )
    META_USER: str = field(
        default_factory=lambda: os.getenv("META_USER", "Human")
    )

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

    # API keys - backward compatible with single NVIDIA_API_KEY
    NVIDIA_API_KEY: str = field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY", "None")
    )

    # Per-model API keys (fall back to NVIDIA_API_KEY if not specified)
    FAST_LLM_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "FAST_LLM_API_KEY", os.getenv("NVIDIA_API_KEY", "None")
        )
    )
    LLM_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "LLM_API_KEY", os.getenv("NVIDIA_API_KEY", "None")
        )
    )
    INTELLIGENT_LLM_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "INTELLIGENT_LLM_API_KEY", os.getenv("NVIDIA_API_KEY", "None")
        )
    )
    VLM_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "VLM_API_KEY", os.getenv("NVIDIA_API_KEY", "None")
        )
    )

    # Other API keys
    SERPAPI_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("SERPAPI_KEY")
    )

    # Embedding configuration
    EMBEDDING_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_ENDPOINT")
    )
    EMBEDDING_MODEL: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL")
    )
    EMBEDDING_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_API_KEY", os.getenv("NVIDIA_API_KEY", "None")
        )
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
    DEFAULT_DB: str = field(
        default_factory=lambda: os.getenv("DEFAULT_DB", "milvus")
    )

    # Reranker configuration
    RERANKER_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("RERANKER_ENDPOINT")
    )
    RERANKER_MODEL: Optional[str] = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL")
    )
    RERANKER_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "RERANKER_API_KEY", os.getenv("NVIDIA_API_KEY", "None")
        )
    )

    # Image generation
    IMAGE_ENDPOINT: Optional[str] = field(
        default_factory=lambda: os.getenv("IMAGE_ENDPOINT")
    )
    IMAGE_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "IMAGE_API_KEY", os.getenv("NVIDIA_API_KEY", "None")
        )
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
            ("FAST_LLM_MODEL_NAME", self.FAST_LLM_MODEL_NAME),
            ("FAST_LLM_ENDPOINT", self.FAST_LLM_ENDPOINT),
            # Only require NVIDIA_API_KEY if no individual API keys are set
            (
                "NVIDIA_API_KEY or individual model API keys",
                (
                    self.NVIDIA_API_KEY
                    if (
                        not self.FAST_LLM_API_KEY
                        or self.FAST_LLM_API_KEY == "None"
                        or not self.LLM_API_KEY
                        or self.LLM_API_KEY == "None"
                    )
                    else "Set"
                ),
            ),
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
        self.tools = ToolConfig()

        # Validate environment variables
        missing_vars = self.env.validate_required_env_vars()
        if missing_vars:
            logging.warning(
                "Missing required environment variables: %s",
                ", ".join(missing_vars),
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
                "Missing required environment variables: %s",
                ", ".join(missing_vars),
            )

        # Additional validation for specific endpoints
        if (
            self.env.NVINGEST_ENDPOINT
            and not self.env.NVINGEST_ENDPOINT.startswith(
                ("http://", "https://")
            )
        ):
            raise ValueError(
                "NVINGEST_ENDPOINT must be a valid HTTP/HTTPS URL"
            )

        logging.info(
            "Environment configuration validation completed successfully"
        )


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
