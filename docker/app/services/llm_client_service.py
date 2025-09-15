"""
LLM Client Service

This service provides the appropriate LLM client based on the requested type.
It ensures tools get the correct client for their configured LLM type.
"""

from typing import Literal, Optional

import httpx
from models.chat_config import ChatConfig
from openai import AsyncOpenAI, OpenAI

from utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMClientService:
    """Service for providing LLM clients based on type"""

    _instance: Optional["LLMClientService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config: Optional[ChatConfig] = None
        self._clients = {}
        self._async_clients = {}
        self._initialized = True
        logger.debug("LLM Client Service instance created")

    def initialize(self, config: ChatConfig):
        """
        Initialize the service with configuration

        Args:
            config: Chat configuration
        """
        # Skip if already initialized with same config
        if self._config is not None:
            logger.debug(
                "LLM Client Service already initialized, skipping"
                " re-initialization"
            )
            return

        self._config = config
        self._clients = {}  # Clear any cached clients
        self._async_clients = {}  # Clear any cached async clients
        logger.info("LLM Client Service initialized")

    def _create_async_http_client(self) -> httpx.AsyncClient:
        """Create an async HTTP client configured for concurrent requests"""
        return httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20,  # Keep connections alive
                max_connections=100,  # Allow many concurrent connections
                keepalive_expiry=30,  # Keep connections alive for 30 seconds
            ),
            timeout=httpx.Timeout(
                connect=30.0,  # 30 seconds to connect
                read=600.0,  # 10 minutes to read response (for long LLM calls)
                write=30.0,  # 30 seconds to write request
                pool=10.0,  # 10 seconds to get connection from pool
            ),
        )

    def get_client(
        self, llm_type: Literal["fast", "llm", "intelligent", "vlm"]
    ) -> OpenAI:
        """
        Get an OpenAI client for the specified LLM type

        Args:
            llm_type: The type of LLM client to get

        Returns:
            OpenAI client configured for the specified type

        Raises:
            ValueError: If service not initialized or invalid type
        """
        if not self._config:
            # Try to initialize with default config
            try:
                self._config = ChatConfig.from_environment()
            except Exception as e:
                raise ValueError(
                    "LLM Client Service not initialized and failed to"
                    f" auto-initialize: {e}"
                )

        # Check cache first
        if llm_type in self._clients:
            return self._clients[llm_type]

        # Create new client based on type
        try:
            if llm_type == "fast":
                client = OpenAI(
                    api_key=self._config.fast_llm_api_key,
                    base_url=self._config.fast_llm_endpoint,
                )
            elif llm_type == "llm":
                client = OpenAI(
                    api_key=self._config.llm_api_key,
                    base_url=self._config.llm_endpoint,
                )
            elif llm_type == "intelligent":
                client = OpenAI(
                    api_key=self._config.intelligent_llm_api_key,
                    base_url=self._config.intelligent_llm_endpoint,
                )
            elif llm_type == "vlm":
                client = OpenAI(
                    api_key=self._config.vlm_api_key,
                    base_url=self._config.vlm_endpoint,
                )
            else:
                raise ValueError(f"Invalid LLM type: {llm_type}")

            # Cache the client
            self._clients[llm_type] = client
            logger.debug(f"Created {llm_type} LLM client")

            return client

        except Exception as e:
            logger.error(f"Failed to create {llm_type} client: {e}")
            raise

    def get_async_client(
        self, llm_type: Literal["fast", "llm", "intelligent", "vlm"]
    ) -> AsyncOpenAI:
        """
        Get an async OpenAI client for the specified LLM type

        Args:
            llm_type: The type of LLM client to get

        Returns:
            AsyncOpenAI client configured for the specified type

        Raises:
            ValueError: If service not initialized or invalid type
        """
        if not self._config:
            # Try to initialize with default config
            try:
                self._config = ChatConfig.from_environment()
            except Exception as e:
                raise ValueError(
                    "LLM Client Service not initialized and failed to"
                    f" auto-initialize: {e}"
                )

        # Check cache first
        if llm_type in self._async_clients:
            return self._async_clients[llm_type]

        # Create new async client based on type with optimized HTTP client
        try:
            http_client = self._create_async_http_client()

            if llm_type == "fast":
                client = AsyncOpenAI(
                    api_key=self._config.fast_llm_api_key,
                    base_url=self._config.fast_llm_endpoint,
                    http_client=http_client,
                )
            elif llm_type == "llm":
                client = AsyncOpenAI(
                    api_key=self._config.llm_api_key,
                    base_url=self._config.llm_endpoint,
                    http_client=http_client,
                )
            elif llm_type == "intelligent":
                client = AsyncOpenAI(
                    api_key=self._config.intelligent_llm_api_key,
                    base_url=self._config.intelligent_llm_endpoint,
                    http_client=http_client,
                )
            elif llm_type == "vlm":
                client = AsyncOpenAI(
                    api_key=self._config.vlm_api_key,
                    base_url=self._config.vlm_endpoint,
                    http_client=http_client,
                )
            else:
                raise ValueError(f"Invalid LLM type: {llm_type}")

            # Cache the client
            self._async_clients[llm_type] = client
            logger.debug(
                f"Created {llm_type} async LLM client with concurrent"
                " connection support"
            )

            return client

        except Exception as e:
            logger.error(f"Failed to create async {llm_type} client: {e}")
            raise

    def get_model_name(
        self, llm_type: Literal["fast", "llm", "intelligent", "vlm"]
    ) -> str:
        """
        Get the model name for the specified LLM type

        Args:
            llm_type: The type of LLM

        Returns:
            Model name string
        """
        if not self._config:
            self._config = ChatConfig.from_environment()

        if llm_type == "fast":
            return self._config.fast_llm_model_name
        elif llm_type == "llm":
            return self._config.llm_model_name
        elif llm_type == "intelligent":
            return self._config.intelligent_llm_model_name
        elif llm_type == "vlm":
            return self._config.vlm_model_name
        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")


# Global instance
llm_client_service = LLMClientService()
