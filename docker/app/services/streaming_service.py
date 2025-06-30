"""
Streaming Service

This service handles streaming responses from LLM models with
a simplified approach that avoids complex threading patterns.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from models.chat_config import ChatConfig
from openai import AsyncOpenAI, OpenAI
from utils.config import config
from utils.exceptions import StreamingError

logger = logging.getLogger(__name__)


class StreamingService:
    """Service for handling streaming LLM responses"""

    def __init__(self, config_obj: ChatConfig):
        """
        Initialize the streaming service

        Args:
            config_obj: Application configuration
        """
        self.config = config_obj
        self._init_clients()

    def _init_clients(self):
        """Initialize OpenAI clients"""
        try:
            # Sync clients for backward compatibility
            self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.llm_endpoint)
            self.fast_client = OpenAI(api_key=self.config.api_key, base_url=self.config.fast_llm_endpoint)
            self.intelligent_client = OpenAI(
                api_key=self.config.api_key, base_url=self.config.intelligent_llm_endpoint
            )

            # Async clients for streaming
            self.async_client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.llm_endpoint)
            self.async_fast_client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.fast_llm_endpoint)
            self.async_intelligent_client = AsyncOpenAI(
                api_key=self.config.api_key, base_url=self.config.intelligent_llm_endpoint
            )
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise StreamingError(f"Client initialization failed: {e}")

    def get_client(self, model_type: str = "fast", async_client: bool = False):
        """
        Get the appropriate client for the model type

        Args:
            model_type: Type of model ("fast", "llm", "intelligent")
            async_client: Whether to return async client

        Returns:
            OpenAI client instance
        """
        client_map = {
            "fast": (self.fast_client, self.async_fast_client),
            "llm": (self.client, self.async_client),
            "intelligent": (self.intelligent_client, self.async_intelligent_client),
        }

        clients = client_map.get(model_type, client_map["fast"])
        return clients[1] if async_client else clients[0]

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        model_type: str = "fast",
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from LLM

        Args:
            messages: Conversation messages
            model: Model name
            model_type: Type of model to use
            tools: Optional tool definitions
            **kwargs: Additional parameters for the API

        Yields:
            Response chunks
        """
        client = self.get_client(model_type, async_client=True)

        try:
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "stream": True,
                **config.get_llm_parameters(),
                **kwargs,
            }

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "required"

            # Create streaming response
            response = await client.chat.completions.create(**api_params)

            # Process stream
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise StreamingError(f"Failed to stream response: {e}")

    def sync_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        model_type: str = "fast",
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """
        Get non-streaming completion from LLM

        Args:
            messages: Conversation messages
            model: Model name
            model_type: Type of model to use
            tools: Optional tool definitions
            **kwargs: Additional parameters for the API

        Returns:
            API response object
        """
        client = self.get_client(model_type, async_client=False)

        try:
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "stream": False,
                **config.get_llm_parameters(),
                **kwargs,
            }

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "required"

            return client.chat.completions.create(**api_params)

        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise StreamingError(f"Failed to get completion: {e}")

    async def simple_stream(self, prompt: str, model_type: str = "fast") -> AsyncGenerator[str, None]:
        """
        Simple streaming for basic prompts

        Args:
            prompt: User prompt
            model_type: Model type to use

        Yields:
            Response chunks
        """
        messages = [{"role": "user", "content": prompt}]
        model = self._get_model_name(model_type)

        async for chunk in self.stream_completion(messages, model, model_type):
            yield chunk

    def _get_model_name(self, model_type: str) -> str:
        """Get model name for the specified type"""
        model_map = {
            "fast": self.config.fast_llm_model_name,
            "llm": self.config.llm_model_name,
            "intelligent": self.config.intelligent_llm_model_name,
        }
        return model_map.get(model_type, self.config.fast_llm_model_name)
