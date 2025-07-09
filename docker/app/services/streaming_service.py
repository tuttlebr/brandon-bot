"""
Streaming Service

This service handles streaming responses from LLM models with
a simplified approach that avoids complex threading patterns.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from models.chat_config import ChatConfig
from services.llm_client_service import llm_client_service
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
        # Initialize llm_client_service if not already done
        llm_client_service.initialize(config_obj)

    def get_client(self, model_type: str, async_client: bool = False):
        """
        Get the appropriate client for the model type

        Args:
            model_type: Type of model ("fast", "llm", "intelligent")
            async_client: Whether to return async client

        Returns:
            OpenAI client instance
        """
        if async_client:
            return llm_client_service.get_async_client(model_type)
        else:
            return llm_client_service.get_client(model_type)

    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        model_type: str,
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

        logger.debug(f"Streaming with model_type: {model_type}, model: {model}")

        try:
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "stream": True,  # Enable streaming for async iteration
                **config.get_llm_parameters(),
                **kwargs,
            }

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = "auto"
                api_params["parallel_tool_calls"] = True

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
        model_type: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs,
    ) -> Any:
        """
        Get non-streaming completion from LLM

        Args:
            messages: Conversation messages
            model: Model name
            model_type: Type of model to use
            tools: Optional tool definitions
            tool_choice: How to handle tool selection ("auto", "none", or specific tool)
            **kwargs: Additional parameters for the API

        Returns:
            API response object
        """
        client = self.get_client(model_type, async_client=False)

        logger.debug(
            f"Sync completion with model_type: {model_type}, model: {model}, tool_choice: {tool_choice}"
        )

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
                api_params["tool_choice"] = tool_choice
                api_params["parallel_tool_calls"] = True
                # Generates more concise responses without extended chain-of-thought or thinking tokens.
                api_params["temperature"] = 0.0
                api_params["max_tokens"] = 200
                del api_params["top_p"]
                del api_params["frequency_penalty"]
                del api_params["presence_penalty"]

            return client.chat.completions.create(**api_params)

        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise StreamingError(f"Failed to get completion: {e}")

    async def simple_stream(
        self, prompt: str, model_type: str
    ) -> AsyncGenerator[str, None]:
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
        return llm_client_service.get_model_name(model_type)
