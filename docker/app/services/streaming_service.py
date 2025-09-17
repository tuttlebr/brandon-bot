"""
Streaming Service

This service handles streaming responses from LLM models with
a simplified approach that avoids complex threading patterns.
"""

from typing import Any, AsyncGenerator, Dict, List, Optional

from models.chat_config import ChatConfig
from services.llm_client_service import llm_client_service
from utils.config import config
from utils.exceptions import StreamingError

from utils.logging_config import get_logger

logger = get_logger(__name__)


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
        extract_tool_calls: bool = False,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream completion from LLM

        Args:
            messages: Conversation messages
            model: Model name
            model_type: Type of model to use
            tools: Optional tool definitions
            extract_tool_calls: Whether to extract tool calls during streaming
            **kwargs: Additional parameters for the API

        Yields:
            Response chunks (with thinking and tool calls filtered if extract_tool_calls=True)
        """
        client = self.get_client(model_type, async_client=True)

        logger.debug(
            "Streaming with model_type: %s, model: %s, extract_tool_calls: %s",
            model_type,
            model,
            extract_tool_calls,
        )

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
                del api_params["temperature"]
                del api_params["top_p"]
                del api_params["frequency_penalty"]
                del api_params["presence_penalty"]

            # Create streaming response
            response = await client.chat.completions.create(**api_params)

            # Initialize filter if tool call extraction is requested
            complete_filter = None
            if extract_tool_calls:
                from utils.text_processing import StreamingCompleteFilter

                complete_filter = StreamingCompleteFilter(model_name=model)
                # Store reference for later tool call extraction
                self._current_filter = complete_filter

            # Process stream
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content

                    if complete_filter:
                        # Filter thinking tags and extract tool calls
                        filtered_content = complete_filter.process_chunk(
                            chunk_content
                        )
                        if filtered_content:
                            yield filtered_content
                    else:
                        # Pass through without filtering
                        yield chunk_content

            # Flush any remaining content if using filter
            if complete_filter:
                final_content = complete_filter.flush()
                if final_content:
                    yield final_content

        except Exception as e:
            logger.error("Streaming error: %s", e)
            # Check for common connection errors
            error_str = str(e).lower()
            if "connection" in error_str or "connect" in error_str:
                raise StreamingError(
                    "Connection to LLM service failed. "
                    "The service may be temporarily unavailable."
                ) from e
            elif "timeout" in error_str or "timed out" in error_str:
                raise StreamingError(
                    "Request timed out. The response is taking too long."
                ) from e
            else:
                raise StreamingError(f"Failed to stream response: {e}") from e

    async def sync_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        model_type: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        stream: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        """
        Get completion from LLM with optional streaming

        This method can use streaming or non-streaming mode. When tools are
        provided, it defaults to non-streaming to ensure proper tool call
        parsing. Otherwise, it defaults to streaming for better performance.

        Args:
            messages: Conversation messages
            model: Model name
            model_type: Type of model to use
            tools: Optional tool definitions
            tool_choice: How to handle tool selection ("auto", None, or
                         "required")
            stream: Force streaming on/off. If None, auto-decides based on
                    whether tools are provided
            **kwargs: Additional parameters for the API

        Returns:
            API response object with content and tool calls
        """
        client = self.get_client(model_type, async_client=True)

        # Auto-decide streaming mode if not specified
        if stream is None:
            # Default to non-streaming when tools are involved
            stream = tools is None

        logger.debug(
            "Completion - model_type: %s, model: %s, tool_choice: %s, "
            "streaming: %s",
            model_type,
            model,
            tool_choice,
            stream,
        )

        try:
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "stream": stream,
                **config.get_llm_parameters(),
                **kwargs,
            }

            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = tool_choice
                api_params["parallel_tool_calls"] = True
                del api_params["temperature"]
                del api_params["top_p"]
                del api_params["frequency_penalty"]
                del api_params["presence_penalty"]

            # Handle non-streaming mode
            if not stream:
                response = await client.chat.completions.create(**api_params)
                return response

            # Handle streaming mode
            response_stream = await client.chat.completions.create(
                **api_params
            )

            # Collect chunks to build complete response
            collected_content = ""
            collected_tool_calls = {}
            finish_reason = None
            response_id = None
            response_model = model
            created_time = None

            async for chunk in response_stream:
                # Collect response metadata from first chunk
                if response_id is None and hasattr(chunk, "id"):
                    response_id = chunk.id
                if created_time is None and hasattr(chunk, "created"):
                    created_time = chunk.created
                if hasattr(chunk, "model"):
                    response_model = chunk.model

                if chunk.choices:
                    choice = chunk.choices[0]

                    # Collect content
                    if (
                        hasattr(choice.delta, "content")
                        and choice.delta.content
                    ):
                        collected_content += choice.delta.content

                    # Collect tool calls
                    if (
                        hasattr(choice.delta, "tool_calls")
                        and choice.delta.tool_calls
                    ):
                        for tool_call_delta in choice.delta.tool_calls:
                            # Handle index - might be None for single calls
                            idx = getattr(tool_call_delta, "index", 0)
                            if idx is None:
                                idx = 0

                            if idx not in collected_tool_calls:
                                collected_tool_calls[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            tc = collected_tool_calls[idx]

                            # Update ID if provided
                            if (
                                hasattr(tool_call_delta, "id")
                                and tool_call_delta.id
                            ):
                                tc["id"] = tool_call_delta.id

                            # Update function details
                            if hasattr(tool_call_delta, "function"):
                                func = tool_call_delta.function

                                # Update name if provided (only comes once)
                                if (
                                    hasattr(func, "name")
                                    and func.name is not None
                                ):
                                    tc["function"]["name"] = func.name

                                # Append arguments if provided
                                if (
                                    hasattr(func, "arguments")
                                    and func.arguments is not None
                                ):
                                    # Log each argument chunk for debugging
                                    logger.debug(
                                        "Tool call %d (%s) - appending "
                                        "arguments chunk: %s (length: %d)",
                                        idx,
                                        tc["function"]["name"] or "unknown",
                                        repr(func.arguments),
                                        len(tc["function"]["arguments"]),
                                    )
                                    tc["function"][
                                        "arguments"
                                    ] += func.arguments

                    # Get finish reason
                    if (
                        hasattr(choice, "finish_reason")
                        and choice.finish_reason
                    ):
                        finish_reason = choice.finish_reason

            # Build response object that mimics the non-streaming
            # response structure. This allows the parsing service
            # to work without modification
            from types import SimpleNamespace

            # Convert collected tool calls to list
            tool_calls_list = []
            for idx in sorted(collected_tool_calls.keys()):
                tc = collected_tool_calls[idx]
                # Log the collected arguments for debugging
                if tc["function"]["arguments"]:
                    logger.debug(
                        "Tool call %d (%s) arguments: %s",
                        idx,
                        tc["function"]["name"],
                        repr(tc["function"]["arguments"]),
                    )
                tool_call_obj = SimpleNamespace(
                    id=tc["id"],
                    type=tc["type"],
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                tool_calls_list.append(tool_call_obj)

            # Build message object
            message = SimpleNamespace(
                content=collected_content if collected_content else None,
                tool_calls=tool_calls_list if tool_calls_list else None,
            )

            # Build choice object
            choice = SimpleNamespace(
                index=0, message=message, finish_reason=finish_reason
            )

            # Build response object
            response = SimpleNamespace(
                id=response_id,
                object="chat.completion",
                created=created_time,
                model=response_model,
                choices=[choice],
            )

            return response

        except Exception as e:
            mode = "streaming" if stream else "non-streaming"
            logger.error("%s completion error: %s", mode.capitalize(), e)
            raise StreamingError(
                f"Failed to get {mode} completion: {e}"
            ) from e

    def get_extracted_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Get tool calls extracted during streaming (if extract_tool_calls was enabled)

        Returns:
            List of extracted tool calls, empty if none extracted
        """
        if hasattr(self, "_current_filter") and self._current_filter:
            return self._current_filter.get_extracted_tool_calls()
        return []

    def clear_extracted_tool_calls(self):
        """Clear any extracted tool calls from the current filter"""
        if hasattr(self, "_current_filter") and self._current_filter:
            self._current_filter.clear_extracted_tool_calls()
