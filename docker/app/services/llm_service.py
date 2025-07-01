"""
Refactored LLM Service

This is a simplified LLM service that orchestrates the focused services
for streaming, parsing, and tool execution.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from models.chat_config import ChatConfig
from services.response_parsing_service import ResponseParsingService
from services.streaming_service import StreamingService
from services.tool_execution_service import ToolExecutionService
from tools.registry import tool_registry
from tools.tool_llm_config import DEFAULT_LLM_TYPE
from utils.config import config
from utils.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class LLMService:
    """Simplified service for LLM interactions"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the LLM service with focused sub-services

        Args:
            config: Application configuration
        """
        self.config = config
        self.streaming_service = StreamingService(config)
        self.parsing_service = ResponseParsingService()
        self.tool_execution_service = ToolExecutionService(config)

        # For backward compatibility
        self.last_tool_responses = []

    async def generate_streaming_response(
        self, messages: List[Dict[str, Any]], model: str, model_type: str = DEFAULT_LLM_TYPE
    ) -> AsyncGenerator[str, str]:
        """
        Generate streaming response with tool support

        Args:
            messages: Conversation messages
            model: Model name
            model_type: Type of model to use

        Yields:
            Response chunks or tool results
        """
        try:
            # Apply sliding window to messages
            windowed_messages = self._apply_sliding_window(messages)

            # Get tool definitions
            tools = tool_registry.get_all_definitions()

            # First, get non-streaming response to check for tool calls
            # ALWAYS use intelligent model for tool selection
            intelligent_model = self.config.intelligent_llm_model_name
            response = self.streaming_service.sync_completion(
                windowed_messages, intelligent_model, "intelligent", tools=tools
            )

            # Parse response for content and tool calls
            content, tool_calls = self.parsing_service.parse_response(response)

            # If there are tool calls, execute them
            if tool_calls:
                # Log which tools were selected
                tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                logger.info(f"Tool selection (intelligent model): {', '.join(tool_names)}")
                yield await self._handle_tool_calls(tool_calls, windowed_messages, model, model_type)
            elif content:
                # Stream the content
                for char in content:
                    yield char
            else:
                # Fallback to streaming if no tool calls
                async for chunk in self.streaming_service.stream_completion(
                    windowed_messages, model, model_type, tools=tools
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Error: {str(e)}"

    async def _handle_tool_calls(
        self, tool_calls: List[Dict[str, Any]], messages: List[Dict[str, Any]], model: str, model_type: str
    ) -> str:
        """Handle tool calls and generate final response"""
        # Determine execution strategy
        strategy = self.tool_execution_service.determine_execution_strategy(tool_calls)

        # Get current user message
        current_user_message = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)

        # Execute tools
        tool_responses = await self.tool_execution_service.execute_tools(
            tool_calls, strategy=strategy, current_user_message=current_user_message, messages=messages
        )

        # Store for context extraction
        self.last_tool_responses = tool_responses

        # Check for direct response tools
        for response in tool_responses:
            if response.get("role") == "direct_response":
                return response.get("content", "")

        # Add tool responses to messages
        extended_messages = messages.copy()
        for response in tool_responses:
            if response.get("role") == "tool":
                extended_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Tool {response.get('tool_name')} returned: {response.get('content')}",
                    }
                )

        # Generate final response based on tool results
        final_response = ""
        async for chunk in self.streaming_service.stream_completion(extended_messages, model, model_type):
            final_response += chunk

        return final_response

    def _apply_sliding_window(
        self, messages: List[Dict[str, Any]], max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Apply sliding window to limit conversation history"""
        if max_turns is None:
            max_turns = config.llm.SLIDING_WINDOW_MAX_TURNS

        if not messages:
            return messages

        # Keep system and tool messages (important context)
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        tool_messages = [msg for msg in messages if msg.get("role") == "tool"]
        conversation_messages = [msg for msg in messages if msg.get("role") not in ["system", "tool"]]

        # Apply window to conversation messages only
        if len(conversation_messages) > max_turns * 2:
            conversation_messages = conversation_messages[-(max_turns * 2) :]

        # Return with system and tool messages preserved
        return system_messages + tool_messages + conversation_messages

    def _validate_and_clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean messages"""
        cleaned = []

        for msg in messages:
            # Skip empty messages
            if not msg.get("content"):
                continue

            # Clean tool instructions from content
            content = msg["content"]
            if isinstance(content, str):
                if self.parsing_service.contains_tool_calls(content):
                    content = self.parsing_service._clean_tool_instructions(content)

                # Filter think tags
                content = self.parsing_service.filter_think_tags(content)

            if content:
                cleaned.append({"role": msg["role"], "content": content})

        return cleaned

    # Backward compatibility methods
    def _filter_think_tags(self, content: str) -> str:
        """Backward compatibility wrapper"""
        return self.parsing_service.filter_think_tags(content)

    async def _generate_simple_response(self, message: str) -> AsyncGenerator[str, str]:
        """Simple response generation for backward compatibility"""
        async for chunk in self.streaming_service.simple_stream(message):
            yield chunk
