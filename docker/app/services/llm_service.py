"""
Refactored LLM Service

This is a simplified LLM service that orchestrates the focused services
for streaming, parsing, and tool execution.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from models.chat_config import ChatConfig
from services.conversation_context_service import ConversationContextService
from services.response_parsing_service import ResponseParsingService
from services.streaming_service import StreamingService
from services.tool_execution_service import ToolExecutionService
from tools.registry import tool_registry
from tools.tool_llm_config import DEFAULT_LLM_TYPE, get_tool_llm_type
from utils.config import config

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
        self.conversation_context_service = ConversationContextService(config)

        # For backward compatibility
        self.last_tool_responses = []

    def _get_model_for_type(self, model_type: str) -> str:
        """
        Get the appropriate model name for a given model type

        Args:
            model_type: The model type ("fast", "llm", or "intelligent")

        Returns:
            The model name from configuration
        """
        if model_type == "fast":
            return self.config.fast_llm_model_name
        elif model_type == "intelligent":
            return self.config.intelligent_llm_model_name
        else:  # "llm" or any other value defaults to regular model
            return self.config.llm_model_name

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

            # Get current user message for context injection
            current_user_message = ""
            for msg in reversed(windowed_messages):
                if msg.get("role") == "user":
                    current_user_message = msg.get("content", "")
                    break

            # Inject conversation context automatically
            windowed_messages = self.conversation_context_service.inject_conversation_context(
                windowed_messages, current_user_message
            )

            # Get tool definitions
            tools = tool_registry.get_all_definitions()

            # First, get non-streaming response to check for tool calls
            tool_selection_model_type = get_tool_llm_type("tool_selection")
            tool_selection_model = self._get_model_for_type(tool_selection_model_type)
            response = self.streaming_service.sync_completion(
                windowed_messages, tool_selection_model, tool_selection_model_type, tools=tools
            )

            # Parse response for content and tool calls
            content, tool_calls = self.parsing_service.parse_response(response)
            logging.info(f"Tool calls: {tool_calls}")

            # If there are tool calls, execute them
            if tool_calls:
                # Log which tools were selected
                tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                logger.info(f"Tool selection ({tool_selection_model_type} model): {', '.join(tool_names)}")
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

        # CRITICAL FIX: Only apply sliding window for very long conversations
        # For typical conversations (< 50 messages), keep full context
        # This prevents context loss in normal usage
        if len(conversation_messages) <= 50:  # ~25 turns of conversation
            logger.debug(f"Keeping full conversation context ({len(conversation_messages)} messages)")
            return system_messages + tool_messages + conversation_messages

        # For longer conversations, keep more context than before
        # Increase window size to at least 20 turns (40 messages) to maintain context
        effective_max_turns = max(max_turns, 20)
        window_size = effective_max_turns * 2

        if len(conversation_messages) > window_size:
            logger.info(
                f"Applying sliding window: keeping last {window_size} messages out of {len(conversation_messages)}"
            )
            conversation_messages = conversation_messages[-window_size:]

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
