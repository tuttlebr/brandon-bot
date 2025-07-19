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
from tools.registry import get_all_tool_definitions
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

        # Add PDF context service for context injection after tool execution
        from services.pdf_context_service import PDFContextService

        self.pdf_context_service = PDFContextService(config)

        # For backward compatibility
        self.last_tool_responses = []

    def _get_model_for_type(self, model_type: str) -> str:
        """
        Get the appropriate model name for a given model type

        Args:
            model_type: The model type ("fast", "llm", "intelligent", "vlm")

        Returns:
            The model name from configuration
        """
        if model_type == "fast":
            return self.config.fast_llm_model_name
        elif model_type == "intelligent":
            return self.config.intelligent_llm_model_name
        elif model_type == "vlm":
            return self.config.vlm_model_name
        else:
            return self.config.llm_model_name

    async def generate_streaming_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        model_type: str = DEFAULT_LLM_TYPE,
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

            # Check token count and truncate if necessary
            max_tokens = config.llm.MAX_CONTEXT_TOKENS
            estimated_tokens = self._count_message_tokens(windowed_messages)

            if estimated_tokens > max_tokens:
                logger.warning(
                    f"Message tokens ({estimated_tokens}) exceed limit ({max_tokens}). Truncating..."
                )
                windowed_messages, was_truncated = self._truncate_messages(
                    windowed_messages, max_tokens
                )

                if was_truncated:
                    # Yield a warning message to the user
                    yield "\n⚠️ **Note:** The conversation history was truncated to fit within the model's context limit. Some older messages may have been removed.\n\n"

            # Get current user message for context injection
            current_user_message = ""
            for msg in reversed(windowed_messages):
                if msg.get("role") == "user":
                    current_user_message = msg.get("content", "")
                    break

            # Inject conversation context automatically
            windowed_messages = (
                self.conversation_context_service.inject_conversation_context(
                    windowed_messages, current_user_message
                )
            )

            # Filter messages to ensure LLM compatibility
            windowed_messages = self._filter_messages_for_llm(windowed_messages)

            # Get tool definitions
            tools = get_all_tool_definitions()

            # Inject enhanced descriptions to improve tool selection
            try:
                from tools.tool_description_injector import inject_enhanced_descriptions

                tools = inject_enhanced_descriptions(tools)
            except ImportError:
                logger.debug("Enhanced tool descriptions not available")

            # Check if the user message is an acknowledgment
            from tools.tool_descriptions import (
                extract_actual_request,
                is_acknowledgment,
            )

            is_user_acknowledging = is_acknowledgment(current_user_message)
            actual_request = (
                extract_actual_request(current_user_message)
                if is_user_acknowledging
                else None
            )

            # If it's just an acknowledgment, don't use tools
            if is_user_acknowledging and not actual_request:
                logger.info(
                    "User message is an acknowledgment - skipping tool selection"
                )
                # Stream response without tools
                async for chunk in self.streaming_service.stream_completion(
                    windowed_messages, model, model_type, tools=None
                ):
                    yield chunk
                return

            # Insert the guidance right before tool selection
            windowed_messages_with_guidance = windowed_messages

            # First, get non-streaming response to check for tool calls
            tool_selection_model_type = get_tool_llm_type("tool_selection")
            tool_selection_model = self._get_model_for_type(tool_selection_model_type)
            response = self.streaming_service.sync_completion(
                windowed_messages_with_guidance,
                tool_selection_model,
                tool_selection_model_type,
                tools=tools,
                tool_choice="auto",
            )

            # Parse response for content and tool calls
            content, tool_calls = self.parsing_service.parse_response(response)

            # If there are tool calls, execute them and stream the response
            logger.info(f"All tool calls: {tool_calls}")
            if tool_calls:
                # Log which tools were selected (without arguments to avoid logging base64 data)
                tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                logger.info(
                    f"Tool selection ({tool_selection_model_type} model): {', '.join(tool_names)}"
                )

                # Stream chunks from tool handling
                async for chunk in self._handle_tool_calls(
                    tool_calls, windowed_messages, model, model_type
                ):
                    yield chunk

            else:
                # Fallback to streaming if no tool calls
                logger.info("No tool calls found, streaming response")
                async for chunk in self.streaming_service.stream_completion(
                    windowed_messages, model, model_type, tools=None
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            if "maximum context length" in str(e):
                yield "⚠️ The message was too long even after truncation. Please try with a shorter message or start a new conversation."
            else:
                yield f"Error: {str(e)}"

    async def _handle_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        model: str,
        model_type: str,
    ) -> AsyncGenerator[str, None]:
        """Handle tool calls and generate streaming response"""
        # Determine execution strategy
        strategy = self.tool_execution_service.determine_execution_strategy(tool_calls)

        # Get current user message
        current_user_message = next(
            (msg for msg in reversed(messages) if msg.get("role") == "user"), None
        )

        # Execute tools
        tool_responses = await self.tool_execution_service.execute_tools(
            tool_calls,
            strategy=strategy,
            current_user_message=current_user_message,
            messages=messages,
        )

        # Store for context extraction
        self.last_tool_responses = tool_responses

        # Determine the LLM type for final response based on executed tools
        # Use the first tool's configured LLM type (primary tool drives the response)
        response_llm_type = model_type  # fallback to original
        response_model = model  # fallback to original

        if tool_calls:
            primary_tool_name = tool_calls[0].get("name")
            if primary_tool_name:
                response_llm_type = get_tool_llm_type(primary_tool_name)
                response_model = self._get_model_for_type(response_llm_type)
                logger.info(
                    f"Using LLM type '{response_llm_type}' for final response based on primary tool '{primary_tool_name}'"
                )

        # Collect direct response tools with UI elements (like images)
        # instead of returning immediately
        ui_elements = []

        for response in tool_responses:
            if response.get("role") == "direct_response":
                # Check if this is an image generation response
                if response.get("tool_name") == "generate_image":
                    tool_result = response.get("tool_result")
                    if (
                        tool_result
                        and hasattr(tool_result, 'success')
                        and tool_result.success
                    ):
                        ui_elements.append(
                            {
                                "type": "image",
                                "tool_name": "generate_image",
                                "data": {
                                    "success": tool_result.success,
                                    "image_data": getattr(
                                        tool_result, 'image_data', None
                                    ),
                                    "original_prompt": getattr(
                                        tool_result, 'original_prompt', ''
                                    ),
                                    "enhanced_prompt": getattr(
                                        tool_result, 'enhanced_prompt', ''
                                    ),
                                },
                            }
                        )

        # Add all tool responses to messages for LLM synthesis
        extended_messages = messages.copy()

        # Add tool responses in a way that the LLM can understand
        for response in tool_responses:
            if response.get("role") == "tool":
                extended_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Tool {response.get('tool_name')} returned: {response.get('content')}",
                    }
                )
            elif response.get("role") == "direct_response":
                # For direct response tools, add a summary without the full content
                tool_name = response.get("tool_name", "unknown")
                if tool_name == "generate_image":
                    tool_result = response.get("tool_result")
                    if tool_result and hasattr(tool_result, 'success'):
                        if tool_result.success:
                            enhanced_prompt = getattr(
                                tool_result, 'enhanced_prompt', ''
                            )
                            extended_messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"Tool generate_image successfully created an image with prompt: {enhanced_prompt}",
                                }
                            )
                        else:
                            error_msg = getattr(
                                tool_result, 'error_message', 'Unknown error'
                            )
                            extended_messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"Tool generate_image failed: {error_msg}",
                                }
                            )
                else:
                    # For other direct response tools
                    content = response.get("content", "")
                    extended_messages.append(
                        {
                            "role": "assistant",
                            "content": f"Tool {tool_name} returned: {content[:200]}...",  # Truncate long responses
                        }
                    )

        # Re-inject PDF context after tool execution to ensure PDF content is available
        # for the final response generation, especially after PDF-related tool calls
        if current_user_message:
            # Use forced context injection after tool execution to ensure PDF content is available
            extended_messages = self.pdf_context_service.inject_pdf_context_forced(
                extended_messages
            )
            logger.info("Re-injected text context after tool execution")

        # Check token count after adding tool responses and truncate if necessary
        max_tokens = config.llm.MAX_CONTEXT_TOKENS
        estimated_tokens = self._count_message_tokens(extended_messages)

        if estimated_tokens > max_tokens:
            logger.warning(
                f"Messages with tool responses ({estimated_tokens} tokens) exceed limit ({max_tokens}). Truncating..."
            )
            extended_messages, was_truncated = self._truncate_messages(
                extended_messages, max_tokens
            )

            if was_truncated:
                # Yield a warning about truncation
                yield "\n⚠️ **Note:** The conversation including tool responses exceeded the context limit and was truncated.\n\n"

        # Filter messages to ensure LLM compatibility
        extended_messages = self._filter_messages_for_llm(extended_messages)

        # Store UI elements for the response controller to access
        self.last_ui_elements = ui_elements

        # Check if all tool responses are direct responses
        # If so, skip LLM synthesis since direct response tools provide their own final output
        all_direct_responses = all(
            response.get("role") == "direct_response" for response in tool_responses
        )

        if all_direct_responses:
            logger.info(
                "All tools are direct response tools - skipping LLM synthesis to avoid redundant commentary"
            )
            # For direct response tools, yield their content directly to the UI
            # BUT skip image generation tools since they're handled by the response controller
            for response in tool_responses:
                if response.get("role") == "direct_response":
                    tool_name = response.get("tool_name", "")
                    # Skip image generation tools - they're handled separately by response controller
                    if tool_name == "generate_image":
                        continue

                    content = response.get("content", "")
                    if content and content.strip():
                        # Yield the direct response content as chunks
                        yield content
            return
        else:
            # Stream the final synthesized response using the tool's configured LLM type
            logger.info(
                f"Streaming final response with {response_model}:{response_llm_type} (based on tool configuration)"
            )
            async for chunk in self.streaming_service.stream_completion(
                extended_messages, response_model, response_llm_type
            ):
                yield chunk

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
        conversation_messages = [
            msg for msg in messages if msg.get("role") not in ["system", "tool"]
        ]

        # CRITICAL FIX: Only apply sliding window for very long conversations
        # For typical conversations (< 50 messages), keep full context
        # This prevents context loss in normal usage
        if len(conversation_messages) <= 50:  # ~25 turns of conversation
            logger.debug(
                f"Keeping full conversation context ({len(conversation_messages)} messages)"
            )
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

    def _filter_messages_for_llm(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter messages to ensure they're compatible with LLM API

        Args:
            messages: List of messages

        Returns:
            Filtered messages safe for LLM
        """
        filtered = []

        for msg in messages:
            # Skip system messages with non-string content (like image uploads)
            if msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, dict):
                    logger.debug(
                        f"Filtering out system message with dict content of type: {content.get('type')}"
                    )
                    continue

            # Only include messages with string content
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                filtered.append(msg)
            elif isinstance(content, list):
                # Handle messages with list content (for multimodal)
                filtered.append(msg)

        return filtered

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text using character-based approximation

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        # This is a reasonable approximation for English text
        return len(text) // 4

    def _count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total estimated tokens in messages

        Args:
            messages: List of message dictionaries

        Returns:
            Total estimated token count
        """
        total_tokens = 0
        for message in messages:
            # Count role tokens (roughly 1 token)
            total_tokens += 1

            # Count content tokens
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self._estimate_tokens(content)

            # Add some overhead for message structure
            total_tokens += 3  # <|im_start|>, <|im_end|> etc.

        return total_tokens

    def _truncate_messages(
        self, messages: List[Dict[str, Any]], max_tokens: int
    ) -> tuple[List[Dict[str, Any]], bool]:
        """
        Truncate messages to fit within token limit

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum allowed tokens

        Returns:
            Tuple of (truncated messages, was_truncated flag)
        """
        # Always keep system messages and the latest user message
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        if not non_system_messages:
            return messages, False

        # Calculate tokens for system messages
        system_tokens = self._count_message_tokens(system_messages)

        # Reserve tokens for the latest user message (must always be included)
        latest_user_idx = None
        for i in range(len(non_system_messages) - 1, -1, -1):
            if non_system_messages[i].get("role") == "user":
                latest_user_idx = i
                break

        if latest_user_idx is None:
            return messages, False

        latest_user_msg = non_system_messages[latest_user_idx]
        latest_user_tokens = self._count_message_tokens([latest_user_msg])

        # Reserve some tokens for the response
        response_buffer = 4000  # Reserve 4k tokens for response
        available_tokens = (
            max_tokens - system_tokens - latest_user_tokens - response_buffer
        )

        if available_tokens <= 0:
            # Even with just system + latest user message, we're over limit
            # Truncate the user message content
            logger.warning(
                f"Message exceeds token limit even with minimal context. Truncating user message."
            )
            truncated_content = latest_user_msg["content"]
            while self._estimate_tokens(truncated_content) > (
                max_tokens - system_tokens - response_buffer - 100
            ):
                # Remove 25% of the content
                truncated_content = truncated_content[
                    : int(len(truncated_content) * 0.75)
                ]

            latest_user_msg = {
                **latest_user_msg,
                "content": truncated_content
                + "\n\n[Note: Message truncated due to length]",
            }
            return system_messages + [latest_user_msg], True

        # Build message list from most recent, keeping within token limit
        selected_messages = [latest_user_msg]
        selected_tokens = latest_user_tokens

        # Add messages from most recent to oldest (excluding latest user message)
        for i in range(len(non_system_messages) - 1, -1, -1):
            if i == latest_user_idx:
                continue

            msg = non_system_messages[i]
            msg_tokens = self._count_message_tokens([msg])

            if selected_tokens + msg_tokens > available_tokens:
                break

            selected_messages.insert(0, msg)
            selected_tokens += msg_tokens

        # Check if we truncated
        was_truncated = len(selected_messages) < len(non_system_messages)

        # Combine system messages with selected messages
        final_messages = system_messages + selected_messages

        return final_messages, was_truncated
