"""
Generalist Tool - MVC Pattern Implementation

This tool handles general conversation and thoughtful discussion on any topic
that doesn't require external data or specialized tools, following MVC pattern.
"""

from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import (
    BaseTool,
    BaseToolResponse,
    ExecutionMode,
    StreamingToolResponse,
    ToolController,
    ToolView,
)
from utils.logging_config import get_logger
from utils.text_processing import (
    StreamingCombinedThinkingFilter,
    strip_all_thinking_formats,
)

logger = get_logger(__name__)


class GeneralistResponse(BaseToolResponse):
    """Response from the generalist tool"""

    query: str = Field(description="The original user query, VERBATIM only.")
    response: str = Field(description="The conversational response")
    direct_response: bool = Field(
        default=True,
        description=(
            "Flag indicating this response should be returned directly to user"
        ),
    )


class StreamingGeneralistResponse(StreamingToolResponse):
    """Streaming response from the generalist tool"""

    query: str = Field(description="The original user query, VERBATIM only.")
    direct_response: bool = Field(
        default=True,
        description=(
            "Flag indicating this response should be returned directly to user"
        ),
    )


class GeneralistController(ToolController):
    """Controller handling generalist conversation logic"""

    def __init__(self, llm_type: str):
        self.llm_type = llm_type

    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the generalist conversation request"""
        query = params["query"]
        messages = params.get("messages")

        try:
            # Get LLM client and model based on tool configuration
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            # Create system prompt for general conversation
            system_prompt = self._get_system_prompt()

            # Build conversation messages
            final_messages = self._build_conversation_messages(
                system_prompt, query, messages
            )

            logger.debug(
                f"Generating conversational response using {model_name}"
            )

            # Generate response
            response = client.chat.completions.create(
                model=model_name, messages=final_messages
            )

            result = response.choices[0].message.content.strip()

            # Strip think tags from response
            cleaned_result = strip_all_thinking_formats(result)

            return {
                "query": query,
                "response": cleaned_result,
                "direct_response": True,
            }

        except Exception as e:
            logger.error(f"Error generating generalist response: {e}")
            raise

    async def process_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the generalist conversation request asynchronously with streaming"""
        query = params["query"]
        messages = params.get("messages")

        try:
            # Create streaming generator
            content_generator = self._generate_streaming_response(
                query, messages
            )

            return {
                "query": query,
                "content_generator": content_generator,
                "direct_response": True,
                "is_streaming": True,
            }

        except Exception as e:
            logger.error(
                f"Error setting up streaming generalist response: {e}"
            )
            raise

    async def _generate_streaming_response(
        self, query: str, messages: Optional[List[Dict[str, Any]]]
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response for generalist conversation"""
        try:
            # Get async LLM client and model based on tool configuration
            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            # Create system prompt for general conversation
            # Get configured system prompt if available
            from tools.tool_llm_config import get_tool_system_prompt

            system_prompt = get_tool_system_prompt(
                "generalist_conversation",
                "you're a helpful assistant that can answer questions and help"
                " with tasks or just chat. This tool needs the user's message,"
                " verbatim.",
            )

            # Build conversation messages
            final_messages = self._build_conversation_messages(
                system_prompt, query, messages
            )

            logger.debug(
                "Generating streaming conversational response using"
                f" {model_name}"
            )

            # Generate response with streaming
            response = await client.chat.completions.create(
                model=model_name,
                messages=final_messages,
                stream=True,
            )

            # Create think tag filter for streaming with model name
            think_filter = StreamingCombinedThinkingFilter(
                model_name=model_name
            )

            # Process stream with think tag filtering and yield chunks
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    # Filter think tags from the chunk
                    filtered_content = think_filter.process_chunk(
                        chunk_content
                    )
                    if filtered_content:
                        yield filtered_content

            # Yield any remaining content from the filter
            final_content = think_filter.flush()
            if final_content:
                yield final_content

        except Exception as e:
            logger.error(f"Error in streaming generalist response: {e}")
            yield f"Error generating response: {str(e)}"

    def _get_system_prompt(self) -> str:
        """Get the system prompt for general conversation"""
        # Get current date and time
        current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        # Get configured system prompt if available
        from tools.tool_llm_config import get_tool_system_prompt

        # Default prompt for generalist conversation
        default_prompt = f"""The current date and time is {current_date}."""

        # Use the configured system prompt if set, otherwise use default
        system_prompt = get_tool_system_prompt(
            "generalist_conversation", default_prompt
        )

        return system_prompt

    def _build_conversation_messages(
        self,
        system_prompt: str,
        query: str,
        messages: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, str]]:
        """Build the conversation messages for the LLM"""

        final_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation context if available
        if messages:
            # Filter and include recent conversation history (last 10 messages)
            conversation_messages = []
            for msg in messages:
                if msg.get("role") in ["user", "assistant"]:
                    # Clean the content
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        cleaned_content = strip_all_thinking_formats(content)
                        if cleaned_content.strip():
                            conversation_messages.append(
                                {
                                    "role": msg["role"],
                                    "content": cleaned_content,
                                }
                            )

            # Include last 10 messages for context
            recent_messages = (
                conversation_messages[-10:]
                if len(conversation_messages) > 10
                else conversation_messages
            )
            final_messages.extend(recent_messages)

        # Add the current user query
        final_messages.append({"role": "user", "content": query})

        return final_messages


class GeneralistView(ToolView):
    """View for formatting generalist responses"""

    def format_response(
        self, data: Dict[str, Any], response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format raw data into GeneralistResponse"""
        try:
            # Check if this is a streaming response
            if data.get("is_streaming") and data.get("content_generator"):
                return StreamingGeneralistResponse(**data)
            else:
                return GeneralistResponse(**data)
        except Exception as e:
            logger.error(f"Error formatting generalist response: {e}")
            return GeneralistResponse(
                query=data.get("query", ""),
                response="",
                success=False,
                error_message=f"Response formatting error: {str(e)}",
                error_code="FORMAT_ERROR",
            )

    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error into GeneralistResponse"""
        error_code = "UNKNOWN_ERROR"
        if isinstance(error, ValueError):
            error_code = "VALIDATION_ERROR"
        elif isinstance(error, TimeoutError):
            error_code = "TIMEOUT_ERROR"
        elif isinstance(error, ConnectionError):
            error_code = "CONNECTION_ERROR"

        return GeneralistResponse(
            query="",
            response=(
                "I apologize, but I encountered an error while processing"
                f" your message: {str(error)}"
            ),
            success=False,
            error_message=str(error),
            error_code=error_code,
            direct_response=True,
        )


class GeneralistTool(BaseTool):
    """
    Generalist Tool for General Conversation

    This tool handles thoughtful discussion on any topic that doesn't require
    external data, specialized tools, or real-time information. Use this for
    philosophical discussions, explanations of concepts, creative writing,
    general advice, and casual conversation.
    """

    def __init__(self):
        super().__init__()
        self.name = "generalist_conversation"
        self.description = (
            "Handle general conversation without external tools. "
            "CRITICAL: The 'query' parameter MUST contain the user's "
            "EXACT message without ANY modifications. Use for explanations, "
            "discussions, advice, creative writing, and casual chat."
        )
        self.supported_contexts = [
            "general_conversation",
            "discussion",
            "explanation",
        ]
        self.execution_mode = ExecutionMode.AUTO  # Support both sync and async
        self.timeout = 256.0

    def _initialize_mvc(self):
        """Initialize MVC components"""
        self._controller = GeneralistController(self.llm_type)
        self._view = GeneralistView()

    def get_definition(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The user's EXACT message, word-for-word. "
                                "Do NOT modify, expand, or rephrase. "
                                "If user says 'Hello!', pass 'Hello!' not "
                                "'Hello! How can I help you today?'"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def get_response_type(self) -> Type[BaseToolResponse]:
        """Get the response type for this tool"""
        return GeneralistResponse


# Helper functions for backward compatibility
def get_generalist_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for generalist conversation"""
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("generalist_conversation", GeneralistTool)

    # Get the tool instance and return its definition
    tool = get_tool("generalist_conversation")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get generalist tool definition")
