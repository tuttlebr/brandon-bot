"""
Generalist Tool - MVC Pattern Implementation

This tool handles general conversation and thoughtful discussion on any topic
that doesn't require external data or specialized tools, following MVC pattern.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import (
    BaseTool,
    BaseToolResponse,
    ExecutionMode,
    ToolController,
    ToolView,
)
from utils.text_processing import strip_think_tags

logger = logging.getLogger(__name__)


class GeneralistResponse(BaseToolResponse):
    """Response from the generalist tool"""

    query: str = Field(description="The original user query")
    response: str = Field(description="The conversational response")
    direct_response: bool = Field(
        default=True,
        description="Flag indicating this response should be returned directly to user",
    )


class GeneralistController(ToolController):
    """Controller handling generalist conversation logic"""

    def __init__(self, llm_type: str):
        self.llm_type = llm_type

    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the generalist conversation request"""
        query = params['query']
        messages = params.get('messages')

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

            logger.debug(f"Generating conversational response using {model_name}")

            # Generate response
            response = client.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=0.0,
            )

            result = response.choices[0].message.content.strip()

            # Strip think tags from response
            cleaned_result = strip_think_tags(result)

            return {
                "query": query,
                "response": cleaned_result,
                "direct_response": True,
            }

        except Exception as e:
            logger.error(f"Error generating generalist response: {e}")
            raise

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
        self, system_prompt: str, query: str, messages: Optional[List[Dict[str, Any]]]
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
                        cleaned_content = strip_think_tags(content)
                        if cleaned_content.strip():
                            conversation_messages.append(
                                {"role": msg["role"], "content": cleaned_content}
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
            return GeneralistResponse(**data)
        except Exception as e:
            logger.error(f"Error formatting generalist response: {e}")
            return GeneralistResponse(
                query=data.get("query", ""),
                response=f"I apologize, but I encountered an error while processing your message.",
                success=False,
                error_message=f"Response formatting error: {str(e)}",
                error_code="FORMAT_ERROR",
                direct_response=True,
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
            response=f"I apologize, but I encountered an error while processing your message: {str(error)}",
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
        self.description = "ONLY use for general conversation, explanations, discussions, advice, or topics that don't require external data, real-time information, or specialized tools. Use for: philosophical discussions, explaining concepts, creative writing, general advice, casual conversation, or when the user wants to have a thoughtful discussion about any topic. DO NOT use for current events, weather, searches, image analysis, document processing, or any task that requires external data - use appropriate specialized tools for those."
        self.supported_contexts = ['general_conversation', 'discussion', 'explanation']
        self.execution_mode = ExecutionMode.SYNC
        self.timeout = 30.0

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
                            "description": "The user's message or question exactly as they provided it",
                        },
                        "but_why": {
                            "type": "string",
                            "description": "A single sentence explaining why this tool was selected for the query.",
                        },
                    },
                    "required": ["query", "but_why"],
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
