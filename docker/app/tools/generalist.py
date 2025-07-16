"""
Generalist Tool

This tool handles general conversation and thoughtful discussion on any topic
that doesn't require external data or specialized tools.
"""

import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import BaseTool, BaseToolResponse
from utils.config import config as app_config
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

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert the tool to OpenAI function calling format"""
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

    def execute(self, params: Dict[str, Any]) -> GeneralistResponse:
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def generate_response(
        self,
        query: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> GeneralistResponse:
        """
        Generate a thoughtful conversational response

        Args:
            query: The user's query or message exactly as provided
            messages: Optional conversation context

        Returns:
            GeneralistResponse with the conversational response
        """
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

            return GeneralistResponse(
                query=query,
                response=cleaned_result,
                direct_response=True,
            )

        except Exception as e:
            logger.error(f"Error generating generalist response: {e}")
            return GeneralistResponse(
                query=query,
                response=f"I apologize, but I encountered an error while processing your message: {str(e)}",
                success=False,
                error_message=str(e),
                direct_response=True,
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for general conversation"""

        return """detailed thinking off"""

    def _build_conversation_messages(
        self, system_prompt: str, query: str, messages: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
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

    def run_with_dict(self, params: Dict[str, Any]) -> GeneralistResponse:
        """
        Execute generalist conversation with parameters provided as a dictionary

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'query', optionally 'messages'

        Returns:
            GeneralistResponse: The conversational response
        """
        if "query" not in params:
            raise ValueError("'query' key is required in parameters dictionary")

        query = params["query"]
        messages = params.get("messages", None)

        logger.debug(f"Generalist conversation: query='{query[:100]}...'")

        return self.generate_response(query, messages)


# Create a global instance and helper functions
generalist_tool = GeneralistTool()


def get_generalist_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for generalist conversation"""
    return generalist_tool.to_openai_format()


def execute_generalist_conversation(
    query: str, messages: Optional[List[Dict[str, Any]]] = None
) -> GeneralistResponse:
    """
    Execute a generalist conversation

    Args:
        query: The user's query or message exactly as provided
        messages: Optional conversation context

    Returns:
        GeneralistResponse: The conversational response
    """
    return generalist_tool.generate_response(query, messages)


def execute_generalist_with_dict(params: Dict[str, Any]) -> GeneralistResponse:
    """
    Execute generalist conversation with parameters as dictionary

    Args:
        params: Dictionary containing parameters

    Returns:
        GeneralistResponse: The conversational response
    """
    return generalist_tool.run_with_dict(params)
