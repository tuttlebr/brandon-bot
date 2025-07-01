import logging
from typing import Any, Dict, Optional

from models.chat_config import ChatConfig
from openai import OpenAI
from pydantic import BaseModel, Field
from services.llm_client_service import llm_client_service
from tools.base import BaseTool, BaseToolResponse
from utils.config import config as app_config

# Configure logger
logger = logging.getLogger(__name__)


class DefaultFallbackResponse(BaseToolResponse):
    """Response from default fallback tool"""

    response: str = Field(description="The LLM response to the user's query")
    direct_response: bool = Field(default=False, description="Whether this is a direct response")

    def json(self) -> str:
        """Return JSON representation of the response"""
        return f'{{"response": "{self.response}", "direct_response": {str(self.direct_response).lower()}}}'

    @property
    def result(self) -> str:
        """Get the response content for direct responses"""
        return self.response


class DefaultFallbackTool(BaseTool):
    """Default fallback tool when no other tool matches"""

    def __init__(self):
        super().__init__()
        self.name = "default_fallback"
        self.description = "Use this tool ONLY for: (1) Social interactions - greetings, thanks, goodbyes; (2) General knowledge questions that don't require real-time data; (3) Explanations of concepts or ideas; (4) Advice or opinions; (5) Any query that doesn't match other tools. Do NOT use for: weather, news, web searches, PDFs, image generation, or text processing tasks."
        # Use fast model for quick responses
        self.llm_type = "fast"

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI function calling format

        Returns:
            Dict containing the OpenAI-compatible tool definition
        """
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
                            "description": "The user's original query that couldn't be matched to a specific tool",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context about why no specific tool was matched",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for BaseTool interface"""
        return self.to_openai_format()

    def _generate_response(
        self, query: str, context: Optional[str] = None, config: Optional[ChatConfig] = None
    ) -> str:
        """
        Generate a response using the Fast LLM service

        Args:
            query: The user's query
            context: Optional context information
            config: Chat configuration for API access

        Returns:
            Generated response string

        Raises:
            Exception: If response generation fails
        """
        try:
            # Get the appropriate client and model based on this tool's LLM type
            fast_client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            # Import SYSTEM_PROMPT lazily to avoid circular imports
            from utils.system_prompt import SYSTEM_PROMPT

            # Prepare the user message
            user_message = query
            if context:
                user_message = f"Context: {context}\n\nQuery: {query}"

            logger.debug(f"Making LLM request with model: {model_name} (type: {self.llm_type})")

            # Generate response using Fast LLM
            response = fast_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                max_tokens=500,  # Reasonable limit for general responses
                stream=True,  # Enable streaming for better UX
            )

            # Handle streaming response
            generated_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    generated_response += chunk.choices[0].delta.content

            generated_response = generated_response.strip()

            if not generated_response:
                logger.warning("Fast LLM returned empty response")
                return "I apologize, but I wasn't able to generate a proper response to your query. Please try rephrasing your question."

            logger.debug(f"Generated response for query: '{query[:50]}...' -> '{generated_response[:100]}...'")
            return generated_response

        except Exception as e:
            logger.error(f"Error generating response with Fast LLM: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def run_with_dict(self, params: Dict[str, Any]) -> DefaultFallbackResponse:
        """
        Execute the default fallback tool with parameters provided as a dictionary

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'query', optional: 'context', 'config'

        Returns:
            DefaultFallbackResponse: The response from the Fast LLM
        """
        if "query" not in params:
            raise ValueError("'query' key is required in parameters dictionary")

        query = params["query"]
        context = params.get("context")
        config = params.get("config")

        logger.info(f"Default fallback tool called with query: '{query[:50]}...'")

        try:
            # Generate response using Fast LLM
            response_text = self._generate_response(query, context, config)

            # Return as a direct response so it streams back to the UI
            result = DefaultFallbackResponse(response=response_text, direct_response=True)

            logger.debug(f"Default fallback completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in default fallback tool: {e}")
            # Return error as direct response
            return DefaultFallbackResponse(
                response=f"I apologize, but I encountered an error: {str(e)}", direct_response=True
            )

    def execute(self, params: Dict[str, Any]) -> DefaultFallbackResponse:
        """
        Execute the tool with given parameters

        Args:
            params: Dictionary containing the required parameters

        Returns:
            DefaultFallbackResponse
        """
        return self.run_with_dict(params)


# Create a global instance and helper functions for easy access
default_fallback_tool = DefaultFallbackTool()


def get_default_fallback_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for default fallback

    Returns:
        Dict containing the OpenAI tool definition
    """
    return default_fallback_tool.to_openai_format()


def execute_default_fallback_with_dict(params: Dict[str, Any]) -> DefaultFallbackResponse:
    """
    Execute the default fallback tool with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters
               Expected keys: 'query', optional: 'context', 'config'

    Returns:
        DefaultFallbackResponse: The response from the Fast LLM
    """
    return default_fallback_tool.run_with_dict(params)
