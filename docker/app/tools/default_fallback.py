import logging
from typing import Any, Dict, Optional

from models.chat_config import ChatConfig
from openai import OpenAI
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class DefaultFallbackResponse(BaseModel):
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


class DefaultFallbackTool:
    """Default fallback tool for handling general queries using Fast LLM"""

    def __init__(self):
        self.name = "default_fallback"
        self.description = "Fallback tool for general conversational queries and questions that don't require specialized tools. Uses the Fast LLM service to provide quick, helpful responses to everyday questions, casual conversation, explanations, and general assistance."

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
                            "description": "The user's question or query that needs a general response",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context or additional information to help provide a better response",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def _create_fast_llm_client(self, config: ChatConfig) -> OpenAI:
        """Create a Fast LLM client for generating responses"""
        try:
            return OpenAI(api_key=config.api_key, base_url=config.fast_llm_endpoint)
        except Exception as e:
            logger.error(f"Failed to create Fast LLM client: {e}")
            raise

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
        if not config:
            raise ValueError("ChatConfig is required for LLM access")

        try:
            # Create Fast LLM client
            fast_client = self._create_fast_llm_client(config)

            # Prepare the system prompt for general assistance
            system_prompt = """You are a helpful AI assistant. Provide clear, accurate, and concise responses to user questions.
Be friendly and conversational while being informative. If you're unsure about something, say so honestly.
Keep responses focused and avoid unnecessary verbosity unless the user specifically asks for detailed information."""

            # Prepare the user message
            user_message = query
            if context:
                user_message = f"Context: {context}\n\nQuery: {query}"

            # Generate response using Fast LLM
            response = fast_client.chat.completions.create(
                model=config.fast_llm_model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                temperature=0.3,
                max_tokens=500,  # Reasonable limit for general responses
                stream=False,
            )

            generated_response = response.choices[0].message.content.strip()

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

            # Return as a direct response
            result = DefaultFallbackResponse(response=response_text, direct_response=False)

            logger.debug(f"Default fallback completed successfully")
            return result

        except Exception as e:
            logger.error(f"Error in default fallback tool: {e}")
            # Return error as direct response
            return DefaultFallbackResponse(
                response=f"I apologize, but I encountered an error: {str(e)}", direct_response=False
            )


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
