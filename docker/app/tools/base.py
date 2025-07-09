"""
Base Tool Interface

This module provides the base interface for all tools in the system,
ensuring consistent implementation and behavior across all tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class BaseToolResponse(BaseModel):
    """Base response model that all tool responses should inherit from

    Attributes:
        success: Whether the tool execution was successful
        error_message: Error message if the tool failed
        direct_response: When True, the tool's response will be returned directly to the user
                        without being wrapped in tool response formatting. This is useful for
                        conversational tools (like text_assistant) that generate natural
                        language responses. When False (default), the response will be
                        formatted as a tool response for the LLM to process.
    """

    success: bool = True
    error_message: Optional[str] = None
    direct_response: bool = False  # Whether to return response directly to user


class BaseTool(ABC):
    """Abstract base class for all tools"""

    def __init__(self):
        self._name: str = ""
        self.description: str = ""
        # LLM type configuration - which LLM this tool should use
        # This will be automatically set based on the tool name
        self.llm_type: Literal["fast", "llm", "intelligent", "vlm"] = "intelligent"
        # Contexts this tool supports (for system prompt context mapping)
        self.supported_contexts: List[str] = []

    @property
    def name(self) -> str:
        """Get the tool name"""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the tool name and automatically configure llm_type"""
        self._name = value
        if value:  # Only set if name is not empty
            from tools.tool_llm_config import get_tool_llm_type

            self.llm_type = get_tool_llm_type(value)

    def get_definition(self) -> Dict[str, Any]:
        """
        Return OpenAI-compatible tool definition

        Returns:
            Dict containing the tool definition in OpenAI function calling format
        """
        # Default implementation: use to_openai_format if available
        if hasattr(self, 'to_openai_format'):
            return self.to_openai_format()
        else:
            raise NotImplementedError(
                "Tool must implement either to_openai_format() or override get_definition()"
            )

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> BaseToolResponse:
        """
        Execute the tool with given parameters

        Args:
            params: Dictionary of parameters for the tool

        Returns:
            A response object inheriting from BaseToolResponse
        """

    def validate_params(self, params: Dict[str, Any], required: list) -> None:
        """
        Validate that required parameters are present

        Args:
            params: Parameters to validate
            required: List of required parameter names

        Raises:
            ValueError: If required parameters are missing
        """
        missing = [param for param in required if param not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    def get_llm_type(self) -> str:
        """
        Get the LLM type this tool should use

        Returns:
            The LLM type: "fast", "llm", "intelligent", or "vlm"
        """
        return self.llm_type
