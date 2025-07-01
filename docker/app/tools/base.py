"""
Base Tool Interface

This module provides the base interface for all tools in the system,
ensuring consistent implementation and behavior across all tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class BaseToolResponse(BaseModel):
    """Base response model that all tool responses should inherit from

    Attributes:
        success: Whether the tool execution was successful
        error_message: Error message if the tool failed
        direct_response: When True, the tool's response will be returned directly to the user
                        without being wrapped in tool response formatting. This is useful for
                        conversational tools (like default_fallback) that generate natural
                        language responses. When False (default), the response will be
                        formatted as a tool response for the LLM to process.
    """

    success: bool = True
    error_message: Optional[str] = None
    direct_response: bool = False  # Whether to return response directly to user


class BaseTool(ABC):
    """Abstract base class for all tools"""

    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        # LLM type configuration - which LLM this tool should use
        # Options: "fast", "llm", "intelligent"
        # Default to "fast" for efficiency, tools can override
        self.llm_type: Literal["fast", "llm", "intelligent"] = "fast"

    @abstractmethod
    def get_definition(self) -> Dict[str, Any]:
        """
        Return OpenAI-compatible tool definition

        Returns:
            Dict containing the tool definition in OpenAI function calling format
        """
        pass

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> BaseToolResponse:
        """
        Execute the tool with given parameters

        Args:
            params: Dictionary of parameters for the tool

        Returns:
            A response object inheriting from BaseToolResponse
        """
        pass

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
            The LLM type: "fast", "llm", or "intelligent"
        """
        return self.llm_type
