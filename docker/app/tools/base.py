"""
Base Tool Interface - MVC Pattern Implementation

This module provides the base interface for all tools in the system,
following the Model-View-Controller pattern with proper separation of concerns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Execution modes for tools"""

    SYNC = "sync"
    ASYNC = "async"
    AUTO = "auto"  # Automatically determine based on context


class BaseToolResponse(BaseModel):
    """Base response model that all tool responses should inherit from

    Attributes:
        success: Whether the tool execution was successful
        error_message: Error message if the tool failed
        error_code: Error code for programmatic handling
        direct_response: When True, the tool's response will be returned directly to the user
                        without being wrapped in tool response formatting. This is useful for
                        conversational tools (like text_assistant) that generate natural
                        language responses. When False (default), the response will be
                        formatted as a tool response for the LLM to process.
        metadata: Additional metadata about the tool execution
    """

    success: bool = True
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    direct_response: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamingToolResponse(BaseToolResponse):
    """Base response model for tools that support streaming

    This response type allows tools to return an async generator that yields
    content chunks as they arrive, enabling true end-to-end streaming.
    """

    content_generator: Optional[Any] = Field(
        None, description="Async generator that yields content chunks"
    )
    is_streaming: bool = Field(
        default=True, description="Flag indicating this is a streaming response"
    )

    class Config:
        arbitrary_types_allowed = True


class ToolController(ABC):
    """Abstract controller for tool business logic"""

    @abstractmethod
    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the tool request and return raw data"""


class ToolView(ABC):
    """Abstract view for formatting tool responses"""

    @abstractmethod
    def format_response(
        self, data: Dict[str, Any], response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format raw data into appropriate response model"""

    @abstractmethod
    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error into appropriate response model"""


class BaseTool(ABC):
    """Abstract base class for all tools following MVC pattern"""

    def __init__(self):
        self._name: str = ""
        self.description: str = ""
        # LLM type configuration - which LLM this tool should use
        self.llm_type: Literal["fast", "llm", "intelligent", "vlm"] = "llm"
        # Contexts this tool supports (for system prompt context mapping)
        self.supported_contexts: List[str] = []
        # Execution mode
        self.execution_mode: ExecutionMode = ExecutionMode.SYNC
        # Timeout for async operations
        self.timeout: Optional[float] = 30.0

        # MVC components (to be initialized in subclasses)
        self._controller: Optional[ToolController] = None
        self._view: Optional[ToolView] = None

        # Initialize MVC components
        self._initialize_mvc()

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

    @abstractmethod
    def _initialize_mvc(self):
        """Initialize Model-View-Controller components. Must be implemented by subclasses."""

    @abstractmethod
    def get_definition(self) -> Dict[str, Any]:
        """
        Return OpenAI-compatible tool definition

        Returns:
            Dict containing the tool definition in OpenAI function calling format
        """

    def execute(self, params: Dict[str, Any]) -> BaseToolResponse:
        """
        Execute the tool with given parameters following MVC pattern

        Args:
            params: Dictionary of parameters for the tool

        Returns:
            A response object inheriting from BaseToolResponse
        """
        try:
            # Validate parameters
            self._validate_params(params)

            # Controller: Process business logic
            if self._controller is None:
                raise RuntimeError(f"Controller not initialized for {self.name}")

            # Determine execution mode
            if self.execution_mode == ExecutionMode.ASYNC or (
                self.execution_mode == ExecutionMode.AUTO
                and self._has_async_implementation()
            ):
                # Handle async execution
                raw_data = self._execute_async_wrapper(params)
            else:
                # Synchronous execution
                raw_data = self._controller.process(params)

            # View: Format response
            if self._view is None:
                raise RuntimeError(f"View not initialized for {self.name}")

            return self._view.format_response(raw_data, self.get_response_type())

        except ValidationError as e:
            logger.error(f"Validation error in {self.name}: {e}")
            if self._view:
                return self._view.format_error(e, self.get_response_type())
            else:
                return self._create_error_response(
                    f"Invalid parameters: {str(e)}", "VALIDATION_ERROR"
                )
        except PydanticValidationError as e:
            logger.error(f"Pydantic validation error in {self.name}: {e}")
            if self._view:
                return self._view.format_error(e, self.get_response_type())
            else:
                return self._create_error_response(
                    f"Invalid parameters: {str(e)}", "VALIDATION_ERROR"
                )
        except Exception as e:
            logger.error(f"Error executing {self.name}: {e}", exc_info=True)
            if self._view:
                return self._view.format_error(e, self.get_response_type())
            else:
                return self._create_error_response(str(e), "EXECUTION_ERROR")

    def _execute_async_wrapper(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper to execute async methods in sync context"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            if loop.is_running():
                # If loop is already running, use run_coroutine_threadsafe
                logger.info(
                    f"Executing {self.name} async with timeout={self.timeout}s in running loop"
                )
                future = asyncio.run_coroutine_threadsafe(
                    self._execute_controller_async(params), loop
                )
                result = future.result(timeout=self.timeout)
                logger.info(f"Async execution of {self.name} completed successfully")
                return result
            else:
                # If no loop is running, use run_until_complete
                logger.info(f"Executing {self.name} async in new loop")
                return loop.run_until_complete(self._execute_controller_async(params))
        except asyncio.TimeoutError:
            logger.error(f"Timeout executing {self.name} after {self.timeout}s")
            raise TimeoutError(f"Execution timed out after {self.timeout} seconds")

    async def _execute_controller_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute controller asynchronously"""
        if hasattr(self._controller, 'process_async'):
            return await self._controller.process_async(params)
        else:
            # Fallback to sync in thread
            return await asyncio.to_thread(self._controller.process, params)

    def _has_async_implementation(self) -> bool:
        """Check if controller has async implementation"""
        return hasattr(self._controller, 'process_async')

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """
        Enhanced parameter validation using tool definition schema

        Args:
            params: Parameters to validate

        Raises:
            ValidationError: If validation fails
        """
        definition = self.get_definition()
        if "function" not in definition:
            return  # Skip if no schema available

        function_schema = definition.get("function", {})
        param_schema = function_schema.get("parameters", {})

        required = param_schema.get("required", [])
        properties = param_schema.get("properties", {})

        # Special handling for 'but_why' parameter - provide default for internal calls
        if "but_why" in required and "but_why" not in params:
            if "but_why" in properties:
                but_why_property = properties["but_why"]
                but_why_type = but_why_property.get("type")

                # Set appropriate default based on expected type
                if but_why_type == "integer":
                    params["but_why"] = 5  # High confidence for internal calls
                else:
                    params["but_why"] = (
                        f"Internal call to {self.name} tool for processing"
                    )

                logger.debug(
                    f"Added default but_why ({params['but_why']}) for internal {self.name} call"
                )

        # Check required parameters
        missing = [param for param in required if param not in params]
        if missing:
            raise ValidationError(f"Missing required parameters: {', '.join(missing)}")

        # Basic type validation
        for param_name, param_value in params.items():
            if param_name in properties:
                expected = properties[param_name]
                param_type = expected.get("type")

                # Type checking
                if param_type == "string" and not isinstance(param_value, str):
                    raise ValidationError(f"Parameter '{param_name}' must be a string")
                elif param_type == "integer" and not isinstance(param_value, int):
                    raise ValidationError(
                        f"Parameter '{param_name}' must be an integer"
                    )
                elif param_type == "number" and not isinstance(
                    param_value, (int, float)
                ):
                    raise ValidationError(f"Parameter '{param_name}' must be a number")
                elif param_type == "boolean" and not isinstance(param_value, bool):
                    raise ValidationError(f"Parameter '{param_name}' must be a boolean")
                elif param_type == "array" and not isinstance(param_value, list):
                    raise ValidationError(f"Parameter '{param_name}' must be an array")

                # Enum validation
                if "enum" in expected and param_value not in expected["enum"]:
                    raise ValidationError(
                        f"Parameter '{param_name}' must be one of: {expected['enum']}"
                    )

    def _create_error_response(
        self, error_message: str, error_code: str = "UNKNOWN_ERROR"
    ) -> BaseToolResponse:
        """Create a standardized error response"""
        response_type = self.get_response_type()
        try:
            return response_type(
                success=False, error_message=error_message, error_code=error_code
            )
        except Exception:
            # Fallback to base response
            return BaseToolResponse(
                success=False, error_message=error_message, error_code=error_code
            )

    @abstractmethod
    def get_response_type(self) -> Type[BaseToolResponse]:
        """
        Get the response type for this tool

        Returns:
            The response class type
        """
