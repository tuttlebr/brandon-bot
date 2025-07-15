"""
Tool Entity Domain Model

Represents a tool entity in the chatbot system with proper validation,
metadata management, and execution tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class ToolStatus(str, Enum):
    """Tool status enumeration"""

    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


class ToolType(str, Enum):
    """Tool type enumeration"""

    TEXT_PROCESSING = "text_processing"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    WEB_SEARCH = "web_search"
    DATA_RETRIEVAL = "data_retrieval"
    UTILITY = "utility"
    CUSTOM = "custom"


class ExecutionMetrics(BaseModel):
    """Tool execution metrics"""

    total_executions: int = Field(
        default=0, ge=0, description="Total number of executions"
    )
    successful_executions: int = Field(
        default=0, ge=0, description="Number of successful executions"
    )
    failed_executions: int = Field(
        default=0, ge=0, description="Number of failed executions"
    )
    average_execution_time: float = Field(
        default=0.0, ge=0, description="Average execution time in seconds"
    )
    last_execution: Optional[datetime] = Field(
        None, description="Timestamp of last execution"
    )
    last_success: Optional[datetime] = Field(
        None, description="Timestamp of last successful execution"
    )
    last_error: Optional[str] = Field(None, description="Last error message")

    def get_success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100

    def record_execution(
        self, success: bool, execution_time: float, error_message: Optional[str] = None
    ) -> None:
        """Record a tool execution"""
        self.total_executions += 1
        self.last_execution = datetime.now()

        if success:
            self.successful_executions += 1
            self.last_success = datetime.now()
        else:
            self.failed_executions += 1
            self.last_error = error_message

        # Update average execution time
        if self.total_executions == 1:
            self.average_execution_time = execution_time
        else:
            self.average_execution_time = (
                self.average_execution_time * (self.total_executions - 1)
                + execution_time
            ) / self.total_executions


class ToolParameter(BaseModel):
    """Tool parameter definition"""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    enum_values: Optional[List[str]] = Field(None, description="Allowed enum values")

    @validator('name')
    def validate_name(cls, v):
        """Validate parameter name"""
        if not v or not v.strip():
            raise ValueError("Parameter name cannot be empty")
        if not v.replace('_', '').isalnum():
            raise ValueError("Parameter name must be alphanumeric with underscores")
        return v.strip()


class ToolEntity(BaseModel):
    """
    Tool entity domain model representing a chatbot tool

    This model encapsulates tool metadata, configuration,
    execution tracking, and business rules.
    """

    tool_id: str = Field(..., description="Unique tool identifier")
    name: str = Field(..., description="Tool name")
    display_name: str = Field(..., description="Human-readable tool name")
    description: str = Field(..., description="Tool description")
    tool_type: ToolType = Field(..., description="Tool type category")
    llm_type: Literal["fast", "llm", "intelligent", "vlm"] = Field(
        default="intelligent", description="Required LLM type"
    )
    status: ToolStatus = Field(
        default=ToolStatus.AVAILABLE, description="Current tool status"
    )
    version: str = Field(default="1.0.0", description="Tool version")
    supported_contexts: List[str] = Field(
        default_factory=list, description="Supported context types"
    )
    parameters: List[ToolParameter] = Field(
        default_factory=list, description="Tool parameters"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Tool creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )
    metrics: ExecutionMetrics = Field(
        default_factory=ExecutionMetrics, description="Execution metrics"
    )
    configuration: Dict[str, Any] = Field(
        default_factory=dict, description="Tool configuration"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tool tags for categorization"
    )

    class Config:
        """Pydantic configuration"""

        use_enum_values = True
        validate_assignment = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    @validator('tool_id')
    def validate_tool_id(cls, v):
        """Validate tool ID format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Tool ID cannot be empty")
        if len(v) > 100:
            raise ValueError("Tool ID cannot exceed 100 characters")
        return v.strip()

    @validator('name')
    def validate_name(cls, v):
        """Validate tool name"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Tool name cannot be empty")
        if len(v) > 200:
            raise ValueError("Tool name cannot exceed 200 characters")
        return v.strip()

    @validator('version')
    def validate_version(cls, v):
        """Validate version format (semantic versioning)"""
        import re

        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        if not re.match(pattern, v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.0.0)")
        return v

    def update_timestamp(self) -> None:
        """Update the tool's last update timestamp"""
        self.updated_at = datetime.now()

    def set_status(self, status: ToolStatus) -> None:
        """
        Set tool status

        Args:
            status: New tool status
        """
        self.status = status
        self.update_timestamp()

    def is_available(self) -> bool:
        """
        Check if tool is available for execution

        Returns:
            True if tool is available, False otherwise
        """
        return self.status == ToolStatus.AVAILABLE

    def add_parameter(
        self,
        name: str,
        param_type: str,
        description: str,
        required: bool = True,
        default: Any = None,
        enum_values: Optional[List[str]] = None,
    ) -> None:
        """
        Add a parameter to the tool

        Args:
            name: Parameter name
            param_type: Parameter type
            description: Parameter description
            required: Whether parameter is required
            default: Default value
            enum_values: Allowed enum values
        """
        parameter = ToolParameter(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=default,
            enum_values=enum_values,
        )
        self.parameters.append(parameter)
        self.update_timestamp()

    def get_parameter(self, name: str) -> Optional[ToolParameter]:
        """
        Get parameter by name

        Args:
            name: Parameter name

        Returns:
            ToolParameter object or None if not found
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def get_required_parameters(self) -> List[ToolParameter]:
        """
        Get list of required parameters

        Returns:
            List of required ToolParameter objects
        """
        return [p for p in self.parameters if p.required]

    def validate_parameters(self, provided_params: Dict[str, Any]) -> List[str]:
        """
        Validate provided parameters against tool requirements

        Args:
            provided_params: Dictionary of provided parameters

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required parameters
        for param in self.get_required_parameters():
            if param.name not in provided_params:
                errors.append(f"Required parameter '{param.name}' is missing")

        # Check enum values
        for param in self.parameters:
            if param.name in provided_params and param.enum_values:
                value = provided_params[param.name]
                if value not in param.enum_values:
                    errors.append(
                        f"Parameter '{param.name}' must be one of {param.enum_values}"
                    )

        return errors

    def record_execution(
        self, success: bool, execution_time: float, error_message: Optional[str] = None
    ) -> None:
        """
        Record a tool execution

        Args:
            success: Whether execution was successful
            execution_time: Execution time in seconds
            error_message: Error message if execution failed
        """
        self.metrics.record_execution(success, execution_time, error_message)
        self.update_timestamp()

        # Update status based on execution result
        if not success and self.status != ToolStatus.ERROR:
            self.set_status(ToolStatus.ERROR)
        elif success and self.status == ToolStatus.ERROR:
            self.set_status(ToolStatus.AVAILABLE)

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the tool

        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self.update_timestamp()

    def remove_tag(self, tag: str) -> None:
        """
        Remove a tag from the tool

        Args:
            tag: Tag to remove
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self.update_timestamp()

    def has_tag(self, tag: str) -> bool:
        """
        Check if tool has a specific tag

        Args:
            tag: Tag to check

        Returns:
            True if tool has the tag
        """
        return tag in self.tags

    def supports_context(self, context: str) -> bool:
        """
        Check if tool supports a specific context

        Args:
            context: Context to check

        Returns:
            True if tool supports the context
        """
        return context in self.supported_contexts

    def get_openai_function_definition(self) -> Dict[str, Any]:
        """
        Get OpenAI function definition format

        Returns:
            Dictionary in OpenAI function format
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop_def = {"type": param.type, "description": param.description}

            if param.enum_values:
                prop_def["enum"] = param.enum_values

            properties[param.name] = prop_def

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_registry_format(self) -> Dict[str, Any]:
        """
        Convert to tool registry format

        Returns:
            Dictionary suitable for tool registry
        """
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "tool_type": self.tool_type,
            "llm_type": self.llm_type,
            "status": self.status,
            "version": self.version,
            "supported_contexts": self.supported_contexts,
            "parameters": [p.dict() for p in self.parameters],
            "metrics": self.metrics.dict(),
            "configuration": self.configuration,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
