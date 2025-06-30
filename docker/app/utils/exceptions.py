"""
Custom Exception Classes

This module defines custom exceptions for better error handling
and debugging throughout the application.
"""


class ChatbotException(Exception):
    """Base exception for all chatbot-related errors"""

    pass


class ConfigurationError(ChatbotException):
    """Raised when there's an error in configuration"""

    pass


class ToolExecutionError(ChatbotException):
    """Raised when a tool fails to execute"""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class LLMServiceError(ChatbotException):
    """Raised when LLM service encounters an error"""

    pass


class StreamingError(ChatbotException):
    """Raised when streaming response fails"""

    pass


class ValidationError(ChatbotException):
    """Raised when input validation fails"""

    pass


class SessionStateError(ChatbotException):
    """Raised when session state operations fail"""

    pass


class FileProcessingError(ChatbotException):
    """Raised when file processing fails"""

    pass


class MemoryLimitError(ChatbotException):
    """Raised when memory limits are exceeded"""

    pass
