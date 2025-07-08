"""
Custom Exception Classes

This module defines custom exceptions for better error handling
and debugging throughout the application.
"""


class ChatbotException(Exception):
    """Base exception for all chatbot-related errors"""


class ConfigurationError(ChatbotException):
    """Raised when there's an error in configuration"""


class ToolExecutionError(ChatbotException):
    """Raised when a tool fails to execute"""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class LLMServiceError(ChatbotException):
    """Raised when LLM service encounters an error"""


class StreamingError(ChatbotException):
    """Raised when streaming response fails"""


class ValidationError(ChatbotException):
    """Raised when input validation fails"""


class FileProcessingError(ChatbotException):
    """Raised when file processing fails"""


class MemoryLimitError(ChatbotException):
    """Raised when memory limits are exceeded"""
