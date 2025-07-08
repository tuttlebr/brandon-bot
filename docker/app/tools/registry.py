"""
Tool Registry - Central registry for all available tools

This module implements a singleton registry pattern for managing
all tool instances in the application.
"""

import logging
from typing import Any, Dict, List, Optional

from tools.base import BaseTool
from utils.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Singleton registry for managing all tools"""

    _instance: Optional['ToolRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._tools: Dict[str, BaseTool] = {}
        self._initialized = True
        logger.info("Tool registry initialized")

    @classmethod
    def get_instance(cls) -> 'ToolRegistry':
        """Get the singleton instance of the tool registry"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool in the registry

        Args:
            tool: Tool instance to register
        """
        if not tool.name:
            raise ValueError("Tool must have a name")

        if tool.name in self._tools:
            logger.debug(f"Tool '{tool.name}' is already registered, overwriting")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name

        Args:
            name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def get_tool_by_context(self, context: str) -> Optional[BaseTool]:
        """
        Get a tool that supports the given context

        Args:
            context: Context name to find a tool for

        Returns:
            Tool instance that supports this context or None if not found
        """
        for tool in self._tools.values():
            if (
                hasattr(tool, 'supported_contexts')
                and context in tool.supported_contexts
            ):
                return tool
        return None

    def get_all_supported_contexts(self) -> List[str]:
        """
        Get all unique contexts supported by registered tools

        Returns:
            List of unique context names
        """
        contexts = set()
        for tool in self._tools.values():
            if hasattr(tool, 'supported_contexts'):
                contexts.update(tool.supported_contexts)
        return sorted(list(contexts))

    def execute_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool by name

        Args:
            name: Name of the tool
            params: Parameters for the tool

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If tool not found or execution fails
        """
        tool = self.get_tool(name)
        if not tool:
            raise ToolExecutionError(name, f"Tool '{name}' not found in registry")

        try:
            return tool.execute(params)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            raise ToolExecutionError(name, str(e))

    def get_all_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all tool definitions in OpenAI format

        Returns:
            List of tool definitions
        """
        return [tool.get_definition() for tool in self._tools.values()]

    def get_tools_list_text(self) -> str:
        """
        Generate formatted tool list text for prompts

        Returns:
            Formatted string of available tools
        """
        logger.debug(
            f"Generating tools list text. Registry has {len(self._tools)} tools registered."
        )
        tools_text = []
        for name, tool in self._tools.items():
            logger.debug(f"Adding tool to list: {name} - {tool.description}")
            tools_text.append(f"- {name}: {tool.description}")
        result = "\n".join(tools_text)
        logger.debug(f"Generated tools list text: {result}")
        return result

    def clear(self) -> None:
        """Clear all registered tools (mainly for testing)"""
        self._tools.clear()
        logger.info("Tool registry cleared")


# Global registry instance
tool_registry = ToolRegistry()


# Helper functions for backward compatibility
def get_all_tool_definitions() -> List[Dict[str, Any]]:
    """Get all registered tool definitions"""
    return tool_registry.get_all_definitions()


def get_tools_list_text() -> str:
    """Generate formatted tool list text for prompts"""
    return tool_registry.get_tools_list_text()
