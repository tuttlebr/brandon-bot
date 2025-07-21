"""
Tool Registry - Factory Pattern Implementation

This module provides a centralized registry for managing tools
following the Factory and Singleton patterns with dependency injection support.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolFactory:
    """Factory for creating tool instances with dependency injection"""

    def __init__(self):
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        self._dependencies: Dict[str, Dict[str, Any]] = {}

    def register_tool_class(
        self,
        name: str,
        tool_class: Type[BaseTool],
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a tool class with optional configuration and dependencies

        Args:
            name: Tool name
            tool_class: Tool class type
            config: Optional configuration for the tool
            dependencies: Optional dependencies to inject
        """
        self._tool_classes[name] = tool_class
        if config:
            self._tool_configs[name] = config
        if dependencies:
            self._dependencies[name] = dependencies
        logger.info(f"Registered tool class: {name}")

    def create_tool(self, name: str) -> Optional[BaseTool]:
        """
        Create a tool instance with injected dependencies

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        tool_class = self._tool_classes.get(name)
        if not tool_class:
            logger.error(f"Tool class not found: {name}")
            return None

        try:
            # Get dependencies and config
            deps = self._dependencies.get(name, {})
            config = self._tool_configs.get(name, {})

            # Create tool instance with dependencies
            if deps:
                tool = tool_class(**deps)
            else:
                tool = tool_class()

            # Apply configuration
            for key, value in config.items():
                if hasattr(tool, key):
                    setattr(tool, key, value)

            return tool
        except Exception as e:
            logger.error(f"Failed to create tool {name}: {e}")
            return None

    def get_registered_tools(self) -> List[str]:
        """Get list of registered tool names"""
        return list(self._tool_classes.keys())


class ToolRegistry:
    """Singleton registry for managing all tools with factory pattern"""

    _instance: Optional['ToolRegistry'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent reinitialization
        if self._initialized:
            return

        self._factory = ToolFactory()
        self._tools: Dict[str, BaseTool] = {}
        self._context_mapping: Dict[str, str] = {}
        self._initialized = True
        logger.info("ToolRegistry initialized")

    @classmethod
    def get_instance(cls) -> 'ToolRegistry':
        """Get singleton instance of ToolRegistry"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_tool_class(
        self,
        name: str,
        tool_class: Type[BaseTool],
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        lazy_load: bool = True,
    ) -> None:
        """
        Register a tool class with the registry

        Args:
            name: Tool name
            tool_class: Tool class type
            config: Optional configuration
            dependencies: Optional dependencies
            lazy_load: If True, create instance only when needed
        """
        self._factory.register_tool_class(name, tool_class, config, dependencies)

        if not lazy_load:
            # Create instance immediately
            tool = self._factory.create_tool(name)
            if tool:
                self._register_instance(tool)

    def _register_instance(self, tool: BaseTool) -> None:
        """Register a tool instance"""
        if not tool.name:
            logger.error("Cannot register tool without a name")
            return

        self._tools[tool.name] = tool

        # Update context mapping
        for context in tool.supported_contexts:
            self._context_mapping[context] = tool.name

        logger.info(f"Registered tool instance: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name (lazy loading if needed)

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        # Check if already instantiated
        if name in self._tools:
            return self._tools[name]

        # Try to create instance if class is registered
        tool = self._factory.create_tool(name)
        if tool:
            self._register_instance(tool)
            return tool

        logger.warning(f"Tool not found: {name}")
        return None

    def get_tool_by_context(self, context: str) -> Optional[BaseTool]:
        """
        Get a tool by context

        Args:
            context: Context string

        Returns:
            Tool instance or None if not found
        """
        tool_name = self._context_mapping.get(context)
        if tool_name:
            return self.get_tool(tool_name)

        logger.warning(f"No tool found for context: {context}")
        return None

    def get_all_supported_contexts(self) -> List[str]:
        """Get list of all supported contexts"""
        return list(self._context_mapping.keys())

    def execute_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool by name

        Args:
            name: Tool name
            params: Parameters for the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found or disabled
            Exception: If tool execution fails
        """
        from utils.config import config

        # Check if tool is enabled
        if not config.tools.is_tool_enabled(name):
            raise ValueError(f"Tool '{name}' is disabled in configuration")

        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        try:
            return tool.execute(params)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            raise

    def get_all_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible definitions for all registered tools"""
        from utils.config import config

        definitions = []

        # Get all registered tool names (including lazy-loaded ones)
        all_tool_names = set(self._tools.keys()) | set(
            self._factory.get_registered_tools()
        )

        for name in all_tool_names:
            # Check if tool is enabled in configuration
            if not config.tools.is_tool_enabled(name):
                logger.debug(f"Tool '{name}' is disabled in configuration, skipping")
                continue

            tool = self.get_tool(name)
            if tool:
                try:
                    definition = tool.get_definition()
                    definitions.append(definition)
                    logger.debug(f"Added tool definition for '{name}'")
                except Exception as e:
                    logger.error(f"Error getting definition for tool {name}: {e}")

        logger.info(
            f"Returning {len(definitions)} tool definitions (out of {len(all_tool_names)} registered)"
        )
        return definitions

    def get_tools_list_text(self) -> str:
        """Get formatted text list of all available tools"""
        from utils.config import config

        all_tool_names = sorted(
            set(self._tools.keys()) | set(self._factory.get_registered_tools())
        )

        if not all_tool_names:
            return "No tools available."

        lines = ["Available tools:"]
        enabled_count = 0

        for name in all_tool_names:
            # Check if tool is enabled
            if not config.tools.is_tool_enabled(name):
                continue

            tool = self.get_tool(name)
            if tool:
                description = tool.description or "No description available"
                lines.append(f"- {name}: {description}")
                enabled_count += 1

        if enabled_count == 0:
            return "No tools are currently enabled."

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all registered tools (useful for testing)"""
        self._tools.clear()
        self._context_mapping.clear()
        self._factory = ToolFactory()
        logger.info("Cleared all registered tools")


# Global registry instance (for backward compatibility)
_registry = ToolRegistry.get_instance()


# Public API functions
def register_tool_class(
    name: str,
    tool_class: Type[BaseTool],
    config: Optional[Dict[str, Any]] = None,
    dependencies: Optional[Dict[str, Any]] = None,
    lazy_load: bool = True,
) -> None:
    """Register a tool class with the global registry"""
    _registry.register_tool_class(name, tool_class, config, dependencies, lazy_load)


def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool from the global registry"""
    return _registry.get_tool(name)


def get_all_tool_definitions() -> List[Dict[str, Any]]:
    """Get all tool definitions from the global registry"""
    return _registry.get_all_definitions()


def get_tools_list_text() -> str:
    """Get formatted list of all tools from the global registry"""
    return _registry.get_tools_list_text()


def execute_tool(name: str, params: Dict[str, Any]) -> Any:
    """Execute a tool from the global registry"""
    return _registry.execute_tool(name, params)


def get_tool_by_context(context: str) -> Optional[BaseTool]:
    """Get a tool by context from the global registry"""
    return _registry.get_tool_by_context(context)


def get_all_supported_contexts() -> List[str]:
    """Get all supported contexts from the global registry"""
    return _registry.get_all_supported_contexts()
