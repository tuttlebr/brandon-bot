"""
Tool Initialization Module

This module handles the initialization and registration of all tools
with the tool registry at application startup.
"""

import logging
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def initialize_all_tools():
    """Initialize all tools at startup"""
    try:
        # Get the tool registry instance
        tool_registry = ToolRegistry.get_instance()

        # Check if tools are already initialized to prevent re-registration
        if len(tool_registry._tools) > 0:
            logger.debug(
                f"Tools already initialized ({len(tool_registry._tools)} tools found), skipping re-initialization"
            )
            return

        # Clear any existing tools (shouldn't be any if check above works)
        tool_registry._tools.clear()

        # Assistant Tool
        from tools.assistant import AssistantTool

        assistant = AssistantTool()
        tool_registry.register(assistant)

        # Conversation Context Tool
        from tools.conversation_context import ConversationContextTool

        context = ConversationContextTool()
        tool_registry.register(context)

        # Default Fallback Tool
        from tools.default_fallback import DefaultFallbackTool

        fallback = DefaultFallbackTool()
        tool_registry.register(fallback)

        # Image Generation Tool
        from tools.image_gen import ImageGenerationTool

        image_gen = ImageGenerationTool()
        tool_registry.register(image_gen)

        # News Tool
        from tools.news import NewsTool

        news = NewsTool()
        tool_registry.register(news)

        # PDF Summary Tool
        from tools.pdf_summary import PDFSummaryTool

        pdf_summary = PDFSummaryTool()
        tool_registry.register(pdf_summary)

        # PDF Text Processor Tool
        from tools.pdf_text_processor import PDFTextProcessorTool

        pdf_text_processor = PDFTextProcessorTool()
        tool_registry.register(pdf_text_processor)

        # Retriever Tool
        from tools.retriever import RetrieverTool

        retriever = RetrieverTool()
        tool_registry.register(retriever)

        # Tavily Tool
        from tools.tavily import TavilyTool

        tavily = TavilyTool()
        tool_registry.register(tavily)

        # Weather Tool
        from tools.weather import WeatherTool

        weather = WeatherTool()
        tool_registry.register(weather)

        logger.info(f"Successfully initialized {len(tool_registry._tools)} tools")

    except Exception as e:
        logger.error(f"Error initializing tools: {e}")
        raise


def get_tool_by_name(name: str):
    """
    Get a tool instance by name

    Args:
        name: Tool name

    Returns:
        Tool instance or None
    """
    registry = ToolRegistry.get_instance()
    return registry.get_tool(name)


# Optional: Function to get all initialized tool names
def get_initialized_tool_names():
    """Get list of all initialized tool names"""
    registry = ToolRegistry.get_instance()
    return list(registry._tools.keys())


# Optional: Function to verify all tools are properly initialized
def verify_tools_initialized():
    """Verify all tools have been properly initialized"""
    registry = ToolRegistry.get_instance()
    tool_count = len(registry._tools)
    if tool_count == 0:
        raise RuntimeError("No tools have been initialized!")
    return tool_count
