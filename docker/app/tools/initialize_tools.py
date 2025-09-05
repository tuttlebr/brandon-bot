"""
Tool Initialization Module

This module handles the initialization and registration of all tools
with the tool registry at application startup.
"""

import logging

from tools.registry import ToolRegistry, register_tool_class

logger = logging.getLogger(__name__)


def initialize_all_tools():
    """Initialize all tools at startup"""
    try:
        # Get the tool registry instance
        tool_registry = ToolRegistry.get_instance()

        # Check if tools are already initialized to prevent re-registration
        registered_tools = tool_registry._factory.get_registered_tools()
        if len(registered_tools) > 0:
            logger.debug(
                f"Tools already initialized ({len(registered_tools)} tools found), skipping re-initialization"
            )
            return

        # Import tool classes
        from tools.assistant import AssistantTool
        from tools.context_generation import ContextGenerationTool
        from tools.conversation_context import ConversationContextTool
        from tools.extract import WebExtractTool
        from tools.generalist import GeneralistTool
        from tools.image_analysis_tool import ImageAnalysisTool
        from tools.image_gen import ImageGenerationTool
        from tools.news import NewsTool
        from tools.pdf_assistant import (  # New unified PDF tool
            PDFAssistantTool,
        )
        from tools.retriever import RetrieverTool
        from tools.serpapi import SerpAPITool
        from tools.weather import WeatherTool

        # Register tool classes with lazy loading (instances created on demand)
        register_tool_class("text_assistant", AssistantTool)
        register_tool_class("context_generation", ContextGenerationTool)
        register_tool_class("conversation_context", ConversationContextTool)
        register_tool_class("extract_web_content", WebExtractTool)
        register_tool_class("generalist_conversation", GeneralistTool)
        register_tool_class("analyze_image", ImageAnalysisTool)
        register_tool_class("generate_image", ImageGenerationTool)
        register_tool_class("serpapi_news_search", NewsTool)
        register_tool_class(
            "pdf_assistant", PDFAssistantTool
        )  # Single PDF tool
        register_tool_class("retrieval_search", RetrieverTool)
        register_tool_class("serpapi_internet_search", SerpAPITool)
        register_tool_class("get_weather", WeatherTool)

        # Note: The old PDF tools (retrieve_pdf_summary, process_pdf_text) are deprecated
        # All PDF functionality is now handled by pdf_assistant

        logger.info(
            f"Successfully registered {len(tool_registry._factory.get_registered_tools())} tool classes"
        )

    except Exception as e:
        logger.error(f"Error initializing tools: {e}")
        raise
