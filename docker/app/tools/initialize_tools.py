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
                "Tools already initialized (%d tools found), "
                "skipping re-initialization",
                len(registered_tools),
            )
            return

        # Import tool classes
        from tools.assistant import AssistantTool
        from tools.context_generation import ContextGenerationTool
        from tools.conversation_context import ConversationContextTool
        from tools.deepresearcher import DeepResearcherTool
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
        register_tool_class("deep_researcher", DeepResearcherTool)
        register_tool_class("extract_web_content", WebExtractTool)
        register_tool_class("generalist_conversation", GeneralistTool)
        register_tool_class("analyze_image", ImageAnalysisTool)
        register_tool_class("generate_image", ImageGenerationTool)
        register_tool_class("serpapi_news_search", NewsTool)
        register_tool_class("pdf_assistant", PDFAssistantTool)
        register_tool_class("retrieval_search", RetrieverTool)
        register_tool_class("serpapi_internet_search", SerpAPITool)
        register_tool_class("get_weather", WeatherTool)

        tool_count = len(tool_registry._factory.get_registered_tools())
        logger.info("Successfully registered %d tool classes", tool_count)

    except Exception as e:
        logger.error("Error initializing tools: %s", e)
        raise
