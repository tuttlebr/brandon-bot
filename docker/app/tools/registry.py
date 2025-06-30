"""
Tool Registry - Central registry for all available tools
This module prevents circular imports between system_prompt.py and llm_service.py
"""

from typing import Any, Dict, List

from .assistant import get_assistant_tool_definition
from .conversation_context import get_conversation_context_tool_definition
from .default_fallback import get_default_fallback_tool_definition
from .image_gen import get_image_generation_tool_definition
from .news import get_news_tool_definition
from .pdf_parser import get_pdf_parser_tool_definition
from .retriever import get_retrieval_tool_definition
from .tavily import get_tavily_tool_definition
from .weather import get_weather_tool_definition


def get_all_tool_definitions() -> List[Dict[str, Any]]:
    """Get all registered tool definitions"""
    return [
        get_assistant_tool_definition(),
        get_conversation_context_tool_definition(),
        get_default_fallback_tool_definition(),
        get_image_generation_tool_definition(),
        get_news_tool_definition(),
        get_pdf_parser_tool_definition(),
        get_retrieval_tool_definition(),
        get_tavily_tool_definition(),
        get_weather_tool_definition(),
    ]


def get_tools_list_text() -> str:
    """Generate formatted tool list text for prompts"""
    tools_text = []
    for tool_def in get_all_tool_definitions():
        if isinstance(tool_def, dict) and "function" in tool_def:
            function_info = tool_def["function"]
            name = function_info.get("name", "Unknown")
            description = function_info.get("description", "No description available")
            tools_text.append(f"- {name}: {description}")

    return "\n".join(tools_text)
