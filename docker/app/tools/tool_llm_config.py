"""
Tool LLM Configuration

This module documents the LLM type configuration for each tool in the system.
Tools can use "fast", "llm", "intelligent", or "vlm" models based on their requirements.

IMPORTANT: All tools and LLMs should return direct responses without meta-commentary.
Never reference tools, data sources, or provide phrases like "according to the search"
or "the tool shows". Answer directly as if you inherently know the information.
Tool context and sourcing is handled separately from the response content.
"""

from typing import Dict

# Tool LLM type configurations - matching actual tool names from tool classes
TOOL_LLM_TYPES = {
    "conversation_context": "llm",
    "extract_web_content": "llm",
    "get_weather": "fast",
    "tavily_news_search": "fast",
    "tavily_internet_search": "fast",
    "retrieval_search": "fast",
    "retrieve_pdf_summary": "fast",
    "process_pdf_text": "fast",
    "text_assistant": "llm",
    "generate_image": "fast",
    "analyze_image": "vlm",
    "generalist_conversation": "fast",
    "tool_selection": "intelligent",
}

# Default LLM type if not specified
DEFAULT_LLM_TYPE = "llm"

# Tool-specific system prompt overrides
# These will be used instead of default prompts when tools make LLM calls
TOOL_SYSTEM_PROMPTS: Dict[str, str] = {}

# Global system prompt prefix for all tool LLM calls
# This will be prepended to all tool-specific prompts
GLOBAL_TOOL_PROMPT_PREFIX = """detailed thinking off - You are an internal tool component processing data.
Return only the requested information without meta-commentary or explanations about your process.
Never mention that you are a tool or reference data sources. You have access to real-time data and the internet."""


def get_tool_llm_type(tool_name: str) -> str:
    """
    Get the LLM type for a specific tool or operation

    Args:
        tool_name: Name of the tool or operation

    Returns:
        LLM type: "fast", "llm", "intelligent", or "vlm"
    """
    return TOOL_LLM_TYPES.get(tool_name, DEFAULT_LLM_TYPE)


def get_tool_system_prompt(tool_name: str, default_prompt: str) -> str:
    """
    Get the system prompt for a tool's LLM calls

    Args:
        tool_name: Name of the tool
        default_prompt: The default prompt to use if no override exists

    Returns:
        The system prompt to use for this tool
    """
    # Check for tool-specific override
    if tool_name in TOOL_SYSTEM_PROMPTS:
        return TOOL_SYSTEM_PROMPTS[tool_name]

    # Otherwise, use default with global prefix if configured
    if GLOBAL_TOOL_PROMPT_PREFIX:
        return f"{GLOBAL_TOOL_PROMPT_PREFIX}\n\n{default_prompt}"

    return default_prompt


def configure_tool_prompt(tool_name: str, system_prompt: str) -> None:
    """
    Configure a custom system prompt for a specific tool

    Args:
        tool_name: Name of the tool to configure
        system_prompt: The system prompt to use for this tool's LLM calls
    """
    TOOL_SYSTEM_PROMPTS[tool_name] = system_prompt
