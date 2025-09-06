"""
Tool LLM Configuration

This module documents the LLM type configuration for each tool in the system.
Tools can use "fast", "llm", "intelligent", or "vlm" models based on their
requirements.

All tool-specific prompts are now managed centrally in utils/system_prompts.py
"""

from utils.system_prompts import prompt_manager

# Tool LLM type configurations - matching actual tool names from tool classes
TOOL_LLM_TYPES = {
    "conversation_context": "fast",
    "extract_web_content": "fast",
    "get_weather": "fast",
    "serpapi_news_search": "llm",
    "serpapi_internet_search": "llm",
    "retrieval_search": "llm",
    "pdf_assistant": "fast",
    "text_assistant": "fast",
    "generate_image": "fast",
    "analyze_image": "vlm",
    "generalist_conversation": "llm",
    "tool_selection": "intelligent",
}

# Default LLM type if not specified
DEFAULT_LLM_TYPE = "llm"


def get_tool_llm_type(tool_name: str) -> str:
    """
    Get the LLM type for a specific tool or operation

    Args:
        tool_name: Name of the tool or operation

    Returns:
        LLM type: "fast", "llm", "intelligent", or "vlm"
    """
    return TOOL_LLM_TYPES.get(tool_name, DEFAULT_LLM_TYPE)


def get_tool_system_prompt(tool_name: str, default_prompt: str = None) -> str:
    """
    Get the system prompt for a tool's LLM calls

    Args:
        tool_name: Name of the tool
        default_prompt: (deprecated)

    Returns:
        The system prompt to use for this tool
    """
    # Delegate to centralized prompt manager
    return prompt_manager.get_tool_prompt(tool_name)


def configure_tool_prompt(tool_name: str, system_prompt: str) -> None:
    """
    Configure a custom system prompt for a specific tool

    DEPRECATED: Tool prompts should now be configured in
    utils/system_prompts.py

    Args:
        tool_name: Name of the tool to configure
        system_prompt: The system prompt to use for this tool's LLM calls
    """
    import warnings

    warnings.warn(
        "configure_tool_prompt is deprecated. Config utils/system_prompts.py",
        DeprecationWarning,
        stacklevel=2,
    )
