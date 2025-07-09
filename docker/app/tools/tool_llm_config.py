"""
Tool LLM Configuration

This module documents the LLM type configuration for each tool in the system.
Tools can use "fast", "llm", "intelligent", or "vlm" models based on their requirements.
"""

# Tool LLM type configurations - matching actual tool names from tool classes
TOOL_LLM_TYPES = {
    "conversation_context": "llm",
    "extract_web_content": "fast",
    "get_weather": "fast",
    "tavily_news_search": "llm",
    "tavily_internet_search": "fast",
    "retrieval_search": "llm",
    "retrieve_pdf_summary": "intelligent",
    "process_pdf_text": "intelligent",
    "text_assistant": "intelligent",
    "generate_image": "fast",
    "analyze_image": "vlm",
    "tool_selection": "llm",
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
