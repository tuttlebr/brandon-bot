"""
Tool LLM Configuration

This module documents the LLM type configuration for each tool in the system.
Tools can use "fast", "llm", or "intelligent" models based on their requirements.
"""

# Tool LLM type configurations
TOOL_LLM_TYPES = {
    "conversation_context": "intelligent",  # Quick context analysis
    "extract_web_content": "fast",  # Simple API calls and formatting
    "weather": "fast",  # Simple API calls and formatting
    "news": "llm",  # Simple API calls and formatting
    "tavily": "fast",  # Complex web search and synthesis
    "retriever": "llm",  # Semantic search and retrieval
    "pdf_summary": "llm",  # PDF summarization
    "pdf_text_processor": "llm",  # PDF text processing
    "text_assistant": "intelligent",  # Complex text processing tasks
    "image_gen": "fast",  # Complex prompt enhancement for image generation
    "analyze_image": "vlm",  # Vision analysis requires vlm model
    "tool_selection": "llm",  # Tool selection, no reasoning - determines which tools to use for user queries
}

# Default LLM type if not specified
DEFAULT_LLM_TYPE = "llm"


def get_tool_llm_type(tool_name: str) -> str:
    """
    Get the LLM type for a specific tool or operation

    Args:
        tool_name: Name of the tool or operation

    Returns:
        LLM type: "fast", "llm", or "intelligent"
    """
    return TOOL_LLM_TYPES.get(tool_name, DEFAULT_LLM_TYPE)
