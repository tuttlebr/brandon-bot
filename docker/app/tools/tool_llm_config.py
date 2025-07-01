"""
Tool LLM Configuration

This module documents the LLM type configuration for each tool in the system.
Tools can use "fast", "llm", or "intelligent" models based on their requirements.
"""

# Tool LLM type configurations
TOOL_LLM_TYPES = {
    "default_fallback": "fast",  # General conversation responses
    "conversation_context": "llm",  # Quick context analysis
    "weather": "fast",  # Simple API calls and formatting
    "news": "llm",  # Simple API calls and formatting
    "tavily": "fast",  # Complex web search and synthesis
    "retriever": "llm",  # Semantic search and retrieval
    "pdf_summary": "llm",  # PDF summarization
    "pdf_text_processor": "llm",  # PDF text processing
    "text_assistant": "intelligent",  # Complex text processing tasks
    "image_gen": "fast",  # Complex prompt enhancement for image generation
}

# Default LLM type if not specified
DEFAULT_LLM_TYPE = "llm"
