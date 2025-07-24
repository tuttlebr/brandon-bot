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
    "extract_web_content": "intelligent",
    "get_weather": "fast",
    "tavily_news_search": "intelligent",
    "tavily_internet_search": "llm",
    "retrieval_search": "intelligent",
    "retrieve_pdf_summary": "intelligent",
    "process_pdf_text": "intelligent",
    "text_assistant": "intelligent",
    "generate_image": "fast",
    "analyze_image": "vlm",
    "generalist_conversation": "llm",
    "tool_selection": "intelligent",
}

# Default LLM type if not specified
DEFAULT_LLM_TYPE = "llm"

# Tool-specific system prompt overrides
# These will be used instead of default prompts when tools make LLM calls
TOOL_SYSTEM_PROMPTS: Dict[str, str] = {
    "extract_web_content": """Extract and convert web content to clean markdown format.

Instructions:
1. Extract ONLY the main article/content from the webpage
2. Ignore navigation, ads, sidebars, headers, footers
3. Convert to clean markdown with proper structure
4. Preserve images, links, and formatting from main content
5. Return content verbatim, not summarized

Output only the extracted markdown content.""",
    "generate_image_enhancement": """Transform user requests into detailed image generation prompts.

Core rules:
- Focus on the user's ORIGINAL request as primary subject
- Add artistic details: lighting, composition, style, atmosphere
- Use vivid, descriptive language
- Keep prompts concise (1-2 sentences)
- Include quality indicators appropriate to style

Never change the core subject. Context is for subtle enhancement only.""",
    "document_analysis": """Analyze documents to answer specific questions concisely.

Provide focused, executive-summary style responses that directly address the user's question.""",
    "conversation_context_summary": """Summarize conversation history with focus on latest message.

Create a concise overview capturing main themes and objectives.

For the latest message, clearly identify:
- ACTION REQUEST: Asks for something to be done (create, generate, analyze, etc.)
- ACKNOWLEDGMENT: Thanks or comments on completed work
- MIXED: Both acknowledgment and new request

State explicitly: Does the latest message require action? YES or NO.""",
    "conversation_context_recent_topics": """List main topics from the conversation.

Extract primary discussion threads and current focus. Note recurring themes.""",
    "conversation_context_user_preferences": """Analyze user interaction patterns.

Identify communication style, request types, and preferences. Note expertise level.""",
    "conversation_context_task_continuity": """Track task progression and continuity.

Document main objective, completed steps, current stage, and likely next steps.""",
    "conversation_context_creative_director": """Maintain creative project continuity.

Track vision, goals, concept evolution, and style consistency.""",
    "conversation_context_document_analysis": """Analyze document content in conversation context.

Summarize key points, structure, and connections to user queries.""",
    "generalist_conversation": """You are a helpful AI assistant engaged in general conversation.

Provide thoughtful, informative responses for discussions, explanations, creative writing, and advice.
Draw from your knowledge to answer questions thoroughly and engagingly.""",
}


# Global system prompt prefix for all tool LLM calls
# This will be prepended to all tool-specific prompts
GLOBAL_TOOL_PROMPT_PREFIX = """detailed thinking on"""


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
