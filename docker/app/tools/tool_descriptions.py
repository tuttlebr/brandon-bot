"""
Tool Descriptions and Selection Guidelines

This module provides the SINGLE SOURCE OF TRUTH for all tool descriptions,
metadata, and decision logic to help the primary tool-calling LLM make better
tool selection choices.

METADATA FIELD REFERENCE:
- description: (required) Clear description of what the tool does
- trigger_words: List of words that indicate this tool should be used
- anti_trigger_words: List of words that indicate this tool should NOT be used
- example_uses: List of example user queries where this tool SHOULD be used
- example_non_uses: List of example queries where this tool should NOT be used
- context_requirement: Type of context required (e.g., "uploaded_image", "uploaded_pdf")
- requires_*: Boolean flags for specific requirements (e.g., requires_location, requires_url)
- is_*: Boolean flags for tool characteristics (e.g., is_default, is_internal, is_multi_turn)
- specialized_domains: List of domains this tool specializes in
- strict_domain_check: Whether to strictly enforce domain specialization
"""

from typing import Dict, List, Optional

# SINGLE SOURCE OF TRUTH for all tool descriptions and metadata
TOOL_DEFINITIONS = {
    "generate_image": {
        "description": (
            "Generate AI images from text descriptions. Use when user requests"
            " creating, generating, making, or drawing images."
        ),
        "trigger_words": [
            "generate",
            "create",
            "make",
            "draw",
            "design",
            "produce",
            "render",
        ],
        "anti_trigger_words": [
            "thanks",
            "thank you",
            "great",
            "nice",
            "good",
            "perfect",
            "looks",
            "is",
            "what",
            "how",
            "why",
            "can you",
        ],
        "requires_action_verb": True,
        "example_uses": [
            "Generate an image of a sunset",
            "Create a picture of a cat",
            "Make me a logo design",
        ],
        "example_non_uses": [
            "Thanks! (acknowledgment - no tool needed)",
            "That looks great (comment - no tool needed)",
            "Perfect! (acknowledgment - no tool needed)",
        ],
    },
    "text_assistant": {
        "description": (
            "Process text with specific operations: summarize, translate,"
            " proofread, rewrite, analyze documents, or develop code. Use when"
            " user provides text AND requests processing."
        ),
        "trigger_words": [
            "analyze",
            "summarize",
            "proofread",
            "rewrite",
            "translate",
            "critic",
            "develop",
            "process",
            "edit",
        ],
        "anti_trigger_words": [
            "thanks",
            "what is",
            "how does",
            "tell me about",
            "explain",
        ],
        "requires_target_text": True,
        "example_uses": [
            "Summarize this article: [text]",
            "Proofread my essay: [text]",
            "Translate this to Spanish: [text]",
            "Analyze this document: [text]",
        ],
        "example_non_uses": [
            "What is machine learning? (general question - no tool needed)",
            "How are you? (conversation - no tool needed)",
            "Thanks for the help (acknowledgment - no tool needed)",
        ],
    },
    "analyze_image": {
        "description": (
            "Analyze uploaded images to describe content or answer visual"
            " questions. Use when user uploads an image AND asks about it."
        ),
        "trigger_words": [
            "look at",
            "analyze",
            "describe",
            "what's in",
            "identify",
            "examine",
        ],
        "context_requirement": "uploaded_image",
        "example_uses": [
            "What's in this image? (with uploaded image)",
            "Describe what you see (with uploaded image)",
            "Is there a cat in this picture? (with uploaded image)",
        ],
        "example_non_uses": [
            "Generate an image (use generate_image instead)",
            "What is a cat? (no image uploaded - no tool needed)",
            "Thanks for analyzing (acknowledgment - no tool needed)",
        ],
    },
    "serpapi_internet_search": {
        "description": (
            "Search the internet for current information and real-time data."
            " Use for up-to-date facts, current events, or information that"
            " changes frequently."
        ),
        "trigger_words": [
            "search",
            "find",
            "look up",
            "current",
            "latest",
            "recent",
            "today",
            "now",
        ],
        "anti_trigger_words": [
            "thanks",
            "what is",
            "explain",
            "tell me about",
        ],
        "requires_specific_query": True,
        "example_uses": [
            "What's happening in tech today?",
            "How late is Domino's pizza open tonight?",
        ],
        "example_non_uses": [
            "What is Python? (general knowledge - no tool needed)",
            "How does gravity work? (general knowledge - no tool needed)",
            "Thanks for the search results (acknowledgment - no tool needed)",
        ],
    },
    "generalist_conversation": {
        "description": (
            "DEFAULT: Handle general conversation without external tools. Use"
            " for explanations, discussions, advice, and casual chat. Do not"
            " use for creative writing or when the user asks about the bots"
            " capabilities. Limit verbosity and be direct. You have access to"
            " all tools."
        ),
        "is_default": True,
        "trigger_words": [
            "explain",
            "tell me about",
            "what is",
            "how does",
            "why",
            "describe",
            "discuss",
        ],
        "example_uses": [
            "What is machine learning?",
            "Tell me about philosophy",
            "Can you explain quantum physics?",
            "Thanks for your help!",
            "That's interesting!",
            "How are you?",
        ],
        "example_non_uses": [
            "Generate an image of... (use generate_image)",
            "Search for latest news (use serpapi_news_search)",
            "What's the weather? (use get_weather)",
        ],
    },
    "get_weather": {
        "description": (
            "Get current weather for a specific location. Use when user asks"
            " for weather, temperature, or forecast AND provides a"
            " city/location."
        ),
        "trigger_words": [
            "weather",
            "temperature",
            "forecast",
            "rain",
            "snow",
            "sunny",
            "cloudy",
        ],
        "requires_location": True,
        "example_uses": [
            "What's the weather in Paris?",
            "Is it raining in Seattle?",
            "Temperature in Tokyo today?",
        ],
        "example_non_uses": [
            "Thanks for the weather info (acknowledgment - no tool needed)",
            "What causes rain? (general question - no tool needed)",
            "Tell me about weather patterns (general topic - no tool needed)",
        ],
    },
    "retrieval_search": {
        "description": (
            "NVIDIA-ONLY technical documentation knowledge base. "
            "Contains proprietary NVIDIA product specs, CUDA docs, "
            "GPU architectures, and internal technical resources. "
            "Use EXCLUSIVELY for NVIDIA hardware/software questions. "
            "For ANY other company's products or general AI/ML topics, "
            "use serpapi_internet_search instead."
        ),
        "trigger_words": [
            "nvidia",
            "gpu",
            "cuda",
            "tensorrt",
            "cudnn",
            "nvlink",
            "dgx",
            "h100",
            "a100",
            "rtx",
            "geforce",
        ],
        "anti_trigger_words": [
            "anthropic",
            "openai",
            "google",
            "meta",
            "microsoft",
            "amazon",
            "claude",
            "gpt",
            "gemini",
            "llama",
        ],
        "specialized_domains": ["nvidia"],
        "strict_domain_check": True,
        "example_uses": [
            "How many GPUs are in a NVL72 GB200?",
            "What is CUDA programming?",
            "Tell me about NVIDIA H100 specifications",
            "How does NVLink work?",
            "What's the memory bandwidth of A100?",
        ],
        "example_non_uses": [
            (
                "Does Anthropic have a model with 1M context? (use"
                " serpapi_internet_search)"
            ),
            "What is OpenAI's latest model? (use serpapi_internet_search)",
            (
                "How does transformer architecture work? (general knowledge -"
                " no tool)"
            ),
            (
                "What are the latest AI developments? (use"
                " serpapi_internet_search)"
            ),
            "Compare Claude vs GPT-4 (use serpapi_internet_search)",
        ],
    },
    "conversation_context": {
        "description": (
            "INTERNAL SYSTEM TOOL: Analyze conversation history for context."
            " Never select for user queries."
        ),
        "is_internal": True,
        "never_use_for_user_queries": True,
    },
    "extract_web_content": {
        "description": (
            "Extract and read content from a specific URL. Use when user"
            " provides a URL AND asks to read or analyze it."
        ),
        "trigger_words": ["read", "extract", "analyze", "check", "look at"],
        "requires_url": True,
        "example_uses": [
            "Read this article: https://example.com",
            "Extract content from https://example.com",
            "What does this webpage say: [URL]",
        ],
        "example_non_uses": [
            "Search for information about X (use serpapi_internet_search)",
            "Find me articles about Y (use serpapi_internet_search)",
            "Thanks! (acknowledgment - no tool needed)",
        ],
    },
    "pdf_assistant": {
        "description": (
            "Handle ALL PDF-related operations: summarization, Q&A, page"
            " extraction, and analysis. Use whenever user asks about an"
            " uploaded PDF."
        ),
        "context_requirement": "uploaded_pdf",
        "trigger_words": [
            "pdf",
            "document",
            "file",
            "page",
            "summary",
            "summarize",
            "extract",
            "analyze",
            "text",
            "content",
            "says",
            "mentions",
            "according to",
            "what does",
            "find",
            "search",
        ],
        "anti_trigger_words": [
            "generate pdf",
            "create pdf",
            "make pdf",
        ],
        "requires_uploaded_file": True,
        "example_uses": [
            "Summarize this PDF",
            "What does the document say about X?",
            "Extract page 5",
            "Analyze the methodology section",
            "What are the key findings?",
            "Search for information about Y in the PDF",
        ],
        "example_non_uses": [
            "Create a PDF file (not supported)",
            "Generate a PDF report (not supported)",
            "Thanks for the summary (acknowledgment - no tool needed)",
        ],
    },
    "serpapi_news_search": {
        "description": (
            "Search specifically for news articles and breaking events. Use"
            " when user explicitly asks for news, headlines, or current"
            " events."
        ),
        "trigger_words": [
            "news",
            "headlines",
            "breaking",
            "events",
            "happening",
            "latest news",
            "current events",
        ],
        "anti_trigger_words": [
            "history",
            "explain",
            "general",
        ],
        "is_specialized_search": True,
        "example_uses": [
            "What's in the news today?",
            "Latest headlines about AI",
            "Breaking news from tech industry",
        ],
        "example_non_uses": [
            "History of newspapers (general knowledge - no tool needed)",
            (
                "How does news reporting work? (general knowledge - no tool"
                " needed)"
            ),
            "Thanks for the news (acknowledgment - no tool needed)",
        ],
    },
    "context_generation": {
        "description": (
            "Generate or modify images based on an existing image and text"
            " prompt. Use when user wants to edit/transform an existing"
            " image or create variations. Requires an uploaded image."
            " Supports OpenAI's image edit API when configured."
        ),
        "trigger_words": [
            "edit",
            "modify",
            "transform",
            "change",
            "alter",
            "adjust",
            "variation",
            "based on",
        ],
        "anti_trigger_words": [
            "generate new",
            "create new",
            "make new",
        ],
        "context_requirement": "uploaded_image",
        "requires_existing_image": True,
        "example_uses": [
            "Change the background of this image to sunset",
            "Make this person wear a hat",
            "Transform this image to winter scene",
            "Create a variation of this image",
        ],
        "example_non_uses": [
            "Generate a new image (use generate_image instead)",
            "Create an image from scratch (use generate_image instead)",
            "Thanks for the edit (acknowledgment - no tool needed)",
        ],
    },
    "deep_researcher": {
        "description": (
            "Perform deep, multi-turn research on complex topics."
            " This tool makes multiple iterations using search, extraction,"
            " and analysis tools to provide comprehensive, well-researched"
            " answers with citations. Use for questions requiring in-depth"
            " research, fact-checking, or comprehensive analysis."
        ),
        "trigger_words": [
            "research",
            "investigate",
            "deep dive",
            "comprehensive",
            "fact check",
            "verify",
            "detailed analysis",
            "thorough",
        ],
        "anti_trigger_words": [
            "quick",
            "simple",
            "basic",
            "brief",
        ],
        "requires_complex_query": True,
        "is_multi_turn": True,
        "example_uses": [
            "Research the impact of climate change on agriculture",
            "Investigate the history and development of quantum computing",
            "Do a deep dive on renewable energy technologies",
            "Fact-check claims about vaccine effectiveness",
            "Comprehensive analysis of cryptocurrency regulations worldwide",
        ],
        "example_non_uses": [
            "What is Python? (simple question - use generalist_conversation)",
            "Current weather (use get_weather instead)",
            "Generate an image (use generate_image instead)",
            "Quick summary of this text (use text_assistant instead)",
        ],
    },
}


def get_tool_description(tool_name: str) -> str:
    """
    Get the description for a specific tool from the single source of truth.

    Args:
        tool_name: The name of the tool

    Returns:
        The tool description, or a default message if not found
    """
    if tool_name in TOOL_DEFINITIONS:
        return TOOL_DEFINITIONS[tool_name]["description"]
    return f"Tool '{tool_name}' - No description available"


# Acknowledgment patterns that should NOT trigger tools
ACKNOWLEDGMENT_PATTERNS = [
    "thanks",
    "thank you",
    "thx",
    "ty",
    "great",
    "awesome",
    "perfect",
    "nice",
    "good",
    "okay",
    "ok",
    "cool",
    "interesting",
    "got it",
    "understood",
    "I see",
    "that's helpful",
    "that helps",
    "appreciate it",
    "well done",
    "excellent",
    "fantastic",
    "that's great",
    "that's perfect",
    "that's nice",
    "wonderful",
    "amazing",
    "brilliant",
]


def is_acknowledgment(message: str) -> bool:
    """
    Check if a message is primarily an acknowledgment

    Args:
        message: The user's message

    Returns:
        True if the message is an acknowledgment
    """
    message_lower = message.lower().strip()

    # Very short messages that match acknowledgment patterns
    if len(message.split()) <= 3:
        for pattern in ACKNOWLEDGMENT_PATTERNS:
            if pattern in message_lower:
                return True

    # Check if the message starts with acknowledgment
    for pattern in ACKNOWLEDGMENT_PATTERNS:
        if message_lower.startswith(pattern):
            # Check if there's a follow-up request after the acknowledgment
            # e.g., "Thanks! Now generate another image"
            if (
                "now" in message_lower
                or "next" in message_lower
                or "another" in message_lower
            ):
                return False
            return True

    return False


def extract_actual_request(message: str) -> Optional[str]:
    """
    Extract the actual request from a message that might contain acknowledgment

    Args:
        message: The user's message

    Returns:
        The actual request part, or None if it's just acknowledgment
    """
    message_lower = message.lower()

    # Common patterns that separate acknowledgment from new request
    separators = [
        "now",
        "next",
        "also",
        "and",
        "but",
        "can you",
        "could you",
        "please",
    ]

    for separator in separators:
        if separator in message_lower:
            parts = message.split(separator, 1)
            if len(parts) > 1 and len(parts[1].strip()) > 5:
                return parts[1].strip()

    return None


def validate_tool_definitions() -> Dict[str, List[str]]:
    """
    Validate that all tool definitions have comprehensive metadata

    Returns:
        Dictionary mapping tool names to lists of missing fields
    """
    required_fields = ["description", "example_uses"]
    recommended_fields = ["trigger_words", "example_non_uses"]

    issues = {}

    for tool_name, tool_def in TOOL_DEFINITIONS.items():
        missing = []

        # Check required fields
        for field in required_fields:
            if field not in tool_def:
                missing.append(f"REQUIRED: {field}")

        # Check recommended fields (except for special tools)
        if not tool_def.get("is_internal", False):
            for field in recommended_fields:
                if field not in tool_def:
                    missing.append(f"RECOMMENDED: {field}")

        if missing:
            issues[tool_name] = missing

    return issues


def get_all_registered_tools() -> List[str]:
    """
    Get list of all tools that should be registered in the system

    Returns:
        List of tool names from this single source of truth
    """
    return list(TOOL_DEFINITIONS.keys())
