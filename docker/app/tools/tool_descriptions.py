"""
Tool Descriptions and Selection Guidelines

This module provides the SINGLE SOURCE OF TRUTH for all tool descriptions,
metadata, and decision logic to help the primary tool-calling LLM make better
tool selection choices.
"""

from typing import Any, Dict, List, Optional

# SINGLE SOURCE OF TRUTH for all tool descriptions and metadata
TOOL_DEFINITIONS = {
    "generate_image": {
        "description": "Generate AI images from text descriptions. Use when user requests creating, generating, making, or drawing images.",
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
        "description": "Process text with specific operations: summarize, translate, proofread, rewrite, analyze documents, or develop code. Use when user provides text AND requests processing.",
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
        "description": "Analyze uploaded images to describe content or answer visual questions. Use when user uploads an image AND asks about it.",
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
    "tavily_internet_search": {
        "description": "Search the internet for current information and real-time data. Use for up-to-date facts, current events, or information that changes frequently.",
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
        "anti_trigger_words": ["thanks", "what is", "explain", "tell me about"],
        "requires_specific_query": True,
        "example_uses": [
            "What's happening in tech today?",
            "Search for latest AI developments",
            "Find current stock prices",
        ],
        "example_non_uses": [
            "What is Python? (general knowledge - no tool needed)",
            "How does gravity work? (general knowledge - no tool needed)",
            "Thanks for the search results (acknowledgment - no tool needed)",
        ],
    },
    "generalist_conversation": {
        "description": "DEFAULT: Handle general conversation without external tools. Use for explanations, discussions, advice, creative writing, and casual chat.",
        "is_default": True,
        "example_uses": [
            "What is machine learning?",
            "Tell me about philosophy",
            "Can you explain quantum physics?",
            "Thanks for your help!",
            "That's interesting!",
            "How are you?",
        ],
    },
    "get_weather": {
        "description": "Get current weather for a specific location. Use when user asks for weather, temperature, or forecast AND provides a city/location.",
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
        "description": "Search specialized knowledge base for mental health resources or NVIDIA technical documentation. Use ONLY for these specific domains.",
        "trigger_words": [
            "nvidia",
            "gpu",
            "cuda",
            "mental health",
            "therapy",
            "depression",
            "anxiety",
        ],
        "specialized_domains": ["nvidia", "mental_health"],
        "example_uses": [
            "Tell me about NVIDIA GPUs",
            "What is CUDA programming?",
            "Information about depression treatment",
        ],
        "example_non_uses": [
            "What is a computer? (general topic - no tool needed)",
            "How does AI work? (general topic - no tool needed)",
            "Thanks! (acknowledgment - no tool needed)",
        ],
    },
    "conversation_context": {
        "description": "INTERNAL SYSTEM TOOL: Analyze conversation history for context. Never select for user queries.",
        "is_internal": True,
        "never_use_for_user_queries": True,
    },
    "extract_web_content": {
        "description": "Extract and read content from a specific URL. Use when user provides a URL AND asks to read or analyze it.",
        "trigger_words": ["read", "extract", "analyze", "check", "look at"],
        "requires_url": True,
        "example_uses": [
            "Read this article: https://example.com",
            "Extract content from https://example.com",
            "What does this webpage say: [URL]",
        ],
        "example_non_uses": [
            "Search for information about X (use tavily_internet_search)",
            "Find me articles about Y (use tavily_internet_search)",
            "Thanks! (acknowledgment - no tool needed)",
        ],
    },
    "retrieve_pdf_summary": {
        "description": "Get quick summary of uploaded PDF documents. Use when user uploads a PDF AND wants an overview or summary.",
        "context_requirement": "uploaded_pdf",
        "trigger_words": ["summary", "summarize", "overview", "brief"],
        "example_uses": [
            "Summarize this PDF (with uploaded PDF)",
            "Give me an overview of the document (with uploaded PDF)",
            "What's the main point of this PDF? (with uploaded PDF)",
        ],
    },
    "process_pdf_text": {
        "description": "Process specific pages or answer questions about uploaded PDFs. Use for detailed PDF analysis, specific page extraction, or Q&A about PDF content.",
        "context_requirement": "uploaded_pdf",
        "trigger_words": [
            "page",
            "section",
            "extract",
            "process",
            "question",
            "analyze",
        ],
        "requires_page_specification": False,  # Changed to allow general Q&A
        "example_uses": [
            "Extract text from page 5",
            "What does the PDF say about methodology?",
            "Analyze the conclusions section",
        ],
    },
    "tavily_news_search": {
        "description": "Search specifically for news articles and breaking events. Use when user explicitly asks for news, headlines, or current events.",
        "trigger_words": ["news", "headlines", "breaking", "events", "happening"],
        "is_specialized_search": True,
        "example_uses": [
            "What's in the news today?",
            "Latest headlines about AI",
            "Breaking news from tech industry",
        ],
    },
}


class ToolDescriptionEnhancer:
    """Enhances tool descriptions with context-aware guidance"""

    # Use the single source of truth
    ENHANCED_DESCRIPTIONS = TOOL_DEFINITIONS

    @classmethod
    def get_decision_prompt(cls) -> str:
        """Get the decision-making prompt for tool selection"""
        return """## Tool Selection Guidelines

IMPORTANT: Tool selection is OPTIONAL. Most user messages do NOT require any tool. Default to NO TOOL selection unless clearly needed.

### Decision Process:

1. **No Tool Needed (Most Common)**:
   - Acknowledgments: "thanks", "great", "perfect", "nice", etc.
   - General questions that can be answered from knowledge
   - Casual conversation and chat
   - Comments about previous responses
   - Explanations of concepts
   - Any message where you can provide a helpful response without external tools

2. **Tool Selection Criteria**:
   - User makes an EXPLICIT request matching tool functionality
   - Request contains BOTH action verb AND target (e.g., "generate" + "image")
   - Required context is present (e.g., uploaded file for file analysis)
   - Information genuinely requires external data (e.g., current weather, today's news)

3. **Red Flags Against Tool Use**:
   - Message is primarily acknowledgment or comment
   - General knowledge question
   - No clear action requested
   - Missing required context (no URL for extract_web_content, no image for analyze_image)
   - Can be answered without external data

4. **Examples of NO TOOL Needed**:
   - "Thanks!" → Just acknowledge
   - "What is machine learning?" → Explain from knowledge
   - "That's perfect" → Acknowledge their satisfaction
   - "How does X work?" → Explain the concept
   - "Tell me about Y" → Share knowledge

5. **Examples Requiring Tools**:
   - "Generate an image of a cat" → generate_image
   - "What's the weather in NYC?" → get_weather
   - "Search for latest AI news" → tavily_internet_search
   - "Analyze this image [uploaded]" → analyze_image

Remember: When in doubt, NO TOOL is usually correct. Only select a tool when the user's request clearly matches its specific purpose AND cannot be fulfilled without it."""

    @classmethod
    def should_use_tool(
        cls,
        user_message: str,
        tool_name: str,
        conversation_context: Optional[Dict] = None,
    ) -> bool:
        """
        Determine if a tool should be used based on the user message and context

        Args:
            user_message: The user's message
            tool_name: The tool being considered
            conversation_context: Optional context about recent conversation

        Returns:
            True if tool should be used, False otherwise
        """
        if tool_name not in cls.ENHANCED_DESCRIPTIONS:
            return True  # Unknown tool, let system decide

        tool_info = cls.ENHANCED_DESCRIPTIONS[tool_name]
        message_lower = user_message.lower()

        # Check for anti-trigger words (strong signal NOT to use the tool)
        if "anti_trigger_words" in tool_info:
            for word in tool_info["anti_trigger_words"]:
                if word in message_lower:
                    # Check if it's the primary intent (not just part of a larger request)
                    if (
                        len(user_message.split()) < 5
                    ):  # Short message, likely just acknowledgment
                        return False

        # Check for required context
        if "context_requirement" in tool_info:
            if (
                not conversation_context
                or tool_info["context_requirement"] not in conversation_context
            ):
                return False

        # Check for required elements
        if tool_info.get("requires_action_verb", False):
            has_action_verb = any(
                word in message_lower for word in tool_info.get("trigger_words", [])
            )
            if not has_action_verb:
                return False

        if tool_info.get("requires_location", False):
            # Simple check for location indicators
            location_indicators = [
                "in",
                "at",
                "near",
                "weather",
                "forecast",
                "temperature",
            ]
            has_location = any(
                indicator in message_lower for indicator in location_indicators
            )
            if not has_location:
                return False

        if tool_info.get("requires_url", False):
            # Check for URL patterns
            url_patterns = ["http://", "https://", "www.", ".com", ".org", ".net"]
            has_url = any(pattern in user_message for pattern in url_patterns)
            if not has_url:
                return False

        # Internal tools should never be used for user queries
        if tool_info.get("is_internal", False):
            return False

        return True

    @classmethod
    def get_enhanced_tool_description(cls, tool_name: str) -> str:
        """Get the enhanced description for a tool"""
        if tool_name in cls.ENHANCED_DESCRIPTIONS:
            return cls.ENHANCED_DESCRIPTIONS[tool_name]["description"]
        return None

    @classmethod
    def get_tool_examples(cls, tool_name: str) -> Dict[str, List[str]]:
        """Get examples of when to use and not use a tool"""
        if tool_name in cls.ENHANCED_DESCRIPTIONS:
            tool_info = cls.ENHANCED_DESCRIPTIONS[tool_name]
            return {
                "use": tool_info.get("example_uses", []),
                "dont_use": tool_info.get("example_non_uses", []),
            }
        return {"use": [], "dont_use": []}


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


def get_tool_metadata(tool_name: str) -> Dict[str, Any]:
    """
    Get all metadata for a specific tool from the single source of truth.

    Args:
        tool_name: The name of the tool

    Returns:
        The tool metadata dictionary, or empty dict if not found
    """
    return TOOL_DEFINITIONS.get(tool_name, {})


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
    separators = ["now", "next", "also", "and", "but", "can you", "could you", "please"]

    for separator in separators:
        if separator in message_lower:
            parts = message.split(separator, 1)
            if len(parts) > 1 and len(parts[1].strip()) > 5:
                return parts[1].strip()

    return None
