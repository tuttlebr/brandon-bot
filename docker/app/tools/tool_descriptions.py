"""
Tool Descriptions and Selection Guidelines

This module provides enhanced tool descriptions and decision logic to help
the primary tool-calling LLM make better tool selection choices.
"""

from typing import Dict, List, Optional


class ToolDescriptionEnhancer:
    """Enhances tool descriptions with context-aware guidance"""

    # Enhanced descriptions that emphasize when NOT to use each tool
    ENHANCED_DESCRIPTIONS = {
        "generate_image": {
            "description": "Creates AI-generated images based on text descriptions. ONLY use when user explicitly asks to 'create', 'generate', 'make', or 'draw' an image. DO NOT use for: acknowledgments ('thanks', 'great'), follow-up comments, questions about existing images, or any non-creation requests.",
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
            "description": "Processes text and documents through various operations. ONLY use when user explicitly requests text processing: analyze, summarize, proofread, rewrite, translate, critique, or code development. DO NOT use for: general questions, acknowledgments, or when no text processing is explicitly requested.",
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
                "Summarize this article",
                "Proofread my essay",
                "Translate this to Spanish",
            ],
            "example_non_uses": [
                "What is machine learning? (general question - use generalist)",
                "Thanks for the help (acknowledgment - no tool needed)",
            ],
        },
        "analyze_image": {
            "description": "Analyzes uploaded images to describe content, identify objects, or answer visual questions. ONLY use when user asks about an image they've uploaded. DO NOT use for: generating images, text analysis, or general questions.",
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
                "What's in this image?",
                "Describe what you see",
                "Is there a cat in this picture?",
            ],
            "example_non_uses": [
                "Generate an image (use generate_image)",
                "Thanks for analyzing (acknowledgment - no tool needed)",
            ],
        },
        "tavily_internet_search": {
            "description": "Searches the internet for current information, facts, and real-time data. ONLY use when user needs up-to-date information, current events, or facts that require web search. DO NOT use for: general knowledge, acknowledgments, or when information can be answered without search.",
            "trigger_words": [
                "search",
                "find",
                "look up",
                "current",
                "latest",
                "recent",
                "today",
                "news",
            ],
            "anti_trigger_words": ["thanks", "what is", "explain", "tell me about"],
            "requires_specific_query": True,
            "example_uses": [
                "What's the current weather in NYC?",
                "Search for latest AI news",
                "Find information about recent events",
            ],
            "example_non_uses": [
                "What is Python? (general knowledge - use generalist)",
                "Thanks for the search results (acknowledgment - no tool needed)",
            ],
        },
        "generalist_conversation": {
            "description": "Engages in general conversation, explanations, and discussions without needing external data. Use for: philosophical discussions, concept explanations, general advice, creative writing, or casual chat. This is the DEFAULT tool when no specific action is requested.",
            "is_default": True,
            "example_uses": [
                "What is machine learning?",
                "Tell me about philosophy",
                "Can you explain quantum physics?",
                "Thanks for your help!",
                "That's interesting!",
            ],
        },
        "get_weather": {
            "description": "Retrieves current weather data for specific locations. ONLY use when user explicitly asks for weather information. DO NOT use for: general climate questions, acknowledgments, or non-weather queries.",
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
                "What causes rain? (general question - use generalist)",
            ],
        },
        "retrieval_search": {
            "description": "Searches specialized knowledge base for NVIDIA products, technologies, and mental health topics. ONLY use for queries specifically about these topics. DO NOT use for: general questions, acknowledgments, or topics outside the knowledge base.",
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
                "What is a computer? (too general - use generalist)",
                "Thanks! (acknowledgment - no tool needed)",
            ],
        },
        "conversation_context": {
            "description": "INTERNAL TOOL: Analyzes conversation history. NEVER use for direct user queries. Only for system-level conversation analysis when explicitly needed for continuity.",
            "is_internal": True,
            "never_use_for_user_queries": True,
        },
        "extract_web_content": {
            "description": "Extracts and reads content from specific URLs provided by user. ONLY use when user provides a URL and asks to read/extract/analyze it. DO NOT use for: general web searches, acknowledgments, or when no URL is provided.",
            "trigger_words": ["read", "extract", "analyze", "check"],
            "requires_url": True,
            "example_uses": [
                "Read this article: https://example.com",
                "Extract content from [URL]",
                "What does this webpage say?",
            ],
            "example_non_uses": [
                "Search for information (use tavily_internet_search)",
                "Thanks! (acknowledgment - no tool needed)",
            ],
        },
        "retrieve_pdf_summary": {
            "description": "Retrieves summaries of uploaded PDF documents. ONLY use when user has uploaded a PDF and asks for a summary. DO NOT use for: general text, acknowledgments, or non-PDF content.",
            "context_requirement": "uploaded_pdf",
            "trigger_words": ["summary", "summarize", "overview"],
            "example_uses": [
                "Summarize this PDF",
                "Give me an overview of the document",
                "What's the main point of this PDF?",
            ],
        },
        "process_pdf_text": {
            "description": "Processes specific pages or sections of uploaded PDFs. ONLY use for detailed PDF text operations on specific pages. DO NOT use for: general summaries, acknowledgments, or non-PDF content.",
            "context_requirement": "uploaded_pdf",
            "trigger_words": ["page", "section", "extract", "process"],
            "requires_page_specification": True,
            "example_uses": [
                "Extract text from page 5",
                "Analyze pages 10-15",
                "Process the methodology section",
            ],
        },
        "tavily_news_search": {
            "description": "Searches specifically for news articles and current events. ONLY use when user explicitly asks for news or recent events. DO NOT use for: general web search, acknowledgments, or non-news queries.",
            "trigger_words": ["news", "headlines", "breaking", "events", "happening"],
            "is_specialized_search": True,
            "example_uses": [
                "What's in the news today?",
                "Latest headlines about AI",
                "Breaking news from tech industry",
            ],
        },
    }

    @classmethod
    def get_decision_prompt(cls) -> str:
        """Get the decision-making prompt for tool selection"""
        return """## Tool Selection Guidelines

CRITICAL: Analyze the user's message carefully before selecting any tool.

1. **Acknowledgments and Comments**: If the user is saying thanks, acknowledging, or making a comment about previous output, DO NOT use any tool. Just respond conversationally.

2. **Action vs. Non-Action**: Determine if the user is requesting an ACTION or just making a statement/asking a general question.
   - ACTION requests contain verbs like: create, generate, analyze, search, extract, etc.
   - NON-ACTION includes: thanks, comments, general questions, acknowledgments

3. **Context Awareness**: Consider what just happened in the conversation.
   - If you just generated an image and user says "thanks", they're acknowledging, not requesting another image
   - If you just searched and user says "great", they're commenting, not requesting another search

4. **Default to No Tool**: When in doubt, especially for acknowledgments or general conversation, use NO tool and respond directly.

5. **Specific Tool Requirements**:
   - Image Generation: ONLY when explicitly asked to create/generate/make an image
   - Text Processing: ONLY when given text to process with specific operation
   - Search Tools: ONLY when user needs current/external information
   - Weather: ONLY when asking for weather in a specific location
   - PDF Tools: ONLY when user has uploaded a PDF and asks about it

Remember: Most "thanks", "great", "perfect", "nice" responses need NO tools - just acknowledge politely."""

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
