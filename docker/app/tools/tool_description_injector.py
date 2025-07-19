"""
Tool Description Injector

This module provides a way to inject enhanced descriptions into tools
at runtime without modifying the tool source files.
"""

import logging

logger = logging.getLogger(__name__)

# Enhanced tool descriptions focused on preventing repetitive calls
ENHANCED_TOOL_DESCRIPTIONS = {
    "generate_image": "Creates AI-generated images based on text descriptions. ONLY use when user explicitly asks to 'create', 'generate', 'make', or 'draw' an image. DO NOT use for: acknowledgments ('thanks', 'great'), follow-up comments, questions about existing images, or any non-creation requests. Examples of proper use: 'Generate an image of a sunset', 'Create a picture of a cat'. Examples of improper use: 'Thanks!' (acknowledgment), 'That looks great' (comment).",
    "text_assistant": "Processes text and documents through various operations. ONLY use when user explicitly requests text processing: analyze, summarize, proofread, rewrite, translate, critique, or code development. DO NOT use for: general questions, acknowledgments ('thanks', 'great'), or when no text processing is explicitly requested.",
    "analyze_image": "Analyzes uploaded images to describe content, identify objects, or answer visual questions. ONLY use when user asks about an image they've uploaded. DO NOT use for: generating images, text analysis, acknowledgments, or general questions.",
    "tavily_internet_search": "Searches the internet for current information, facts, and real-time data. ONLY use when user needs up-to-date information, current events, or facts that require web search. DO NOT use for: general knowledge questions, acknowledgments ('thanks'), or when information can be answered without search.",
    "generalist_conversation": "Engages in general conversation, explanations, and discussions without needing external data. Use for: philosophical discussions, concept explanations, general advice, creative writing, casual chat, AND acknowledgments like 'thanks', 'great', 'perfect'. This is the DEFAULT tool when no specific action is requested.",
    "get_weather": "Retrieves current weather data for specific locations. ONLY use when user explicitly asks for weather information with a location. DO NOT use for: general climate questions, acknowledgments, or non-weather queries.",
    "retrieval_search": "Searches specialized knowledge base for NVIDIA products, technologies, and mental health topics. ONLY use for queries specifically about these topics. DO NOT use for: general questions, acknowledgments, or topics outside the knowledge base.",
    "conversation_context": "INTERNAL TOOL: Analyzes conversation history. NEVER use for direct user queries. Only for system-level conversation analysis.",
    "extract_web_content": "Extracts and reads content from specific URLs provided by user. ONLY use when user provides a URL and asks to read/extract/analyze it. DO NOT use for: general web searches, acknowledgments, or when no URL is provided.",
    "retrieve_pdf_summary": "Retrieves summaries of uploaded PDF documents. ONLY use when user has uploaded a PDF and asks for a summary. DO NOT use for: general text, acknowledgments, or non-PDF content.",
    "process_pdf_text": "Processes specific pages or sections of uploaded PDFs. ONLY use for detailed PDF text operations on specific pages. DO NOT use for: general summaries, acknowledgments, or non-PDF content.",
    "tavily_news_search": "Searches specifically for news articles and current events. ONLY use when user explicitly asks for news or recent events. DO NOT use for: general web search, acknowledgments, or non-news queries.",
}


def inject_enhanced_descriptions(tool_definitions: list) -> list:
    """
    Inject enhanced descriptions into tool definitions

    Args:
        tool_definitions: List of tool definitions

    Returns:
        Modified tool definitions with enhanced descriptions
    """
    for tool_def in tool_definitions:
        if "function" in tool_def and "name" in tool_def["function"]:
            tool_name = tool_def["function"]["name"]

            if tool_name in ENHANCED_TOOL_DESCRIPTIONS:
                # Update the description
                tool_def["function"]["description"] = ENHANCED_TOOL_DESCRIPTIONS[
                    tool_name
                ]
                logger.debug(f"Injected enhanced description for tool: {tool_name}")

    return tool_definitions


def get_enhanced_description(tool_name: str) -> str:
    """
    Get enhanced description for a specific tool

    Args:
        tool_name: Name of the tool

    Returns:
        Enhanced description or empty string if not found
    """
    return ENHANCED_TOOL_DESCRIPTIONS.get(tool_name, "")
