#!/usr/bin/env python3
"""
Demonstration of the Conversation Context Tool

This script shows how the conversation context tool can analyze conversation history
and provide context summaries for other tools to use.
"""

import json
from typing import Any, Dict, List

# Mock conversation history for demonstration
SAMPLE_CONVERSATIONS = {
    "weather_context": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hi! I'm planning a trip to San Francisco next week."},
        {
            "role": "assistant",
            "content": "That's exciting! San Francisco is a beautiful city. What kind of activities are you planning?",
        },
        {
            "role": "user",
            "content": "I'm thinking of visiting Golden Gate Park and maybe some museums. I'm traveling with my kids.",
        },
        {
            "role": "assistant",
            "content": "Great choices! The Exploratorium and California Academy of Sciences are excellent for families. Golden Gate Park has playgrounds and the Japanese Tea Garden too.",
        },
        {"role": "user", "content": "Perfect! What's the weather going to be like?"},
    ],
    "task_continuity": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "I need help writing a Python script to analyze sales data."},
        {
            "role": "assistant",
            "content": "I'd be happy to help! What kind of analysis do you need? Are you looking to calculate totals, averages, trends, or something else?",
        },
        {"role": "user", "content": "I want to find the top 10 selling products by month and create a visualization."},
        {
            "role": "assistant",
            "content": "Here's a Python script using pandas and matplotlib:\n\n```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n# Load your data\ndf = pd.read_csv('sales_data.csv')\n\n# Group by month and product\nmonthly_sales = df.groupby(['month', 'product'])['quantity'].sum().reset_index()\n\n# Get top 10 for each month\ntop_products = monthly_sales.groupby('month').apply(\n    lambda x: x.nlargest(10, 'quantity')\n).reset_index(drop=True)\n\n# Create visualization\nfig, ax = plt.subplots(figsize=(12, 8))\n# ... plotting code ...\n```\n\nWould you like me to explain any part of this code?",
        },
        {"role": "user", "content": "Can you help me add error handling and make it more robust?"},
    ],
    "user_preferences": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you explain machine learning in simple terms?"},
        {
            "role": "assistant",
            "content": "Sure! Machine learning is like teaching a computer to recognize patterns...",
        },
        {
            "role": "user",
            "content": "That's helpful, but can you give me a more technical explanation? I have a computer science background.",
        },
        {
            "role": "assistant",
            "content": "Absolutely! Machine learning involves algorithms that learn from data to make predictions...",
        },
        {
            "role": "user",
            "content": "Perfect! I prefer detailed technical explanations. Can you also explain neural networks?",
        },
        {
            "role": "assistant",
            "content": "Neural networks are computational models inspired by biological neural networks...",
        },
        {
            "role": "user",
            "content": "Great! Now can you summarize the key differences between supervised and unsupervised learning?",
        },
    ],
}


def simulate_conversation_context_tool(
    messages: List[Dict[str, Any]], context_type: str, message_count: int = 6, focus_query: str = None
) -> Dict[str, Any]:
    """
    Simulate the conversation context tool for demonstration purposes.
    In real usage, this would call the actual LLM for analysis.
    """

    # Filter to recent messages (excluding system)
    recent_messages = []
    for msg in reversed(messages):
        if msg.get("role") != "system":
            recent_messages.append(msg)
            if len(recent_messages) >= message_count:
                break
    recent_messages.reverse()

    # Simulate different context types
    if context_type == "conversation_summary":
        if "San Francisco" in str(messages):
            return {
                "context_type": "conversation_summary",
                "summary": "User is planning a family trip to San Francisco next week. They want to visit Golden Gate Park and museums with their kids. They've expressed interest in family-friendly activities and are now asking about weather conditions for their trip.",
                "relevant_messages_count": len(recent_messages),
                "key_topics": ["San Francisco", "family trip", "Golden Gate Park", "museums", "travel planning"],
                "user_intent": "information_search",
            }
        elif "Python" in str(messages):
            return {
                "context_type": "conversation_summary",
                "summary": "User is working on a Python script for sales data analysis. They want to identify top 10 selling products by month and create visualizations. They've received initial code and are now asking for improvements like error handling and robustness.",
                "relevant_messages_count": len(recent_messages),
                "key_topics": ["Python", "sales data analysis", "data visualization", "code improvement"],
                "user_intent": "problem_solving",
            }
        elif "machine learning" in str(messages):
            return {
                "context_type": "conversation_summary",
                "summary": "User has been asking about machine learning concepts. They indicated they have a computer science background and prefer detailed technical explanations. They've asked about ML basics, neural networks, and now want to understand supervised vs unsupervised learning.",
                "relevant_messages_count": len(recent_messages),
                "key_topics": ["machine learning", "neural networks", "technical explanations", "supervised learning"],
                "user_intent": "seeking_information",
            }

    elif context_type == "recent_topics":
        topics = []
        content = " ".join([msg.get("content", "") for msg in recent_messages])
        if "San Francisco" in content:
            topics = ["San Francisco travel", "family activities", "Golden Gate Park", "museums", "weather planning"]
        elif "Python" in content:
            topics = ["Python programming", "data analysis", "sales data", "visualization", "code robustness"]
        elif "machine learning" in content:
            topics = [
                "machine learning",
                "neural networks",
                "supervised learning",
                "unsupervised learning",
                "technical concepts",
            ]

        return {
            "context_type": "recent_topics",
            "summary": f"Main topics discussed: {', '.join(topics)}",
            "relevant_messages_count": len(recent_messages),
            "key_topics": topics,
            "user_intent": "seeking_information",
        }

    elif context_type == "user_preferences":
        content = " ".join([msg.get("content", "") for msg in recent_messages])
        if "technical" in content.lower() or "computer science" in content.lower():
            return {
                "context_type": "user_preferences",
                "summary": "User has indicated they have a computer science background and prefer detailed technical explanations over simplified ones. They appreciate comprehensive, in-depth responses with technical accuracy.",
                "relevant_messages_count": len(recent_messages),
                "key_topics": ["technical communication", "detailed explanations", "computer science background"],
                "user_intent": "seeking_information",
            }
        else:
            return {
                "context_type": "user_preferences",
                "summary": "User communication style appears to be conversational and friendly. They ask follow-up questions and seem to prefer step-by-step explanations.",
                "relevant_messages_count": len(recent_messages),
                "key_topics": ["conversational style", "follow-up questions", "step-by-step learning"],
                "user_intent": "seeking_information",
            }

    elif context_type == "task_continuity":
        content = " ".join([msg.get("content", "") for msg in recent_messages])
        if "Python" in content and "sales data" in content:
            return {
                "context_type": "task_continuity",
                "summary": "User is working on a Python data analysis project. They've received initial code for analyzing sales data and creating visualizations. Current task is to improve the code with error handling and make it more robust. Next steps would be implementing try-catch blocks, input validation, and handling edge cases.",
                "relevant_messages_count": len(recent_messages),
                "key_topics": ["Python development", "data analysis", "code improvement", "error handling"],
                "user_intent": "problem_solving",
            }
        else:
            return {
                "context_type": "task_continuity",
                "summary": "User appears to be in an information-gathering phase, asking related questions to build understanding of a topic progressively.",
                "relevant_messages_count": len(recent_messages),
                "key_topics": ["information gathering", "progressive learning"],
                "user_intent": "seeking_information",
            }

    # Default response
    return {
        "context_type": context_type,
        "summary": "Context analysis completed",
        "relevant_messages_count": len(recent_messages),
        "key_topics": ["general conversation"],
        "user_intent": "seeking_information",
    }


def demonstrate_context_tool():
    """Demonstrate how the conversation context tool works with different scenarios."""

    print("=" * 80)
    print("CONVERSATION CONTEXT TOOL DEMONSTRATION")
    print("=" * 80)

    for scenario_name, messages in SAMPLE_CONVERSATIONS.items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name.replace('_', ' ').title()}")
        print(f"{'='*60}")

        # Show the conversation history
        print("\nConversation History:")
        print("-" * 30)
        for i, msg in enumerate(messages):
            if msg["role"] != "system":
                role = msg["role"].title()
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"{role}: {content}")

        print("\nContext Analysis Results:")
        print("-" * 30)

        # Demonstrate different context types
        context_types = ["conversation_summary", "recent_topics", "user_preferences", "task_continuity"]

        for context_type in context_types:
            print(f"\n{context_type.replace('_', ' ').title()}:")
            result = simulate_conversation_context_tool(messages, context_type)

            print(f"  Summary: {result['summary']}")
            print(f"  Key Topics: {', '.join(result['key_topics'])}")
            print(f"  User Intent: {result['user_intent']}")
            print(f"  Messages Analyzed: {result['relevant_messages_count']}")


def demonstrate_tool_integration():
    """Show how the context tool can be used with other tools."""

    print("\n" + "=" * 80)
    print("TOOL INTEGRATION EXAMPLES")
    print("=" * 80)

    # Example 1: Weather tool with location context
    print("\nExample 1: Weather Tool with Location Context")
    print("-" * 50)

    weather_messages = SAMPLE_CONVERSATIONS["weather_context"]
    context_result = simulate_conversation_context_tool(weather_messages, "conversation_summary")

    print("1. User asks: 'What's the weather going to be like?'")
    print("2. LLM calls conversation_context tool to understand context")
    print(f"3. Context tool returns: {context_result['summary'][:100]}...")
    print("4. LLM extracts location 'San Francisco' from context")
    print("5. LLM calls weather tool with location='San Francisco'")
    print("6. Weather tool returns forecast for San Francisco")

    # Example 2: Code assistance with task continuity
    print("\nExample 2: Code Assistance with Task Continuity")
    print("-" * 50)

    code_messages = SAMPLE_CONVERSATIONS["task_continuity"]
    context_result = simulate_conversation_context_tool(code_messages, "task_continuity")

    print("1. User asks: 'Can you help me add error handling and make it more robust?'")
    print("2. LLM calls conversation_context tool to understand current task")
    print(f"3. Context tool identifies: {context_result['summary'][:100]}...")
    print("4. LLM provides specific error handling improvements for the sales analysis script")
    print("5. Response is tailored to the ongoing Python data analysis project")


def show_configuration_examples():
    """Show different configuration options for the context tool."""

    print("\n" + "=" * 80)
    print("CONFIGURATION EXAMPLES")
    print("=" * 80)

    messages = SAMPLE_CONVERSATIONS["user_preferences"]

    print("\nDifferent Message Count Settings:")
    print("-" * 40)

    for count in [3, 6, 10]:
        result = simulate_conversation_context_tool(messages, "conversation_summary", message_count=count)
        print(f"Message count = {count}: Analyzed {result['relevant_messages_count']} messages")

    print("\nFocus Query Example:")
    print("-" * 25)

    # Simulate focus query (in real implementation, this would affect the analysis)
    result = simulate_conversation_context_tool(messages, "recent_topics", focus_query="learning preferences")
    print(f"Focus query 'learning preferences': {result['summary']}")


if __name__ == "__main__":
    demonstrate_context_tool()
    demonstrate_tool_integration()
    show_configuration_examples()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        """
The Conversation Context Tool provides:

1. **Efficient Context Analysis**: Analyzes conversation history to extract relevant context
2. **Multiple Context Types**: Different analysis modes for different use cases
3. **Automatic Integration**: Works seamlessly with existing tools
4. **Configurable Parameters**: Adjust message count and focus areas
5. **Performance Optimized**: Generates summaries instead of passing full history

Key Benefits:
- Enables context-aware tool calls
- Reduces token usage compared to full history passing
- Improves user experience with more relevant responses
- Supports complex multi-turn conversations
- Maintains conversation continuity across tool calls

To use in your application:
1. The tool is automatically available to the LLM
2. LLM decides when context would be helpful
3. Conversation messages are automatically injected
4. Context results can inform subsequent tool calls
5. All happens transparently to the user
    """
    )
