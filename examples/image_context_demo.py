#!/usr/bin/env python3
"""
Image Generation with Conversation Context Demo

This script demonstrates how the enhanced image generation tool can use
conversation history to create more contextually relevant images.
"""

import logging
import os
import sys

# Add the docker/app directory to the path so we can import the tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../docker/app'))

from tools.image_gen import execute_image_generation_with_dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample conversation scenarios for testing
SAMPLE_CONVERSATIONS = {
    "fantasy_story": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "I'm working on a fantasy story about a young wizard named Elena who lives in a mystical forest.",
        },
        {
            "role": "assistant",
            "content": "That sounds like a wonderful story! Tell me more about Elena and her mystical forest.",
        },
        {
            "role": "user",
            "content": "Elena has long silver hair and wears a emerald green cloak. The forest is filled with ancient oak trees and glowing mushrooms that light up at night.",
        },
        {
            "role": "assistant",
            "content": "What an enchanting setting! The contrast between Elena's silver hair and emerald cloak against the backdrop of ancient oaks and bioluminescent mushrooms creates a vivid magical atmosphere.",
        },
        {"role": "user", "content": "Can you generate an image of Elena in her forest?"},
    ],
    "sci_fi_adventure": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "I'm brainstorming ideas for a sci-fi adventure. The main character is Captain Nova, who pilots a sleek silver spacecraft.",
        },
        {"role": "assistant", "content": "Interesting! What kind of adventures does Captain Nova go on?"},
        {
            "role": "user",
            "content": "She explores distant planets with purple skies and crystalline mountains. Her ship has blue energy engines that leave glowing trails.",
        },
        {
            "role": "assistant",
            "content": "That creates a vivid picture! The purple skies and crystalline mountains with Captain Nova's silver ship leaving blue energy trails sounds visually stunning.",
        },
        {"role": "user", "content": "Create an image of Captain Nova's ship approaching one of these alien planets."},
    ],
    "historical_character": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "I'm researching ancient Egyptian architecture for a project."},
        {
            "role": "assistant",
            "content": "Egyptian architecture is fascinating! Are you focusing on any particular period or type of structure?",
        },
        {
            "role": "user",
            "content": "I'm particularly interested in the golden age of pharaohs, with their elaborate temples featuring massive stone columns and hieroglyphic carvings.",
        },
        {
            "role": "assistant",
            "content": "The New Kingdom period produced some of the most magnificent temples, like Karnak and Luxor, with their towering columns and intricate hieroglyphs.",
        },
        {
            "role": "user",
            "content": "Generate an image of a majestic Egyptian temple with golden sunlight streaming through the columns.",
        },
    ],
}


def demonstrate_context_enhanced_image_generation():
    """Demonstrate how conversation context enhances image generation."""

    print("=" * 80)
    print("IMAGE GENERATION WITH CONVERSATION CONTEXT DEMONSTRATION")
    print("=" * 80)

    for scenario_name, messages in SAMPLE_CONVERSATIONS.items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name.replace('_', ' ').title()}")
        print(f"{'='*60}")

        # Show the conversation history
        print("\nConversation History:")
        print("-" * 30)
        for msg in messages:
            if msg["role"] != "system":
                role = msg["role"].title()
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                print(f"{role}: {content}")

        # Extract the user's image request (last user message)
        user_request = messages[-1]["content"]

        # Demonstrate image generation with and without context
        print(f"\n\nImage Generation Comparison:")
        print("-" * 40)

        # Generate with context
        print("\n1. WITH CONVERSATION CONTEXT:")
        try:
            # Extract a simple subject from the request
            if "Elena" in user_request:
                subject = "young wizard Elena in mystical forest"
            elif "Captain Nova" in user_request:
                subject = "silver spacecraft approaching alien planet"
            elif "temple" in user_request:
                subject = "ancient Egyptian temple"
            else:
                subject = "scene from conversation"

            context_params = {
                "user_prompt": user_request,
                "subject": subject,
                "style": "fantasy art" if "fantasy" in scenario_name else "digital art",
                "use_conversation_context": True,
                "messages": messages,
            }

            # Note: This would normally generate an actual image, but for demo purposes
            # we'll just show what the enhanced prompt would be
            print(f"   User Request: {user_request}")
            print(f"   Subject: {subject}")
            print(
                f"   Context Available: YES ({len([m for m in messages if m['role'] != 'system'])} conversation messages)"
            )
            print(f"   Enhanced Prompt: Would include visual elements from conversation context")

        except Exception as e:
            print(f"   Error with context: {e}")

        # Generate without context
        print("\n2. WITHOUT CONVERSATION CONTEXT:")
        try:
            no_context_params = {
                "user_prompt": user_request,
                "subject": subject,
                "style": "fantasy art" if "fantasy" in scenario_name else "digital art",
                "use_conversation_context": False,
                "messages": [],
            }

            print(f"   User Request: {user_request}")
            print(f"   Subject: {subject}")
            print(f"   Context Available: NO")
            print(f"   Enhanced Prompt: Basic enhancement without conversation context")

        except Exception as e:
            print(f"   Error without context: {e}")

        print("\n" + "=" * 60)


def demonstrate_visual_element_extraction():
    """Show how visual elements are extracted from conversation context."""

    print("\n" + "=" * 80)
    print("VISUAL ELEMENT EXTRACTION DEMONSTRATION")
    print("=" * 80)

    test_contexts = [
        {
            "name": "Fantasy Story Context",
            "context": "User is working on a fantasy story about Elena, a young wizard with long silver hair who wears an emerald green cloak. She lives in a mystical forest with ancient oak trees and glowing mushrooms that provide magical light at night.",
            "expected_elements": ["detailed character design", "fantasy elements", "atmospheric environment"],
        },
        {
            "name": "Sci-Fi Adventure Context",
            "context": "Captain Nova pilots a sleek silver spacecraft with blue energy engines. She explores alien planets with purple skies and crystalline mountains in dramatic space adventures.",
            "expected_elements": ["detailed character design", "dynamic action", "futuristic spacecraft"],
        },
        {
            "name": "Historical Context",
            "context": "Ancient Egyptian temples with massive stone columns, intricate hieroglyphic carvings, and golden sunlight streaming through the architectural elements create a majestic atmosphere.",
            "expected_elements": ["atmospheric environment", "ancient architecture", "golden lighting"],
        },
    ]

    print("\nThis demonstrates how the system would extract visual elements:")
    print("(Note: Actual extraction happens during real image generation)\n")

    for test in test_contexts:
        print(f"Context: {test['name']}")
        print(f"Description: {test['context']}")
        print(f"Expected Visual Elements: {', '.join(test['expected_elements'])}")
        print("-" * 40)


if __name__ == "__main__":
    print("IMAGE GENERATION CONVERSATION CONTEXT DEMO")
    print("This demo shows how conversation history enhances image generation.")
    print("\nNOTE: This is a demonstration script. For actual image generation,")
    print("you need to have the IMAGE_ENDPOINT environment variable configured.")
    print("\n" + "=" * 80)

    try:
        demonstrate_context_enhanced_image_generation()
        demonstrate_visual_element_extraction()

        print("\n" + "=" * 80)
        print("DEMO COMPLETED")
        print("=" * 80)
        print("\nKey Benefits of Conversation Context in Image Generation:")
        print("• More contextually relevant images based on ongoing discussions")
        print("• Better character consistency in story-based conversations")
        print("• Enhanced prompts with details from conversation history")
        print("• Improved artistic coherence across related image requests")
        print("• Automatic extraction of visual elements from dialogue")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)
