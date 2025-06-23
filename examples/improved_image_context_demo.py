#!/usr/bin/env python3
"""
Improved Image Generation with Conversation Context Demo

This script demonstrates the enhanced image generation tool that now extracts
specific visual details directly from conversation history rather than using
only high-level summaries.
"""

import re
from typing import Any, Dict, List, Optional


def simulate_raw_visual_extraction(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Simulate the _extract_raw_visual_details method to show how it works
    """
    visual_details = []

    # Look at recent messages (excluding system messages)
    recent_messages = [msg for msg in messages[-10:] if msg.get("role") != "system"]

    for msg in recent_messages:
        content = str(msg.get("content", "")).lower()

        # Extract colors
        colors = re.findall(
            r'\b(?:red|blue|green|yellow|purple|orange|pink|black|white|grey|gray|brown|golden|silver|emerald|crimson|azure|violet|amber|scarlet|turquoise|magenta|cyan|indigo|navy|maroon|teal|olive|lime|coral|salmon|tan|beige|cream|ivory|pearl|platinum|bronze|copper|rose|lavender|mint|jade|ruby|sapphire|diamond|crystal)\b',
            content,
        )
        for color in colors:
            if f"{color} " in content:
                # Try to find what the color describes
                color_match = re.search(rf'{color}\s+(\w+)', content)
                if color_match:
                    visual_details.append(f"{color} {color_match.group(1)}")
                else:
                    visual_details.append(color)

        # Extract physical descriptions
        physical_descriptors = re.findall(
            r'\b(?:long|short|tall|small|large|big|tiny|massive|huge|slender|thick|thin|broad|narrow|wide|deep|shallow|bright|dark|dim|glowing|shining|sparkling|gleaming|radiant|luminous|shadowy|misty|foggy|clear|transparent|opaque|smooth|rough|soft|hard|sharp|dull|curved|straight|twisted|bent|round|square|oval|triangular|angular|flowing|cascading|towering|majestic|elegant|rustic|ancient|modern|futuristic|mysterious|enchanted|magical|mystical|ethereal|otherworldly)\s+(?:hair|eyes|skin|face|hands|arms|legs|body|figure|silhouette|cloak|robe|dress|shirt|pants|boots|shoes|hat|crown|jewelry|necklace|ring|bracelet|earrings|sword|staff|wand|wings|tail|horns|scales|feathers|fur|mane|beard|mustache|eyebrows|lips|nose|ears|neck|shoulders|chest|back|waist|hips|thighs|calves|ankles|feet|toes|fingers|nails|building|house|castle|tower|bridge|tree|forest|mountain|hill|valley|river|lake|ocean|sky|clouds|sun|moon|stars|planet|ship|spacecraft|vehicle|machine|robot|creature|dragon|unicorn|phoenix|griffin|pegasus|centaur|mermaid|fairy|elf|dwarf|giant|wizard|witch|knight|warrior|princess|prince|king|queen)\b',
            content,
            re.IGNORECASE,
        )
        visual_details.extend([desc.lower() for desc in physical_descriptors])

        # Extract specific clothing and accessories
        clothing_items = re.findall(
            r'\b(?:wearing|wears|dressed in|adorned with|carrying|holding|wielding)\s+(?:a|an|the)?\s*([^.,!?;]*?)(?:[.,!?;]|$)',
            content,
            re.IGNORECASE,
        )
        for item in clothing_items:
            if item.strip():
                visual_details.append(item.strip().lower())

        # Extract lighting and atmosphere
        lighting_atmosphere = re.findall(
            r'\b(?:glowing|shining|sparkling|gleaming|radiant|luminous|bright|dim|dark|shadowy|misty|foggy|clear|sunny|cloudy|stormy|peaceful|serene|dramatic|intense|mysterious|magical|mystical|enchanted|ethereal|otherworldly|ancient|modern|futuristic)\b',
            content,
            re.IGNORECASE,
        )
        visual_details.extend([desc.lower() for desc in lighting_atmosphere])

    # Remove duplicates while preserving order
    unique_details = []
    seen = set()
    for detail in visual_details:
        if detail not in seen and len(detail) > 2:  # Skip very short matches
            unique_details.append(detail)
            seen.add(detail)

    return unique_details[:15]  # Limit to most important details


def simulate_enhanced_prompt_generation(
    user_prompt: str, subject: str, raw_details: List[str], style: str = "digital art"
) -> str:
    """
    Simulate how the enhanced prompt would be generated with raw visual details
    """
    enhanced_parts = [subject]

    # Add prioritized raw details
    if raw_details:
        # High priority: color + object combinations
        color_items = [
            detail
            for detail in raw_details
            if any(
                color in detail
                for color in [
                    'red',
                    'blue',
                    'green',
                    'yellow',
                    'purple',
                    'orange',
                    'pink',
                    'black',
                    'white',
                    'grey',
                    'gray',
                    'brown',
                    'golden',
                    'silver',
                    'emerald',
                    'crimson',
                    'azure',
                    'violet',
                    'amber',
                ]
            )
        ]
        enhanced_parts.extend(color_items[:5])

        # Medium priority: physical descriptors
        physical_items = [
            detail
            for detail in raw_details
            if any(
                descriptor in detail
                for descriptor in [
                    'long',
                    'short',
                    'tall',
                    'flowing',
                    'glowing',
                    'shining',
                    'sparkling',
                    'ancient',
                    'magical',
                    'mystical',
                    'enchanted',
                ]
            )
            and detail not in color_items
        ]
        enhanced_parts.extend(physical_items[:4])

        # Add remaining details
        remaining_details = [
            detail for detail in raw_details if detail not in color_items and detail not in physical_items
        ]
        enhanced_parts.extend(remaining_details[:3])

    # Add thematic elements
    enhanced_parts.append("cinematic character portrayal")
    enhanced_parts.append("fantasy artwork aesthetic")

    # Add style and quality
    enhanced_parts.append(f"in {style} style")
    enhanced_parts.extend(["high quality", "detailed", "sharp focus", "professional"])

    return ", ".join(enhanced_parts)


# Enhanced sample conversations with more visual details
SAMPLE_CONVERSATIONS = {
    "fantasy_wizard_elena": [
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
            "content": "Elena has long silver hair that flows like moonlight and wears an emerald green cloak with golden thread embroidery. The forest is filled with ancient oak trees and glowing blue mushrooms that light up at night.",
        },
        {
            "role": "assistant",
            "content": "What an enchanting setting! The contrast between Elena's silver hair and emerald cloak against the backdrop of ancient oaks and bioluminescent mushrooms creates a vivid magical atmosphere.",
        },
        {
            "role": "user",
            "content": "She also carries a crystal staff that sparkles with magical energy and wears silver boots that make no sound when she walks.",
        },
        {
            "role": "assistant",
            "content": "The crystal staff and silent silver boots add wonderful mystical elements to her character. These details really bring Elena to life!",
        },
        {"role": "user", "content": "Can you generate an image of Elena standing in her enchanted forest?"},
    ],
    "sci_fi_captain_nova": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": "I'm creating a sci-fi adventure story about Captain Nova, who pilots a sleek silver spacecraft.",
        },
        {"role": "assistant", "content": "Exciting! What kind of adventures does Captain Nova embark on?"},
        {
            "role": "user",
            "content": "She explores distant planets with purple skies and towering crystalline mountains. Her ship has bright blue energy engines that leave glowing trails across space.",
        },
        {
            "role": "assistant",
            "content": "That creates a vivid picture! The purple alien skies and crystal formations with the blue energy trails sounds visually stunning.",
        },
        {
            "role": "user",
            "content": "Captain Nova wears a white space suit with golden insignia and has short red hair. Her ship's hull gleams like polished silver in the alien sunlight.",
        },
        {
            "role": "assistant",
            "content": "The golden insignia on white and her red hair create a striking character design against the silver spacecraft. Very cinematic!",
        },
        {
            "role": "user",
            "content": "Create an image of Captain Nova's silver ship approaching one of these alien worlds with purple skies.",
        },
    ],
    "egyptian_temple": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "I'm researching ancient Egyptian architecture for a historical project."},
        {"role": "assistant", "content": "Egyptian architecture is fascinating! What aspects are you focusing on?"},
        {
            "role": "user",
            "content": "I'm particularly interested in the massive stone temples with towering columns covered in intricate hieroglyphic carvings painted in bright colors.",
        },
        {
            "role": "assistant",
            "content": "The painted hieroglyphs on those monumental columns are truly impressive works of art and engineering.",
        },
        {
            "role": "user",
            "content": "The temples have golden capstones that catch the desert sunlight and cast long shadows across sandstone floors decorated with colorful murals.",
        },
        {
            "role": "assistant",
            "content": "The interplay of golden light, shadows, and colorful decorations must have created an awe-inspiring atmosphere for ancient visitors.",
        },
        {
            "role": "user",
            "content": "Generate an image of a majestic Egyptian temple with golden sunlight streaming through massive columns with colorful hieroglyphs.",
        },
    ],
}


def demonstrate_improved_context_extraction():
    """Demonstrate the improved conversation context extraction for image generation."""

    print("=" * 90)
    print("IMPROVED IMAGE GENERATION WITH SPECIFIC VISUAL DETAIL EXTRACTION")
    print("=" * 90)

    for scenario_name, messages in SAMPLE_CONVERSATIONS.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name.replace('_', ' ').title()}")
        print(f"{'='*70}")

        # Show the conversation history
        print("\nConversation History:")
        print("-" * 40)
        for msg in messages:
            if msg["role"] != "system":
                role = msg["role"].title()
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"{role}: {content}")

        # Extract visual details
        raw_details = simulate_raw_visual_extraction(messages)
        user_request = messages[-1]["content"]

        print(f"\n\nEXTRACTED VISUAL DETAILS:")
        print("-" * 40)
        print(f"Found {len(raw_details)} specific visual details:")
        for i, detail in enumerate(raw_details, 1):
            print(f"  {i:2d}. {detail}")

        # Generate enhanced prompts
        print(f"\n\nPROMPT ENHANCEMENT COMPARISON:")
        print("-" * 40)

        # Determine subject from scenario
        if "elena" in scenario_name:
            subject = "young wizard Elena in mystical forest"
        elif "nova" in scenario_name:
            subject = "silver spacecraft approaching alien planet"
        elif "egyptian" in scenario_name:
            subject = "ancient Egyptian temple interior"
        else:
            subject = "scene from conversation"

        # OLD WAY (basic enhancement)
        print("\n1. OLD METHOD (Basic Enhancement):")
        basic_prompt = f"{subject}, high quality, detailed, sharp focus, professional"
        print(f"   Enhanced Prompt: {basic_prompt}")

        # NEW WAY (with specific visual details)
        print("\n2. NEW METHOD (With Conversation Context):")
        enhanced_prompt = simulate_enhanced_prompt_generation(user_request, subject, raw_details)
        print(f"   Enhanced Prompt: {enhanced_prompt}")

        # Show the improvement
        improvement_items = [detail for detail in raw_details if detail in enhanced_prompt]
        print(f"\n   IMPROVEMENT: Added {len(improvement_items)} specific details from conversation:")
        for detail in improvement_items[:5]:  # Show first 5
            print(f"   • {detail}")
        if len(improvement_items) > 5:
            print(f"   • ... and {len(improvement_items) - 5} more")

        print("\n" + "=" * 70)


def show_regex_extraction_examples():
    """Show examples of what the regex patterns capture."""

    print("\n" + "=" * 90)
    print("REGEX PATTERN EXTRACTION EXAMPLES")
    print("=" * 90)

    test_sentences = [
        "Elena has long silver hair and wears an emerald green cloak.",
        "Her crystal staff sparkles with magical energy in the moonlight.",
        "The ancient oak trees tower above with glowing blue mushrooms below.",
        "Captain Nova wears a white space suit with golden insignia.",
        "The temple has massive stone columns covered in colorful hieroglyphs.",
        "Golden sunlight streams through the towering sandstone pillars.",
    ]

    print("\nTest sentences and what gets extracted:")
    print("-" * 50)

    for sentence in test_sentences:
        print(f"\nInput: \"{sentence}\"")

        # Simulate extraction on single sentence
        mock_message = [{"role": "user", "content": sentence}]
        extracted = simulate_raw_visual_extraction(mock_message)

        print(f"Extracted: {', '.join(extracted) if extracted else 'None'}")


if __name__ == "__main__":
    print("IMPROVED IMAGE GENERATION CONVERSATION CONTEXT DEMO")
    print("This demo shows how specific visual details are now extracted from conversation history.")
    print("\nNOTE: This demonstrates the extraction logic. For actual image generation,")
    print("you need to have the IMAGE_ENDPOINT environment variable configured.")
    print("\n" + "=" * 90)

    try:
        demonstrate_improved_context_extraction()
        show_regex_extraction_examples()

        print("\n" + "=" * 90)
        print("DEMO COMPLETED")
        print("=" * 90)
        print("\nKey Improvements in the New Implementation:")
        print("• Extracts specific visual details directly from conversation messages")
        print("• Captures colors, physical descriptions, clothing, and environmental details")
        print("• Prioritizes color+object combinations for better visual coherence")
        print("• Combines specific details with high-level thematic elements")
        print("• Results in much more detailed and contextually accurate image prompts")
        print("• Perfect for collaborative storytelling and character consistency")

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
