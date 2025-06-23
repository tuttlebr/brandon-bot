import logging
import re
from typing import Optional, Tuple

from models.chat_config import ChatConfig
from PIL import Image
from utils.image import generate_image


class ImageService:
    """Service for handling image generation operations"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the image service

        Args:
            config: Configuration for the image service
        """
        self.config = config

    def detect_image_generation_request(self, prompt: str) -> bool:
        """
        Check if the prompt contains image generation trigger words

        Args:
            prompt: The user's prompt

        Returns:
            True if image generation is requested, False otherwise
        """
        trigger_patterns = [
            r'\b(generate|create|make|draw|produce|build)\s+(?:an?\s+)?(?:image|photo)\b',
            r'\b(?:image|photo)\s+(?:of|with|showing|depicting)\b',
            r'\b(generate|create|make|draw|produce|build)\s+(?:a\s+)?(?:picture|drawing|illustration|artwork)\b',
            r'\bpicture\s+(?:of|with|showing|depicting)\b',
            r'\b(?:show|visualize)\s+me\s+(?:an?\s+)?(?:image|picture|photo)\b',
            r'\b(?:draw|sketch|paint)\s+(?:me\s+)?(?:an?\s+)?\b',
            r'\bvisual(?:ize|ization)\s+(?:of|this|that)\b',
        ]

        prompt_lower = prompt.lower()
        return any(re.search(pattern, prompt_lower) for pattern in trigger_patterns)

    def extract_image_prompt(self, prompt: str) -> str:
        """
        Extract the actual image description from the user's message

        Args:
            prompt: The user's original prompt

        Returns:
            The extracted image description
        """
        # Remove common trigger phrases and extract the core description
        trigger_removals = [
            r'\b(?:generate|create|make|draw|produce|build)\s+(?:an?\s+)?(?:image|photo|picture|drawing|illustration|artwork)\s+(?:of|with|showing|depicting)\s*',
            r'\b(?:show|visualize)\s+me\s+(?:an?\s+)?(?:image|photo|picture)\s+(?:of|with|showing|depicting)\s*',
            r'\b(?:draw|sketch|paint)\s+(?:me\s+)?(?:an?\s+)?\s*',
            r'\bvisual(?:ize|ization)\s+(?:of|this|that)\s*',
            r'\b(?:image|photo)\s+(?:of|with|showing|depicting)\s*',
            r'\bpicture\s+(?:of|with|showing|depicting)\s*',
        ]

        extracted_prompt = prompt
        for pattern in trigger_removals:
            extracted_prompt = re.sub(pattern, '', extracted_prompt, flags=re.IGNORECASE)

        # Clean up extra whitespace and punctuation
        extracted_prompt = re.sub(r'\s+', ' ', extracted_prompt).strip()
        extracted_prompt = extracted_prompt.strip('.,!?')

        # If nothing meaningful remains, return the original prompt
        if len(extracted_prompt) < 5:
            return prompt

        return extracted_prompt

    def generate_image_response(self, image_prompt: str) -> Tuple[Optional[Image.Image], str]:
        """
        Generate an image using the image generation service

        Args:
            image_prompt: The prompt for image generation

        Returns:
            Tuple of (generated_image, confirmation_message)
        """
        # Check if image endpoint is configured
        if not self.config.image_endpoint:
            return None, "Image generation is not configured. Please set the IMAGE_ENDPOINT environment variable."

        try:
            # Generate image using the image generation service
            generated_image = generate_image(
                invoke_url=self.config.image_endpoint, prompt=image_prompt, mode="base", return_bytes_io=False,
            )

            if generated_image:
                confirmation_msg = f"I've generated an image based on your request: '{image_prompt}'"
                return generated_image, confirmation_msg
            else:
                return None, "I apologize, but I wasn't able to generate an image at this time."

        except Exception as e:
            logging.error(f"Image generation failed: {e}")
            return None, f"I encountered an error while generating the image: {str(e)}"
