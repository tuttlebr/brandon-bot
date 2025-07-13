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

    def generate_image_response(
        self, image_prompt: str
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Generate an image using the image generation service

        Args:
            image_prompt: The prompt for image generation

        Returns:
            Tuple of (generated_image, confirmation_message)
        """
        # Check if image endpoint is configured
        if not self.config.image_endpoint:
            return (
                None,
                "Image generation is not configured. Please set the IMAGE_ENDPOINT environment variable.",
            )

        try:
            # Generate image using the image generation service
            generated_image = generate_image(
                invoke_url=self.config.image_endpoint,
                prompt=image_prompt,
                mode="base",
                return_bytes_io=True,
            )

            if generated_image:
                confirmation_msg = (
                    f"I've generated an image based on your request: '{image_prompt}'"
                )
                return generated_image, ""
            else:
                return (
                    None,
                    "I apologize, but I wasn't able to generate an image at this time.",
                )

        except Exception as e:
            logging.error(f"Image generation failed: {e}")
            return None, f"I encountered an error while generating the image: {str(e)}"
