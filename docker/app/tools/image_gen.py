import logging
from typing import Any, Dict, Optional

from models.chat_config import ChatConfig
from PIL import Image
from pydantic import BaseModel, Field
from utils.image import generate_image, pil_image_to_base64

# Configure logger
logger = logging.getLogger(__name__)


class ImageGenerationResponse(BaseModel):
    """Response from image generation tool"""

    success: bool = Field(description="Whether the image was generated successfully")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    original_prompt: str = Field(description="Original user prompt")
    enhanced_prompt: str = Field(description="Enhanced prompt used for generation")
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    direct_response: bool = Field(False, description="Whether this is a direct response")
    result: Optional[str] = Field(None, description="JSON string representation of the response")


class ImageGenerationTool:
    """Tool for generating images with prompt enhancement"""

    def __init__(self):
        self.name = "generate_image"
        self.description = "Triggered when user requests image generation with phrases like 'create an image', 'generate a picture', 'draw', 'make an image', or 'show me a picture'. Enhances user prompts to create detailed, high-quality image descriptions for optimal generation results."

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI function calling format

        Returns:
            Dict containing the OpenAI-compatible tool definition
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_prompt": {
                            "type": "string",
                            "description": "The user's original message requesting image generation",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Main subject or object to be depicted in the image",
                        },
                        "style": {
                            "type": "string",
                            "description": "Artistic style or aesthetic (e.g., 'photorealistic', 'digital art', 'oil painting', 'minimalist', 'fantasy')",
                            "default": "photorealistic",
                        },
                        "mood": {
                            "type": "string",
                            "description": "Mood or atmosphere (e.g., 'bright and cheerful', 'dark and moody', 'serene', 'dramatic')",
                            "default": "natural",
                        },
                        "details": {
                            "type": "string",
                            "description": "Additional details, lighting, colors, or specific elements to include",
                        },
                    },
                    "required": ["user_prompt", "subject"],
                },
            },
        }

    def _enhance_prompt(
        self, user_prompt: str, subject: str, style: str = "photorealistic", mood: str = "natural", details: str = ""
    ) -> str:
        """
        Enhance the user's prompt to create a more detailed image generation prompt

        Args:
            user_prompt: Original user prompt
            subject: Main subject of the image
            style: Artistic style
            mood: Mood/atmosphere
            details: Additional details

        Returns:
            Enhanced prompt for image generation
        """
        # Base prompt with the main subject
        enhanced_parts = [subject]

        # Add style information
        if style and style.lower() != "natural":
            enhanced_parts.append(f"in {style} style")

        # Add mood/atmosphere
        if mood and mood.lower() != "natural":
            enhanced_parts.append(f"with {mood} atmosphere")

        # Add specific details
        if details:
            enhanced_parts.append(details)

        # Add quality enhancers for better results
        quality_enhancers = ["high quality", "detailed", "sharp focus", "professional"]

        enhanced_prompt = ", ".join(enhanced_parts + quality_enhancers)

        logger.info(f"Enhanced prompt: '{user_prompt}' -> '{enhanced_prompt}'")
        return enhanced_prompt

    def _generate_image_with_config(self, enhanced_prompt: str, config: ChatConfig) -> Optional[Image.Image]:
        """
        Generate image using the enhanced prompt and configuration

        Args:
            enhanced_prompt: Enhanced prompt for image generation
            config: Chat configuration containing image endpoint

        Returns:
            Generated PIL Image or None if failed
        """
        if not config.image_endpoint:
            logger.error("Image generation endpoint not configured")
            return None

        try:
            logger.info(f"Generating image with prompt: '{enhanced_prompt}'")
            generated_image = generate_image(
                invoke_url=config.image_endpoint, prompt=enhanced_prompt, mode="base", return_bytes_io=False,
            )

            if generated_image:
                logger.info("Image generated successfully")
                return generated_image
            else:
                logger.error("Image generation returned None")
                return None

        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            return None

    def generate_image_from_prompt(
        self,
        user_prompt: str,
        subject: str,
        style: str = "photorealistic",
        mood: str = "natural",
        details: str = "",
        config: ChatConfig = None,
    ) -> ImageGenerationResponse:
        """
        Generate an image based on user prompt with enhancement

        Args:
            user_prompt: Original user prompt
            subject: Main subject for the image
            style: Artistic style
            mood: Mood/atmosphere
            details: Additional details
            config: Chat configuration

        Returns:
            ImageGenerationResponse with the result
        """
        if config is None:
            config = ChatConfig.from_environment()

        # Enhance the prompt
        enhanced_prompt = self._enhance_prompt(user_prompt, subject, style, mood, details)

        # Check if image endpoint is configured
        if not config.image_endpoint:
            response_dict = {
                "success": False,
                "original_prompt": user_prompt,
                "enhanced_prompt": enhanced_prompt,
                "error_message": "Image generation is not configured. Please set the IMAGE_ENDPOINT environment variable.",
                "direct_response": True,
            }
            import json

            return ImageGenerationResponse(**response_dict, result=json.dumps(response_dict))

        # Generate the image
        generated_image = self._generate_image_with_config(enhanced_prompt, config)

        if generated_image:
            # Handle the returned image data (it's already base64 string when return_bytes_io=False)
            try:
                # Since return_bytes_io=False, generated_image is already a base64 string
                image_b64 = generated_image

                # Create response dict WITHOUT image_data for the result field (to avoid massive strings in LLM processing)
                result_dict = {
                    "success": True,
                    "original_prompt": user_prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "direct_response": True,
                    "message": f"Successfully generated image with enhanced prompt: {enhanced_prompt}",
                }

                import json

                return ImageGenerationResponse(
                    success=True,
                    image_data=image_b64,  # Keep image data in response object for Streamlit app
                    original_prompt=user_prompt,
                    enhanced_prompt=enhanced_prompt,
                    direct_response=True,
                    result=json.dumps(result_dict),  # Result field excludes image_data
                )
            except Exception as e:
                logger.error(f"Error converting image to base64: {e}")
                response_dict = {
                    "success": False,
                    "original_prompt": user_prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "error_message": f"Error processing generated image: {str(e)}",
                    "direct_response": True,
                }
                import json

                return ImageGenerationResponse(**response_dict, result=json.dumps(response_dict))
        else:
            response_dict = {
                "success": False,
                "original_prompt": user_prompt,
                "enhanced_prompt": enhanced_prompt,
                "error_message": "Failed to generate image. Please try again with a different prompt.",
                "direct_response": True,
            }
            import json

            return ImageGenerationResponse(**response_dict, result=json.dumps(response_dict))

    def run_with_dict(self, params: Dict[str, Any]) -> ImageGenerationResponse:
        """
        Execute image generation with parameters provided as a dictionary

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'user_prompt', 'subject', optionally 'style', 'mood', 'details'

        Returns:
            ImageGenerationResponse: The image generation result
        """
        if "user_prompt" not in params:
            raise ValueError("'user_prompt' key is required in parameters dictionary")
        if "subject" not in params:
            raise ValueError("'subject' key is required in parameters dictionary")

        user_prompt = params["user_prompt"]
        subject = params["subject"]
        style = params.get("style", "photorealistic")
        mood = params.get("mood", "natural")
        details = params.get("details", "")

        logger.debug(f"run_with_dict called with user_prompt: '{user_prompt}', subject: '{subject}'")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self.generate_image_from_prompt(user_prompt, subject, style, mood, details, config)


# Create a global instance and helper functions for easy access
image_generation_tool = ImageGenerationTool()


def get_image_generation_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for image generation

    Returns:
        Dict containing the OpenAI tool definition
    """
    return image_generation_tool.to_openai_format()


def execute_image_generation(
    user_prompt: str, subject: str, style: str = "photorealistic", mood: str = "natural", details: str = ""
) -> ImageGenerationResponse:
    """
    Execute image generation with the given parameters

    Args:
        user_prompt: Original user prompt
        subject: Main subject for the image
        style: Artistic style
        mood: Mood/atmosphere
        details: Additional details

    Returns:
        ImageGenerationResponse: The image generation result
    """
    config = ChatConfig.from_environment()
    return image_generation_tool.generate_image_from_prompt(user_prompt, subject, style, mood, details, config)


def execute_image_generation_with_dict(params: Dict[str, Any]) -> ImageGenerationResponse:
    """
    Execute image generation with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters
               Expected keys: 'user_prompt', 'subject', optionally 'style', 'mood', 'details'

    Returns:
        ImageGenerationResponse: The image generation result
    """
    return image_generation_tool.run_with_dict(params)
