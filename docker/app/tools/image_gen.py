import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from models.chat_config import ChatConfig
from PIL import Image
from pydantic import Field
from tools.base import BaseTool, BaseToolResponse, ExecutionMode
from utils.image import generate_image

# Configure logger
logger = logging.getLogger(__name__)

# Aspect ratio mappings to dimensions
ASPECT_RATIO_MAPPINGS = {
    "square": (1024, 1024),  # 1:1 ratio
    "portrait": (768, 1024),  # 3:4 ratio (vertical)
    "landscape": (1024, 768),  # 4:3 ratio (horizontal)
}

ALLOWED_ASPECT_RATIOS = list(ASPECT_RATIO_MAPPINGS.keys())


def get_dimensions_from_aspect_ratio(aspect_ratio: str) -> Tuple[int, int]:
    """
    Get width and height dimensions from aspect ratio string

    Args:
        aspect_ratio: One of "square", "portrait", or "landscape"

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If aspect_ratio is not recognized
    """
    if aspect_ratio not in ASPECT_RATIO_MAPPINGS:
        raise ValueError(
            f"Invalid aspect ratio '{aspect_ratio}'. Must be one of: {', '.join(ALLOWED_ASPECT_RATIOS)}"
        )

    return ASPECT_RATIO_MAPPINGS[aspect_ratio]


class ImageGenerationResponse(BaseToolResponse):
    """Response from image generation tool"""

    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    original_prompt: str = Field(description="Original user prompt")
    enhanced_prompt: str = Field(description="Enhanced prompt used for generation")
    direct_response: bool = Field(
        False, description="Whether this is a direct response"
    )
    result: Optional[str] = Field(
        None, description="JSON string representation of the response"
    )


class ImageGenerationTool(BaseTool):
    """Tool for generating images with AI"""

    def __init__(self):
        super().__init__()
        self.name = "generate_image"
        self.description = "Generate images or visualizations from text descriptions. Use when user requests creating, generating, making, or drawing images or other visuals. OK to use for graphs, charts or signs with text."
        self.supported_contexts = ['image_generation']
        self.execution_mode = ExecutionMode.SYNC  # Image generation is synchronous
        self.timeout = 120.0  # Image generation can take longer

    def _initialize_mvc(self):
        """Initialize MVC components"""
        # This tool doesn't need separate MVC components as it's simple
        self._controller = None
        self._view = None

    def get_definition(self) -> Dict[str, Any]:
        """
        Return OpenAI-compatible tool definition

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
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for the image. Choose based on the content: 'square' for balanced compositions, social media posts, or general purpose images; 'portrait' for vertical subjects like people, tall buildings, or phone wallpapers; 'landscape' for wide scenes, natural vistas, or desktop wallpapers.",
                            "enum": ALLOWED_ASPECT_RATIOS,
                            "default": "square",
                        },
                        "cfg_scale": {
                            "type": "number",
                            "description": "Guidance scale for how closely the image follows the text prompt. Higher values (3.5-4.5) give closer adherence to prompt but may reduce image quality. Lower values (1.5-3.0) allow more creative interpretation with potentially better image quality.",
                            "minimum": 1.5,
                            "maximum": 4.5,
                            "default": 3.5,
                        },
                        "use_conversation_context": {
                            "type": "boolean",
                            "description": "Whether to use conversation history to enhance the prompt. Useful for generating images related to ongoing discussions or stories.",
                            "default": True,
                        },
                        "but_why": {
                            "type": "integer",
                            "description": "An integer from 1-5 where a larger number indicates confidence this is the right tool to help the user.",
                        },
                        "enhanced_prompt": {
                            "type": "string",
                            "description": "The enhanced/rewritten prompt to use for image generation. If the original user prompt is already detailed and well-formed, you can use it as-is. Otherwise, enhance it with more specific details, artistic direction, and visual elements to improve the image generation result.",
                        },
                    },
                    "required": [
                        "user_prompt",
                        "aspect_ratio",
                        "cfg_scale",
                        "use_conversation_context",
                        "but_why",
                        "enhanced_prompt",
                    ],
                },
            },
        }

    def get_response_type(self) -> Type[ImageGenerationResponse]:
        """Get the response type for this tool"""
        return ImageGenerationResponse

    def _execute_sync(self, params: Dict[str, Any]) -> ImageGenerationResponse:
        """Execute the tool synchronously"""
        user_prompt = params.get("user_prompt")
        aspect_ratio = params.get("aspect_ratio", "square")
        cfg_scale = params.get("cfg_scale", 3.5)
        use_conversation_context = params.get("use_conversation_context", True)
        enhanced_prompt = params.get("enhanced_prompt", user_prompt)
        messages = params.get("messages")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self.generate_image_from_prompt(
            user_prompt,
            aspect_ratio,
            cfg_scale,
            use_conversation_context,
            enhanced_prompt,
            config,
            messages,
        )

    def execute(self, params: Dict[str, Any]) -> ImageGenerationResponse:
        """Execute the tool with given parameters"""
        # This tool doesn't use MVC pattern, so execute directly
        return self._execute_sync(params)

    def _get_conversation_context(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation context for image generation using conversation_context tool

        Args:
            messages: List of conversation messages

        Returns:
            Dictionary with 'summary' or None if context couldn't be retrieved
        """
        if not messages or len(messages) < 2:
            return None

        try:
            logger.info("Retrieving conversation context for image generation")

            # Get high-level context summary focused on visual elements
            context_params = {
                "query": "conversation_summary",
                "max_messages": 8,  # Look at more messages for richer context
                "focus_query": "visual elements, story details, characters, settings, artistic themes, and descriptive elements that could be visualized",
                "messages": messages,
                "but_why": "Gathering conversation context to enhance image generation prompt with relevant visual details",
            }

            # Use the tool registry to execute conversation context tool
            from tools.registry import execute_tool

            context_response = execute_tool("conversation_context", context_params)
            summary = None
            if context_response and context_response.success:
                summary = context_response.analysis

            logger.info(
                f"Retrieved context - summary: {len(summary) if summary else 0} chars"
            )

            return {"summary": summary}

        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return None

    def _generate_image_with_config(
        self,
        enhanced_prompt: str,
        config: ChatConfig,
        width: int = 512,
        height: int = 512,
        cfg_scale: float = 3.5,
    ) -> Optional[Image.Image]:
        """
        Generate image using the enhanced prompt and configuration

        Args:
            enhanced_prompt: Enhanced prompt for image generation
            config: Chat configuration containing image endpoint
            width: Image width in pixels
            height: Image height in pixels
            cfg_scale: Guidance scale for image generation (1.5-4.5)

        Returns:
            Generated PIL Image or None if failed
        """
        if not config.image_endpoint:
            logger.error("Image generation endpoint not configured")
            return None

        try:
            logger.info(
                f"Generating image with prompt: '{enhanced_prompt}', dimensions: {width}x{height}, cfg_scale: {cfg_scale}"
            )
            generated_image = generate_image(
                invoke_url=config.image_endpoint,
                prompt=enhanced_prompt,
                mode="base",
                width=width,
                height=height,
                cfg_scale=cfg_scale,
                return_bytes_io=False,
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
        aspect_ratio: str,
        cfg_scale: float,
        use_conversation_context: bool,
        enhanced_prompt: str,
        config: ChatConfig = None,
        messages: List[Dict[str, Any]] = None,
    ) -> ImageGenerationResponse:
        """
        Generate an image based on user prompt with enhancement

        Args:
            user_prompt: Original user prompt
            aspect_ratio: Aspect ratio for the image ("square", "portrait", or "landscape")
            cfg_scale: Guidance scale for image generation (1.5-4.5)
            use_conversation_context: Whether to use conversation context
            enhanced_prompt: The enhanced/rewritten prompt to use for image generation
            config: Chat configuration
            messages: Optional conversation messages for context

        Returns:
            ImageGenerationResponse with the result
        """
        if config is None:
            config = ChatConfig.from_environment()

        # Validate and convert aspect ratio to dimensions
        try:
            width, height = get_dimensions_from_aspect_ratio(aspect_ratio)
        except ValueError as e:
            logger.warning(
                f"Invalid aspect ratio '{aspect_ratio}', defaulting to 'square': {e}"
            )
            width, height = get_dimensions_from_aspect_ratio("square")
            aspect_ratio = "square"

        # Validate cfg_scale
        if not (1.5 <= cfg_scale <= 4.5):
            logger.warning(f"Invalid cfg_scale {cfg_scale}, defaulting to 3.5")
            cfg_scale = 3.5

        # Get conversation context if requested and available
        conversation_context = None
        if use_conversation_context and messages:
            conversation_context = self._get_conversation_context(messages)

        # Use the provided enhanced prompt, or fall back to user_prompt if not provided
        if enhanced_prompt is None:
            enhanced_prompt = user_prompt
            logger.info(f"No enhanced prompt provided, using original: '{user_prompt}'")
        else:
            logger.info(f"Using provided enhanced prompt: '{enhanced_prompt}'")

        # Check if image endpoint is configured
        if not config.image_endpoint:
            response_dict = {
                "success": False,
                "original_prompt": user_prompt,
                "enhanced_prompt": enhanced_prompt,
                "error_message": "Image generation is not configured. Please set the IMAGE_ENDPOINT environment variable.",
                "error_code": "CONFIGURATION_ERROR",
                "direct_response": True,
            }
            import json

            return ImageGenerationResponse(
                **response_dict, result=json.dumps(response_dict)
            )

        # Generate the image
        generated_image = self._generate_image_with_config(
            enhanced_prompt, config, width, height, cfg_scale
        )

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
                    "aspect_ratio": aspect_ratio,
                    "dimensions": f"{width}x{height}",
                    "cfg_scale": cfg_scale,
                    "direct_response": True,
                    "message": f"Successfully generated {aspect_ratio} ({width}x{height}) image with cfg_scale {cfg_scale} and prompt: {enhanced_prompt}",
                }

                # Add context info if used
                if use_conversation_context and conversation_context:
                    result_dict["used_conversation_context"] = True
                    summary = conversation_context.get("summary", "")
                    result_dict["context_summary"] = (
                        summary[:200] + "..." if len(summary) > 200 else summary
                    )
                else:
                    result_dict["used_conversation_context"] = False

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
                    "error_code": "IMAGE_PROCESSING_ERROR",
                    "direct_response": True,
                }
                import json

                return ImageGenerationResponse(
                    **response_dict, result=json.dumps(response_dict)
                )
        else:
            response_dict = {
                "success": False,
                "original_prompt": user_prompt,
                "enhanced_prompt": enhanced_prompt,
                "error_message": "Failed to generate image. Please try again with a different prompt.",
                "error_code": "GENERATION_FAILED",
                "direct_response": True,
            }
            import json

            return ImageGenerationResponse(
                **response_dict, result=json.dumps(response_dict)
            )

    def run_with_dict(self, params: Dict[str, Any]) -> ImageGenerationResponse:
        """
        Execute image generation with parameters provided as a dictionary
        This method exists for backward compatibility.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'user_prompt', 'aspect_ratio', 'cfg_scale', 'use_conversation_context', 'enhanced_prompt', optionally 'messages'

        Returns:
            ImageGenerationResponse: The image generation result
        """
        if "user_prompt" not in params:
            raise ValueError("'user_prompt' key is required in parameters dictionary")
        if "enhanced_prompt" not in params:
            raise ValueError(
                "'enhanced_prompt' key is required in parameters dictionary"
            )

        user_prompt = params["user_prompt"]
        aspect_ratio = params.get("aspect_ratio", "square")
        cfg_scale = params.get("cfg_scale", 3.5)
        use_conversation_context = params.get("use_conversation_context", True)
        enhanced_prompt = params["enhanced_prompt"]
        messages = params.get("messages", None)

        logger.debug(
            f"run_with_dict called with user_prompt: '{user_prompt}', aspect_ratio: {aspect_ratio}, cfg_scale: {cfg_scale}, use_context: {use_conversation_context}, enhanced_prompt: '{enhanced_prompt}'"
        )

        # Add more detailed logging to debug the issue
        logger.debug(f"Image generation parameters:")
        logger.debug(f"user_prompt: '{user_prompt}'")
        logger.debug(f"use_conversation_context: {use_conversation_context}")
        logger.debug(f"Number of messages: {len(messages) if messages else 0}")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self.generate_image_from_prompt(
            user_prompt,
            aspect_ratio,
            cfg_scale,
            use_conversation_context,
            enhanced_prompt,
            config,
            messages,
        )


# Helper functions for backward compatibility
def get_image_generation_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("generate_image", ImageGenerationTool)

    # Get the tool instance and return its definition
    tool = get_tool("generate_image")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get image generation tool definition")
