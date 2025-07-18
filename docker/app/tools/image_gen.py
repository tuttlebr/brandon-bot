import logging
from typing import Any, Dict, List, Optional, Tuple, Type

from models.chat_config import ChatConfig
from openai import OpenAI
from PIL import Image
from pydantic import Field
from tools.base import BaseTool, BaseToolResponse, ExecutionMode
from utils.config import config as app_config
from utils.image import generate_image

# Configure logger
logger = logging.getLogger(__name__)

# Aspect ratio mappings to dimensions
ASPECT_RATIO_MAPPINGS = {
    "square": (768, 768),  # 1:1 ratio
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
        self.description = "ONLY use when explicitly asked to create, generate, or make an image, picture, photo, artwork, or visual content. Creates AI-generated images based on text descriptions, supporting various artistic styles, moods, and aspect ratios. DO NOT use for general questions, information lookup, or non-image requests."
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
                            "type": "string",
                            "description": "A single sentence explaining why this tool was selected for the query.",
                        },
                    },
                    "required": ["user_prompt", "subject", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[ImageGenerationResponse]:
        """Get the response type for this tool"""
        return ImageGenerationResponse

    def _execute_sync(self, params: Dict[str, Any]) -> ImageGenerationResponse:
        """Execute the tool synchronously"""
        user_prompt = params.get("user_prompt")
        subject = params.get("subject")
        style = params.get("style", "photorealistic")
        mood = params.get("mood", "natural")
        details = params.get("details", "")
        aspect_ratio = params.get("aspect_ratio", "square")
        cfg_scale = params.get("cfg_scale", 3.5)
        use_conversation_context = params.get("use_conversation_context", True)
        messages = params.get("messages")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self.generate_image_from_prompt(
            user_prompt,
            subject,
            style,
            mood,
            details,
            aspect_ratio,
            cfg_scale,
            use_conversation_context,
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

    def _enhance_prompt_with_llm(
        self,
        user_prompt: str,
        subject: str,
        style: str = "photorealistic",
        mood: str = "natural",
        details: str = "",
        conversation_context: Optional[Dict[str, Any]] = None,
        config: ChatConfig = None,
    ) -> str:
        """
        Use LLM to intelligently enhance the user's prompt for better image generation

        Args:
            user_prompt: Original user prompt
            subject: Main subject of the image
            style: Artistic style
            mood: Mood/atmosphere
            details: Additional details
            conversation_context: Optional conversation context dictionary with 'raw_details' and 'summary'
            config: Chat configuration for LLM access

        Returns:
            LLM-enhanced prompt for image generation
        """
        if config is None:
            config = ChatConfig.from_environment()

        # Debug logging
        logger.debug(f"_enhance_prompt_with_llm called with:")
        logger.debug(f"user_prompt: '{user_prompt}'")
        logger.debug(f"subject: '{subject}'")
        logger.debug(f"has conversation_context: {conversation_context is not None}")
        if conversation_context:
            summary = conversation_context.get("summary", "")
            logger.debug(f"context summary length: {len(summary)} chars")
            logger.debug(
                f"context summary preview: {summary[:200]}..."
                if len(summary) > 200
                else f"context summary: {summary}"
            )

        try:
            # Initialize fast LLM client for prompt enhancement
            fast_client = OpenAI(
                api_key=config.api_key, base_url=config.fast_llm_endpoint
            )

            # Create enhancement prompt for the LLM
            enhancement_system_prompt = """detailed thinking off
            **Expert Prompt Refinement for AI Image Generation**

Acting as a seasoned imaging specialist, your task is to elevate a user's foundational image request into a sophisticated, visually evocative prompt that consistently yields high-impact results. Synergize technical precision with artistic nuance to meet the user's creative vision without imposing undue constraints.

**CRITICAL RULE:** The user's original request MUST be the primary focus. Conversation context should only be used to add subtle enhancements, NOT to replace or overwhelm the original request.

**CORE DIRECTIVES:**
- **Vivid Specification:** Convert basic input into immersive, detailed descriptions that stimulate superior visual outputs.
- **Artistic & Technical Integration:** Seamlessly incorporate relevant techniques (e.g., chiaroscuro, impasto), lighting dynamics, compositional principles, and atmospheric conditions.
- **Contextual Harmony:** Naturally assimilate contextual information to enrich the prompt's narrative or conceptual depth WITHOUT changing the core subject.
- **Creative Fidelity:** Preserve the user's core intent while augmenting with discipline-specific terminology (e.g., 'tenebrism' for dramatic lighting).
- **Linguistic Precision:** Employ visceral, evocative language to supplant generic descriptors, amplifying visual impact.
- **Visual Hierarchy:** Prioritize key elements: chromatic schemes, textural contrasts, luminous qualities, spatial composition, and perspectival choices.
- **Concise Elegance:** Balance descriptive richness with brevity to avoid prompt fatigue. response should be only one or two sentences.
- **Quality Signaling:** Embed style-appropriate quality indicators (e.g., '8K resolution' for photorealism, 'intricate linework' for digital art).

**STYLE-SPECIFIC ENHANCEMENT PROTOCOLS:**
- **Photorealistic:** Specify camera parameters (aperture, shutter speed), lighting setups (golden hour, rim lighting), and professional capture terminology.
- **Digital Art:** Detail rendering methodologies (e.g., cel-shading, volumetric lighting), artistic software effects, and post-processing techniques.
- **Oil Painting:** Reference classical approaches (e.g., alla prima, glazing), brushwork characteristics (visible strokes, impasto textures), and art historical periods (Baroque, Impressionist).
- **Fantasy:** Integrate magical phenomena, ethereal luminosity, and mythopoeic motifs while maintaining visual coherence.
- **Minimalist:** Emphasize austere composition, strategic negative space, and the strategic deployment of simple, potent visual elements.

**OUTPUT PARAMETERS:** Deliver the refined prompt ONLY as sentences, no lists, no markdown, ensuring adherence to the aforementioned standards. Do not include any other text or comments regarding what you did to improved the prompt. The output MUST be about the original subject requested, not about conversation history or other topics."""

            user_message = f"""**Original Request:** "{user_prompt}"
**Subject:** {subject}
**Style:** {style}
**Mood:** {mood}
{f'**Details:** {details}' if details else ''}

**Task:** Transform the original request into a single, vivid image generation prompt. Focus ONLY on generating an image of "{subject}" as requested. Any context should be used subtly for atmosphere only."""

            # Add conversation context at the end if available, but make it clear it's secondary
            if conversation_context and conversation_context.get("summary"):
                user_message += f"\n\n**Background Context (use sparingly for subtle enhancement only):**\n{conversation_context['summary']}"

            # Call the fast LLM for enhancement
            response = fast_client.chat.completions.create(
                model=config.fast_llm_model_name,
                messages=[
                    {"role": "system", "content": enhancement_system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=app_config.llm.DEFAULT_TEMPERATURE,  # Some creativity but controlled
                top_p=app_config.llm.DEFAULT_TOP_P,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                max_tokens=128,  # Keep prompts reasonable length
                stream=False,
            )

            enhanced_prompt = response.choices[0].message.content.strip()

            # Debug logging to see what the LLM returned
            logger.debug(f"LLM returned enhanced prompt: '{enhanced_prompt}'")
            logger.debug(f"Enhanced prompt length: {len(enhanced_prompt)} chars")

            # Fallback validation - ensure we got a reasonable response
            if not enhanced_prompt or len(enhanced_prompt) < 20:
                logger.warning(
                    "LLM enhancement produced insufficient result, falling back to basic enhancement"
                )
                return self._basic_prompt_fallback(
                    user_prompt, subject, style, mood, details, conversation_context
                )

            logger.debug(f"LLM Enhanced prompt: '{user_prompt}' -> '{enhanced_prompt}'")
            return enhanced_prompt

        except Exception as e:
            logger.error(f"Error in LLM prompt enhancement: {e}")
            # Fallback to basic enhancement if LLM fails
            return self._basic_prompt_fallback(
                user_prompt, subject, style, mood, details, conversation_context
            )

    def _basic_prompt_fallback(
        self,
        user_prompt: str,
        subject: str,
        style: str = "photorealistic",
        mood: str = "natural",
        details: str = "",
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Fallback basic prompt enhancement if LLM enhancement fails

        Args:
            user_prompt: Original user prompt
            subject: Main subject of the image
            style: Artistic style
            mood: Mood/atmosphere
            details: Additional details
            conversation_context: Optional conversation context dictionary

        Returns:
            Basic enhanced prompt for image generation
        """
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

        # Add basic conversation context
        if conversation_context:
            summary = conversation_context.get("summary", "")
            if summary:
                # Extract key words from summary for basic enhancement
                summary_words = summary.lower().split()
                visual_keywords = [
                    word
                    for word in summary_words
                    if word
                    in [
                        "bright",
                        "dark",
                        "colorful",
                        "vibrant",
                        "scenic",
                        "beautiful",
                        "dramatic",
                        "peaceful",
                        "stormy",
                        "sunny",
                        "cloudy",
                        "misty",
                        "clear",
                        "natural",
                    ]
                ]
                enhanced_parts.extend(
                    visual_keywords[:2]
                )  # Add up to 2 relevant keywords

        # Add quality enhancers for better results
        quality_enhancers = ["high quality", "detailed", "sharp focus", "professional"]
        enhanced_prompt = ", ".join(enhanced_parts + quality_enhancers)

        logger.info(
            f"Basic fallback enhanced prompt: '{user_prompt}' -> '{enhanced_prompt}'"
        )
        return enhanced_prompt

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
        subject: str,
        style: str = "photorealistic",
        mood: str = "natural",
        details: str = "",
        aspect_ratio: str = "square",
        cfg_scale: float = 3.5,
        use_conversation_context: bool = True,
        config: ChatConfig = None,
        messages: List[Dict[str, Any]] = None,
    ) -> ImageGenerationResponse:
        """
        Generate an image based on user prompt with enhancement

        Args:
            user_prompt: Original user prompt
            subject: Main subject for the image
            style: Artistic style
            mood: Mood/atmosphere
            details: Additional details
            aspect_ratio: Aspect ratio for the image ("square", "portrait", or "landscape")
            cfg_scale: Guidance scale for image generation (1.5-4.5)
            use_conversation_context: Whether to use conversation context
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

        # Enhance the prompt using LLM-driven intelligent enhancement
        enhanced_prompt = self._enhance_prompt_with_llm(
            user_prompt, subject, style, mood, details, conversation_context, config
        )

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
                    "message": f"Successfully generated {aspect_ratio} ({width}x{height}) image with cfg_scale {cfg_scale} and enhanced prompt: {enhanced_prompt}",
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
                   Expected keys: 'user_prompt', 'subject', optionally 'style', 'mood', 'details', 'aspect_ratio', 'cfg_scale', 'use_conversation_context', 'messages', 'chain_context'

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
        aspect_ratio = params.get("aspect_ratio", "square")
        cfg_scale = params.get("cfg_scale", 3.5)
        use_conversation_context = params.get("use_conversation_context", True)
        messages = params.get("messages", None)

        logger.debug(
            f"run_with_dict called with user_prompt: '{user_prompt}', subject: '{subject}', aspect_ratio: {aspect_ratio}, cfg_scale: {cfg_scale}, use_context: {use_conversation_context}"
        )

        # Add more detailed logging to debug the issue
        logger.debug(f"Image generation parameters:")
        logger.debug(f"user_prompt: '{user_prompt}'")
        logger.debug(f"subject: '{subject}'")
        logger.debug(f"use_conversation_context: {use_conversation_context}")
        logger.debug(f"Number of messages: {len(messages) if messages else 0}")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self.generate_image_from_prompt(
            user_prompt,
            subject,
            style,
            mood,
            details,
            aspect_ratio,
            cfg_scale,
            use_conversation_context,
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
