"""
Context Generation Tool

This tool generates contextually modified images based on an input image
and text prompt.
"""

import base64
import io
import logging
import os
from typing import Any, Dict, Optional, Type

import requests
from openai import OpenAI
from pydantic import Field
from tools.base import BaseTool, BaseToolResponse, ExecutionMode

# Configure logger
logger = logging.getLogger(__name__)


class ContextGenerationResponse(BaseToolResponse):
    """Response from context generation tool"""

    image_data: Optional[str] = Field(
        None, description="Base64 encoded generated image data"
    )
    original_prompt: str = Field(description="Original text prompt")
    enhanced_prompt: Optional[str] = Field(
        None, description="Enhanced prompt (same as original for context gen)"
    )
    input_image_used: bool = Field(
        description="Whether an input image was provided"
    )
    direct_response: bool = Field(
        True, description="Whether this is a direct response"
    )
    result: Optional[str] = Field(
        None, description="JSON string representation of the response"
    )


class ContextGenerationTool(BaseTool):
    """Tool for generating contextually modified images"""

    def __init__(self):
        super().__init__()
        self.name = "context_generation"
        self.description = (
            "Generate or modify images based on an existing image and text "
            "prompt. Use when user wants to edit/transform an existing "
            "image or create variations. Requires an uploaded image. "
            "Supports OpenAI's image edit API when configured."
        )
        self.supported_contexts = ["image_generation", "image_editing"]
        self.execution_mode = ExecutionMode.SYNC
        self.timeout = 120.0  # Context generation can take time

    def _initialize_mvc(self):
        """Initialize MVC components"""
        # This tool doesn't need separate MVC components
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
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Verbatim, the user's original message requesting image generation"
                            ),
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": (
                                "Aspect ratio for output image. Use "
                                "'match_input_image' to maintain the same "
                                "aspect ratio as input, or specify 'square', "
                                "'portrait', or 'landscape'"
                            ),
                            "enum": ["match_input_image"],
                            "default": "match_input_image",
                        },
                        "steps": {
                            "type": "integer",
                            "description": (
                                "Number of generation steps (higher = better "
                                "quality but slower)"
                            ),
                            "default": 30,
                            "minimum": 25,
                            "maximum": 35,
                        },
                        "cfg_scale": {
                            "type": "number",
                            "description": (
                                "Guidance scale (1.5-4.5). Higher values "
                                "follow the prompt more closely"
                            ),
                            "default": 3.5,
                            "minimum": 2.5,
                            "maximum": 4.5,
                        },
                        "seed": {
                            "type": "integer",
                            "description": (
                                "Random seed for reproducible results. "
                                "Use 0 for random"
                            ),
                            "default": 42,
                        },
                        "but_why": {
                            "type": "integer",
                            "description": (
                                "An integer from 1-5 where a larger number "
                                "indicates confidence this is the right tool "
                                "to help the user."
                            ),
                        },
                    },
                    "required": ["prompt", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[ContextGenerationResponse]:
        """Get the response type for this tool"""
        return ContextGenerationResponse

    def _execute_sync(
        self, params: Dict[str, Any]
    ) -> ContextGenerationResponse:
        """Execute the tool synchronously"""
        prompt = params.get("prompt")
        aspect_ratio = params.get("aspect_ratio", "match_input_image")
        steps = params.get("steps", 30)
        cfg_scale = params.get("cfg_scale", 3.5)
        seed = params.get("seed", 0)

        # Get image from params or session state
        image_base64 = params.get("image_base64")
        if not image_base64:
            import streamlit as st

            if (
                hasattr(st.session_state, "current_image_base64")
                and st.session_state.current_image_base64
            ):
                image_base64 = st.session_state.current_image_base64
            else:
                response_dict = {
                    "success": False,
                    "original_prompt": prompt,
                    "enhanced_prompt": prompt,
                    "input_image_used": False,
                    "error_message": (
                        "No image found. Please upload an image first using "
                        "the image uploader in the sidebar."
                    ),
                    "error_code": "NO_IMAGE_DATA",
                    "direct_response": True,
                }
                import json

                return ContextGenerationResponse(
                    **response_dict, result=json.dumps(response_dict)
                )

        # Validate parameters
        steps = max(10, min(50, steps))
        cfg_scale = max(1.5, min(4.5, cfg_scale))

        # Get configuration
        endpoint = os.getenv("CONTEXT_GENERATION_ENDPOINT")
        api_key = os.getenv("CONTEXT_GENERATION_API_KEY")

        if not endpoint:
            response_dict = {
                "success": False,
                "original_prompt": prompt,
                "enhanced_prompt": prompt,
                "input_image_used": True,
                "error_message": (
                    "Context generation is not configured. Please set the "
                    "CONTEXT_GENERATION_ENDPOINT environment variable."
                ),
                "error_code": "CONFIGURATION_ERROR",
                "direct_response": True,
            }
            import json

            return ContextGenerationResponse(
                **response_dict, result=json.dumps(response_dict)
            )

        if not api_key:
            response_dict = {
                "success": False,
                "original_prompt": prompt,
                "enhanced_prompt": prompt,
                "input_image_used": True,
                "error_message": (
                    "Context generation API key is not configured. Please set "
                    "the CONTEXT_GENERATION_API_KEY environment variable."
                ),
                "error_code": "CONFIGURATION_ERROR",
                "direct_response": True,
            }
            import json

            return ContextGenerationResponse(
                **response_dict, result=json.dumps(response_dict)
            )

        try:
            # Check if using OpenAI API
            if "api.openai.com" in endpoint:
                logger.info("Using OpenAI API for image editing")
                logger.info("Editing image with prompt: '%s'", prompt)

                # Initialize OpenAI client with the API key
                client = OpenAI(api_key=api_key)

                # Convert base64 image to bytes for OpenAI API
                image_bytes = base64.b64decode(image_base64)
                image_file = io.BytesIO(image_bytes)
                image_file.name = (
                    "edit_image.png"  # OpenAI requires a filename
                )

                # Use OpenAI's image edit API
                response = client.images.edit(
                    model="gpt-image-1",
                    image=image_file,
                    prompt=prompt,
                    n=1,
                    input_fidelity="high",
                    quality="high",
                )

                # Get the base64 data from response
                if response.data and len(response.data) > 0:
                    generated_image_data = response.data[0].b64_json
                    logger.info("Image edited successfully via OpenAI API")

                    # Create successful response
                    result_dict = {
                        "success": True,
                        "original_prompt": prompt,
                        "enhanced_prompt": prompt,
                        "input_image_used": True,
                        "aspect_ratio": aspect_ratio,
                        "steps": steps,
                        "cfg_scale": cfg_scale,
                        "seed": seed,
                        "direct_response": True,
                        "message": (
                            f"Successfully edited image with prompt: {prompt}"
                        ),
                    }

                    import json

                    return ContextGenerationResponse(
                        success=True,
                        image_data=generated_image_data,
                        original_prompt=prompt,
                        enhanced_prompt=prompt,
                        input_image_used=True,
                        direct_response=True,
                        result=json.dumps(result_dict),
                    )
                else:
                    raise ValueError("No image data in OpenAI response")

            else:
                # Use existing custom API endpoint
                # Prepare request
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                }

                payload = {
                    "prompt": prompt,
                    "image": f"data:image/png;base64,{image_base64}",
                    "aspect_ratio": aspect_ratio,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                }

                logger.info(
                    "Sending context generation request: prompt='%s', "
                    "aspect_ratio=%s, steps=%s, cfg_scale=%s, seed=%s",
                    prompt,
                    aspect_ratio,
                    steps,
                    cfg_scale,
                    seed,
                )

                # Make request
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                response_body = response.json()
                logger.info("Context generation completed successfully")

                # Log response structure for debugging
                if isinstance(response_body, dict):
                    sample = str(response_body)[:200]
                    if len(str(response_body)) > 200:
                        sample += "..."
                    logger.info(
                        "API response structure - keys: %s, sample: %s",
                        list(response_body.keys()),
                        sample,
                    )
                else:
                    sample = str(response_body)[:100]
                    if len(str(response_body)) > 100:
                        sample += "..."
                    logger.info(
                        "API response type: %s, sample: %s",
                        type(response_body).__name__,
                        sample,
                    )

                # Extract generated image from response
                # Assuming the response contains base64 encoded image data
                generated_image_data = None
                if isinstance(response_body, dict):
                    # Try common response field names
                    field_names = [
                        "image",
                        "data",
                        "result",
                        "output",
                        "artifacts",
                    ]
                    for field in field_names:
                        if field in response_body:
                            field_value = response_body[field]

                            # Handle artifacts array structure
                            # (common in image generation APIs)
                            if (
                                field == "artifacts"
                                and isinstance(field_value, list)
                                and len(field_value) > 0
                            ):
                                artifact = field_value[0]
                                if (
                                    isinstance(artifact, dict)
                                    and "base64" in artifact
                                ):
                                    generated_image_data = artifact["base64"]
                                    logger.info(
                                        "Found image data in artifacts[0].base64"
                                    )
                                    break
                            else:
                                generated_image_data = field_value
                                logger.info(
                                    "Found image data in field: %s", field
                                )
                                break

                if not generated_image_data:
                    # If response is a string, assume it's the image data
                    if isinstance(response_body, str):
                        generated_image_data = response_body
                        logger.info(
                            "Response is a string, using as image data"
                        )
                    else:
                        # Log available fields for debugging
                        if isinstance(response_body, dict):
                            available_fields = list(response_body.keys())
                        else:
                            available_fields = "N/A"
                        raise ValueError(
                            f"Could not find image data in response. "
                            f"Available fields: {available_fields}"
                        )

                # Clean up base64 data if needed
                if generated_image_data.startswith("data:image"):
                    # Extract base64 part from data URL
                    generated_image_data = generated_image_data.split(",", 1)[
                        1
                    ]

                # Create successful response
                result_dict = {
                    "success": True,
                    "original_prompt": prompt,
                    "enhanced_prompt": prompt,
                    "input_image_used": True,
                    "aspect_ratio": aspect_ratio,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "direct_response": True,
                    "message": (
                        f"Successfully generated image with prompt: {prompt}"
                    ),
                }

                import json

                # For consistency with image generation tool,
                # use prompt as enhanced_prompt
                return ContextGenerationResponse(
                    success=True,
                    image_data=generated_image_data,
                    original_prompt=prompt,
                    enhanced_prompt=prompt,  # For caption display
                    input_image_used=True,
                    direct_response=True,
                    result=json.dumps(result_dict),
                )

        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error in context generation: %s", e)
            error_message = f"Context generation API error: {e}"
            if e.response:
                try:
                    error_detail = e.response.json()
                    error_message = (
                        f"Context generation API error: "
                        f"{error_detail.get('error', str(e))}"
                    )
                except Exception:
                    pass

            response_dict = {
                "success": False,
                "original_prompt": prompt,
                "enhanced_prompt": prompt,
                "input_image_used": True,
                "error_message": error_message,
                "error_code": "API_ERROR",
                "direct_response": True,
            }
            import json

            return ContextGenerationResponse(
                **response_dict, result=json.dumps(response_dict)
            )

        except Exception as e:
            logger.error("Error in context generation: %s", e)
            response_dict = {
                "success": False,
                "original_prompt": prompt,
                "enhanced_prompt": prompt,
                "input_image_used": True,
                "error_message": f"Context generation failed: {str(e)}",
                "error_code": "GENERATION_ERROR",
                "direct_response": True,
            }
            import json

            return ContextGenerationResponse(
                **response_dict, result=json.dumps(response_dict)
            )

    def execute(self, params: Dict[str, Any]) -> ContextGenerationResponse:
        """Execute the tool with given parameters"""
        return self._execute_sync(params)


# Helper function for backward compatibility
def get_context_generation_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("context_generation", ContextGenerationTool)

    # Get the tool instance and return its definition
    tool = get_tool("context_generation")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get context generation tool definition")
