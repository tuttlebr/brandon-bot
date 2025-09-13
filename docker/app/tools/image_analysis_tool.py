"""
Image Analysis Tool

This tool analyzes uploaded images using vision-capable language models.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Type

from pydantic import Field
from tools.base import (
    BaseTool,
    BaseToolResponse,
    ExecutionMode,
    StreamingToolResponse,
)

logger = logging.getLogger(__name__)


class ImageAnalysisResponse(BaseToolResponse):
    """Response from image analysis tool"""

    filename: str = Field(description="Name of the image file")
    analysis: str = Field(description="Analysis result")
    question: str = Field(description="Question asked about the image")
    message: str = Field(description="Status message")
    direct_response: bool = Field(
        default=True, description="This provides a direct response to the user"
    )


class StreamingImageAnalysisResponse(StreamingToolResponse):
    """Streaming response from image analysis tool"""

    filename: str = Field(description="Name of the image file")
    question: str = Field(description="Question asked about the image")
    direct_response: bool = Field(
        default=True, description="This provides a direct response to the user"
    )


class ImageAnalysisTool(BaseTool):
    """Tool for analyzing uploaded images using vision-capable LLM models"""

    def __init__(self):
        super().__init__()
        self.name = "analyze_image"
        self.description = (
            "Analyze uploaded images to describe content or answer visual"
            " questions. Use when user uploads an image AND asks about it."
        )
        self.llm_type = "vlm"  # Use VLM model for image analysis
        self.execution_mode = ExecutionMode.AUTO  # Support both sync and async
        self.timeout = 256.0  # Image analysis can take time

    def _initialize_mvc(self):
        """Initialize MVC components"""
        # This tool doesn't need separate MVC components as it's simple
        self._controller = None
        self._view = None

    def get_definition(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": (
                                "The question to ask about the uploaded image"
                            ),
                        },
                        "but_why": {
                            "type": "integer",
                            "description": (
                                "An integer from 1-5 where a larger number"
                                " indicates confidence this is the right tool"
                                " to help the user."
                            ),
                        },
                    },
                    "required": ["question", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[ImageAnalysisResponse]:
        """Get the response type for this tool"""
        return ImageAnalysisResponse

    def execute(self, params: Dict[str, Any]) -> ImageAnalysisResponse:
        """Execute the tool with given parameters"""
        # Since this tool doesn't use MVC, override execute directly
        # Check if we should use async
        if self.execution_mode == ExecutionMode.AUTO:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Use async execution
                    future = asyncio.run_coroutine_threadsafe(
                        self._execute_async(params), loop
                    )
                    return future.result(timeout=self.timeout)
            except:
                pass

        # Fall back to sync execution
        return self._execute_sync(params)

    def _execute_sync(self, params: Dict[str, Any]) -> ImageAnalysisResponse:
        """Execute the tool synchronously"""
        question = params.get("question", "What do you see in this image?")

        # First check if image data was passed in params
        image_base64 = params.get("image_base64")
        filename = params.get("filename", "Unknown")

        # If not in params, check session state
        if not image_base64:
            import streamlit as st

            if (
                not hasattr(st.session_state, "current_image_base64")
                or not st.session_state.current_image_base64
            ):
                return ImageAnalysisResponse(
                    success=False,
                    filename="Unknown",
                    analysis="",
                    question=question,
                    message=(
                        "No image found. Please upload an image first using"
                        " the image uploader in the sidebar."
                    ),
                    error_message="No image data available",
                    error_code="NO_IMAGE_DATA",
                    direct_response=True,
                )

            # Get image data from session state
            image_base64 = st.session_state.current_image_base64
            filename = getattr(
                st.session_state, "current_image_filename", "Unknown"
            )

        logger.info(f"Analyzing image: {filename}")
        logger.debug(f"Image base64 length: {len(image_base64)}")

        # Log the actual size of the image being sent to VLM
        import base64

        try:
            image_bytes = base64.b64decode(image_base64)
            logger.debug(
                "Image size being sent to VLM:"
                f" {len(image_bytes) / 1024:.2f} KB"
            )
        except Exception as e:
            logger.error(f"Failed to decode base64 for size check: {e}")

        try:
            # Analyze the image
            analysis = self._analyze_image_with_llm(image_base64, question)

            return ImageAnalysisResponse(
                success=True,
                filename=filename,
                analysis=analysis,
                question=question,
                message=analysis,  # Direct response
                direct_response=True,
            )

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return ImageAnalysisResponse(
                success=False,
                filename=filename,
                analysis="",
                question=question,
                message=f"Failed to analyze image: {str(e)}",
                error_message=f"Failed to analyze image: {str(e)}",
                error_code="ANALYSIS_ERROR",
                direct_response=True,
            )

    async def _execute_async(
        self, params: Dict[str, Any]
    ) -> StreamingImageAnalysisResponse:
        """Execute the tool asynchronously with streaming"""
        question = params.get("question", "What do you see in this image?")

        # First check if image data was passed in params
        image_base64 = params.get("image_base64")
        filename = params.get("filename", "Unknown")

        # If not in params, check session state
        if not image_base64:
            import streamlit as st

            if (
                not hasattr(st.session_state, "current_image_base64")
                or not st.session_state.current_image_base64
            ):
                # Return error response with empty generator
                async def empty_generator():
                    yield (
                        "No image found. Please upload an image first using"
                        " the image uploader in the sidebar."
                    )

                return StreamingImageAnalysisResponse(
                    success=False,
                    filename="Unknown",
                    question=question,
                    content_generator=empty_generator(),
                    error_message="No image data available",
                    error_code="NO_IMAGE_DATA",
                    direct_response=True,
                )

            # Get image data from session state
            image_base64 = st.session_state.current_image_base64
            filename = getattr(
                st.session_state, "current_image_filename", "Unknown"
            )

        logger.info(f"Analyzing image: {filename}")
        logger.debug(f"Image base64 length: {len(image_base64)}")

        # Log the actual size of the image being sent to VLM
        import base64

        try:
            image_bytes = base64.b64decode(image_base64)
            logger.debug(
                "Image size being sent to VLM:"
                f" {len(image_bytes) / 1024:.2f} KB"
            )
        except Exception as e:
            logger.error(f"Failed to decode base64 for size check: {e}")

        try:
            # Create streaming generator for image analysis
            content_generator = self._analyze_image_with_llm_streaming(
                image_base64, question
            )

            return StreamingImageAnalysisResponse(
                success=True,
                filename=filename,
                question=question,
                content_generator=content_generator,
                direct_response=True,
            )

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")

            # Return error response with error message generator
            async def error_generator():
                yield f"Failed to analyze image: {str(e)}"

            return StreamingImageAnalysisResponse(
                success=False,
                filename=filename,
                question=question,
                content_generator=error_generator(),
                error_message=f"Failed to analyze image: {str(e)}",
                error_code="ANALYSIS_ERROR",
                direct_response=True,
            )

    def _analyze_image_with_llm(self, image_base64: str, question: str) -> str:
        """
        Analyze image using vision-capable LLM

        Args:
            image_base64: Base64 encoded image data
            question: Question about the image

        Returns:
            Analysis result as string
        """
        try:
            # Import here to avoid circular imports
            pass

            from models.chat_config import ChatConfig
            from openai import OpenAI
            from utils.text_processing import StreamingThinkTagFilter

            # Resize image using 12-tile constraint system
            processed_base64 = self._preprocess_image_for_vlm(image_base64)

            config_obj = ChatConfig.from_environment()

            # Use VLM endpoint and model
            client = OpenAI(
                api_key=config_obj.vlm_api_key,
                base_url=config_obj.vlm_endpoint,
            )

            model_name = config_obj.vlm_model_name

            logger.info(
                f"Using VLM model with streaming: {model_name} at endpoint:"
                f" {config_obj.vlm_endpoint}"
            )

            # For better results, make sure to put the image before any text part in the request body.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/png;base64,{processed_base64}"
                                )
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                },
            ]

            # Make streaming VLM call
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0,
                top_p=0.01,
                stream=True,
            )

            # Create think tag filter for streaming
            think_filter = StreamingThinkTagFilter()
            collected_response = ""

            # Process stream with think tag filtering
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    # Filter think tags from the chunk
                    filtered_content = think_filter.process_chunk(
                        chunk_content
                    )
                    if filtered_content:
                        collected_response += filtered_content

            # Get any remaining content from the filter
            final_content = think_filter.flush()
            if final_content:
                collected_response += final_content

            return collected_response.strip()

        except Exception as e:
            logger.error(f"LLM image analysis failed: {e}")
            raise Exception(f"Failed to analyze image with LLM: {str(e)}")

    async def _analyze_image_with_llm_streaming(
        self, image_base64: str, question: str
    ) -> AsyncGenerator[str, None]:
        """
        Analyze image using vision-capable LLM with true streaming

        Args:
            image_base64: Base64 encoded image data
            question: Question about the image

        Yields:
            Analysis response chunks as they arrive
        """
        try:
            # Import here to avoid circular imports
            pass

            from models.chat_config import ChatConfig
            from openai import AsyncOpenAI
            from utils.text_processing import StreamingThinkTagFilter

            # Resize image using 12-tile constraint system
            processed_base64 = await asyncio.to_thread(
                self._preprocess_image_for_vlm, image_base64
            )

            config_obj = ChatConfig.from_environment()

            # Use async VLM client
            client = AsyncOpenAI(
                api_key=config_obj.vlm_api_key,
                base_url=config_obj.vlm_endpoint,
            )

            model_name = config_obj.vlm_model_name

            logger.info(
                f"Using VLM model with streaming: {model_name} at endpoint:"
                f" {config_obj.vlm_endpoint}"
            )

            # For better results, make sure to put the image before any text part in the request body.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:image/png;base64,{processed_base64}"
                                )
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                },
            ]

            # Make streaming VLM call
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0,
                top_p=0.01,
                stream=True,  # Enable streaming
            )

            # Create think tag filter for streaming
            think_filter = StreamingThinkTagFilter()

            # Process stream with think tag filtering and yield chunks
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    # Filter think tags from the chunk
                    filtered_content = think_filter.process_chunk(
                        chunk_content
                    )
                    if filtered_content:
                        yield filtered_content

            # Yield any remaining content from the filter
            final_content = think_filter.flush()
            if final_content:
                yield final_content

        except Exception as e:
            logger.error(f"Streaming LLM image analysis failed: {e}")
            yield f"Error analyzing image: {str(e)}"

    def _preprocess_image_for_vlm(self, image_base64: str) -> str:
        """
        Preprocess image for VLM with 12-tile constraint system

        Args:
            image_base64: Base64 encoded image data

        Returns:
            Processed base64 image string
        """
        try:
            # Import here to avoid circular imports
            import base64
            from io import BytesIO
            from PIL import Image

            # Decode base64 to PIL Image
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(BytesIO(image_bytes))

            original_width, original_height = img.size
            original_aspect = original_width / original_height

            logger.info(
                "VLM preprocessing - Original image dimensions:"
                f" {original_width}x{original_height}, aspect ratio:"
                f" {original_aspect:.3f}"
            )

            # Find best tile configuration (w_tiles × h_tiles ≤ 12) that maintains closest aspect ratio
            # with maximum dimension constraint of 3072 pixels
            best_tiles = (1, 1)
            best_aspect_diff = float("inf")
            max_dimension = 3072
            tile_size = 512
            max_tiles_per_dimension = (
                max_dimension // tile_size
            )  # 6 tiles max per dimension

            for w_tiles in range(1, max_tiles_per_dimension + 1):
                for h_tiles in range(1, max_tiles_per_dimension + 1):
                    if w_tiles * h_tiles > 12:
                        break

                    tile_aspect = w_tiles / h_tiles
                    aspect_diff = abs(original_aspect - tile_aspect)

                    if aspect_diff < best_aspect_diff:
                        best_aspect_diff = aspect_diff
                        best_tiles = (w_tiles, h_tiles)

            # Calculate target dimensions
            target_width = best_tiles[0] * 512
            target_height = best_tiles[1] * 512

            # Ensure minimum 32x32
            target_width = max(target_width, 32)
            target_height = max(target_height, 32)

            # Check if resizing is needed
            if (
                original_width != target_width
                or original_height != target_height
            ):
                # Resize image
                resized_img = img.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )

                logger.info(
                    "Resized image for VLM from"
                    f" {original_width}x{original_height} to"
                    f" {target_width}x{target_height} (tiles:"
                    f" {best_tiles[0]}x{best_tiles[1]})"
                )
            else:
                resized_img = img
                logger.info(
                    "Image already at target size:"
                    f" {target_width}x{target_height}"
                )

            # Convert to RGB and encode (lossless PNG format)
            buffer = BytesIO()
            if resized_img.mode in ("RGBA", "LA", "P"):
                # Convert to RGB (no alpha channel support)
                rgb_img = Image.new("RGB", resized_img.size, (255, 255, 255))
                rgb_img.paste(
                    resized_img,
                    mask=(
                        resized_img.split()[-1]
                        if resized_img.mode == "RGBA"
                        else None
                    ),
                )
                resized_img = rgb_img

            resized_img.save(buffer, format="PNG")
            buffer.seek(0)
            resized_bytes = buffer.getvalue()
            processed_base64 = base64.b64encode(resized_bytes).decode("utf-8")

            logger.info(
                "VLM image size after processing:"
                f" {len(resized_bytes) / 1024:.2f} KB"
            )

            return processed_base64

        except Exception as e:
            logger.error(
                f"Failed to preprocess image for VLM, using original: {e}"
            )
            # Return original if preprocessing fails
            return image_base64


# Helper functions for backward compatibility
def get_image_analysis_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for image analysis

    Returns:
        Dict containing the OpenAI tool definition
    """
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("analyze_image", ImageAnalysisTool)

    # Get the tool instance and return its definition
    tool = get_tool("analyze_image")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get image analysis tool definition")
