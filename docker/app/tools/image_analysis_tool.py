"""
Image Analysis Tool

This tool analyzes uploaded images using vision-capable language models.
"""

import logging
from typing import Any, Dict

from pydantic import Field
from tools.base import BaseTool, BaseToolResponse

logger = logging.getLogger(__name__)


class ImageAnalysisResponse(BaseToolResponse):
    """Response from image analysis tool"""

    success: bool = Field(description="Whether the analysis was successful")
    filename: str = Field(description="Name of the image file")
    analysis: str = Field(description="Analysis result")
    question: str = Field(description="Question asked about the image")
    message: str = Field(description="Status message")
    direct_response: bool = Field(
        default=True, description="This provides a direct response to the user"
    )


class ImageAnalysisTool(BaseTool):
    """Tool for analyzing uploaded images using vision-capable LLM models"""

    def __init__(self):
        super().__init__()
        self.name = "analyze_image"
        self.description = "ONLY use when explicitly asked to analyze, describe, or answer questions about an uploaded image. Analyzes uploaded images using vision-capable LLM models to answer questions about image content, describe what is visible, identify objects, or provide insights about visual elements. DO NOT use for PDF documents, text analysis, or general questions - use appropriate tools for those tasks."
        self.llm_type = "vlm"  # Use VLM model for image analysis

    def to_openai_format(self) -> Dict[str, Any]:
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
                            "description": "The question to ask about the uploaded image",
                        },
                        "but_why": {
                            "type": "string",
                            "description": "A single sentence explaining why this tool was selected for the query.",
                        },
                    },
                    "required": ["question", "but_why"],
                },
            },
        }

    def execute(self, params: Dict[str, Any]) -> ImageAnalysisResponse:
        """Execute the tool with given parameters"""
        question = params.get("question", "What do you see in this image?")

        # First check if image data was passed in params
        image_base64 = params.get("image_base64")
        filename = params.get("filename", "Unknown")

        # If not in params, check session state
        if not image_base64:
            import streamlit as st

            if (
                not hasattr(st.session_state, 'current_image_base64')
                or not st.session_state.current_image_base64
            ):
                return ImageAnalysisResponse(
                    success=False,
                    filename="Unknown",
                    analysis="",
                    question=question,
                    message="No image found. Please upload an image first using the image uploader in the sidebar.",
                    direct_response=True,
                )

            # Get image data from session state
            image_base64 = st.session_state.current_image_base64
            filename = getattr(st.session_state, 'current_image_filename', 'Unknown')

        logger.info(f"Analyzing image: {filename}")

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
            from models.chat_config import ChatConfig
            from openai import OpenAI

            config_obj = ChatConfig.from_environment()

            # Use VLM endpoint and model
            client = OpenAI(
                api_key=config_obj.api_key, base_url=config_obj.vlm_endpoint
            )

            model_name = config_obj.vlm_model_name

            logger.info(
                f"Using VLM model: {model_name} at endpoint: {config_obj.vlm_endpoint}"
            )

            # Create one-shot message with image and question
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

            # Make one-shot VLM call
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0,
                top_p=0.01,
                stream=False,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM image analysis failed: {e}")
            raise Exception(f"Failed to analyze image with LLM: {str(e)}")


# Create a global instance for backward compatibility
image_analysis_tool = ImageAnalysisTool()


def get_image_analysis_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    return image_analysis_tool.to_openai_format()


def execute_image_analysis_with_dict(params: Dict[str, Any]) -> ImageAnalysisResponse:
    """Execute image analysis with parameters as dictionary"""
    return image_analysis_tool.execute(params)
