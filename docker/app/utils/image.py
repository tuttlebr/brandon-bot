import base64
import logging
import re
from io import BytesIO
from typing import Optional

# Third-party imports
import requests
from PIL import Image
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PROMPT = "A simple coffee shop interior"
DEFAULT_MODE = "base"
DEFAULT_CFG_SCALE = 3.5
DEFAULT_WIDTH = 1204
DEFAULT_HEIGHT = 1204
DEFAULT_SEED = 42
DEFAULT_STEPS = 42

BASE64_PREFIX_PATTERN = r"^data:image/.+;base64,"


# Allowed modes for image generation
ALLOWED_MODES = ["base"]

# HTTP constants
HTTP_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""


class ImageProcessingRequest(BaseModel):
    """
    Model for image processing API requests.

    This Pydantic model defines the structure for requests to image
    generation and processing APIs, with appropriate default values.
    """

    prompt: Optional[str] = DEFAULT_PROMPT
    mode: Optional[str] = DEFAULT_MODE
    cfg_scale: Optional[float] = DEFAULT_CFG_SCALE
    width: Optional[int] = DEFAULT_WIDTH
    height: Optional[int] = DEFAULT_HEIGHT
    image: Optional[str | Image.Image] = None
    preprocess_image: bool = True
    seed: Optional[int] = DEFAULT_SEED
    steps: Optional[int] = DEFAULT_STEPS
    disable_safety_checker: Optional[bool] = True

    class Config:  # dead: disable
        """Configuration for the ImageProcessingRequest model."""

        arbitrary_types_allowed = True  # dead: disable


def generate_image(
    invoke_url: str,
    image_b64: Optional[str] = None,
    prompt: Optional[str] = DEFAULT_PROMPT,
    mode: str = DEFAULT_MODE,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    preprocess_image: bool = True,
    seed: int = DEFAULT_SEED,
    steps: int = DEFAULT_STEPS,
    disable_safety_checker: bool = True,
    return_bytes_io: bool = True,
) -> Image.Image:
    """
    Send a request to an image generation API using the requests library and Pydantic model.

    Args:
        invoke_url: The URL endpoint to invoke
        image_b64: Base64 encoded image string (optional)
        prompt: Text prompt describing the desired image
        mode: The generation mode (e.g., "base", "depth")
        cfg_scale: Classifier free guidance scale
        width: Image width in pixels
        height: Image height in pixels
        preprocess_image: Whether to preprocess the input image
        seed: Random seed for reproducibility
        steps: Number of diffusion steps
        disable_safety_checker: Whether to disable safety checker

    Returns:
        PIL Image object created from the API response

    Raises:
        requests.RequestException: If the request fails
        ImageProcessingError: If there's an issue with processing parameters
    """
    try:
        # Create a dictionary of parameters
        params = {
            "prompt": prompt,
            "mode": mode,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "preprocess_image": preprocess_image,
            "seed": seed,
            "steps": steps,
            "disable_safety_checker": disable_safety_checker,
        }

        # Only add the image parameter if mode is not "base" and image_b64 is provided
        if mode in ALLOWED_MODES and image_b64 is not None:
            params["image"] = image_b64
        elif mode == "redux" and image_b64 is not None:
            params.pop("prompt")
            params["image"] = image_b64

        logger.debug(f"Params: {params}")
        # Create the request object using the Pydantic model
        request_data = ImageProcessingRequest(**params)

        # Make the request using the model's json method
        response = requests.post(
            invoke_url, json=request_data.model_dump(), headers=HTTP_HEADERS
        )

        # Raise HTTP errors
        response.raise_for_status()
        if return_bytes_io:
            return base64_to_pil_image(
                response.json()["artifacts"][0]["base64"]
            )
        else:
            return response.json()["artifacts"][0]["base64"]

    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        logger.error(response.json())
        # raise
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise ImageProcessingError(
            f"Failed to generate image: {str(e)}"
        ) from e


def base64_to_pil_image(base64_str: str) -> Optional[Image.Image]:
    """
    Convert a base64 encoded image string to a PIL Image.

    Args:
        base64_str: Base64 encoded image string, with or without data URI prefix

    Returns:
        The decoded PIL Image object or None if conversion fails

    Raises:
        ImageProcessingError: If conversion fails and error handling is not set to return None
    """
    try:
        # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
        if "data:" in base64_str and ";base64," in base64_str:
            base64_str = re.sub(BASE64_PREFIX_PATTERN, "", base64_str)

        # Decode the base64 string to binary
        image_data = base64.b64decode(base64_str)

        # Create a BytesIO object from the binary data
        image_buffer = BytesIO(image_data)

        # Open the BytesIO object as a PIL Image
        image = Image.open(image_buffer)

        # Force load the image to catch potential issues early
        image.load()

        return image
    except Exception as e:
        logger.error(f"Error converting base64 to PIL Image: {str(e)}")
        return None
