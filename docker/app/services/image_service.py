from models.chat_config import ChatConfig


class ImageService:
    """Service for handling image generation operations"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the image service

        Args:
            config: Configuration for the image service
        """
        self.config = config
