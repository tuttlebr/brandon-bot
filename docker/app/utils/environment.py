import logging
import os
from typing import Optional


class Config:
    """
    Singleton configuration class for the application.
    Provides centralized access to all environment variables.
    """

    _instance: Optional['Config'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration values from environment variables"""
        # Bot configuration
        self.BOT_TITLE = os.getenv("BOT_TITLE", "Nano")
        self.META_USER = os.getenv("META_USER", "Brandon")
        self.AUTH_USERNAME = os.getenv("AUTH_USERNAME", "Brandon")
        self.AUTH_KEY = os.getenv("AUTH_KEY", "Brandon")

        # Model configuration
        self.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
        self.LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")

        # API configuration
        self.NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "None")
        self.OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "openai")

        # Embedding configuration
        self.EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

        # Database configuration
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "milvus")
        self.PARTITION_NAME = os.getenv("PARTITION_NAME", "milvus")
        self.DEFAULT_DB = os.getenv("DEFAULT_DB", "milvus")

        # Reranker configuration
        self.RERANKER_ENDPOINT = os.getenv("RERANKER_ENDPOINT")
        self.RERANKER_MODEL = os.getenv("RERANKER_MODEL")

        # Image generation configuration
        self.IMAGE_ENDPOINT = os.getenv("IMAGE_ENDPOINT")

        # UI configuration
        self.USER_AVATAR = "/app/assets/user.png"
        self.ASSISTANT_AVATAR = "/app/assets/nvidia.png"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configuration instance - use this instead of legacy variables
config = Config()
