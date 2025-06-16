from dataclasses import dataclass

import streamlit as st
from utils.environment import Config


@dataclass
class ChatConfig:
    """Configuration for chat application"""

    # Bot configuration
    bot_title: str
    assistant_avatar: str
    user_avatar: str

    # Model configuration
    llm_model_name: str
    llm_endpoint: str
    embedding_endpoint: str
    api_key: str
    embedding_model: str

    # Database configuration
    collection_name: str
    database_url: str
    default_db: str

    # Image generation configuration
    image_endpoint: str

    @classmethod
    def from_environment(cls) -> "ChatConfig":
        """Create configuration from environment variables"""
        config = Config()
        st.set_page_config(
            page_title=config.BOT_TITLE,
            page_icon=config.ASSISTANT_AVATAR,
            # layout="wide",
            initial_sidebar_state="expanded",
        )
        return cls(
            bot_title=config.BOT_TITLE,
            assistant_avatar=config.ASSISTANT_AVATAR,
            user_avatar=config.USER_AVATAR,
            llm_model_name=config.LLM_MODEL_NAME,
            llm_endpoint=config.LLM_ENDPOINT,
            embedding_endpoint=config.EMBEDDING_ENDPOINT,
            api_key=config.NVIDIA_API_KEY,
            embedding_model=config.EMBEDDING_MODEL,
            collection_name=config.COLLECTION_NAME,
            database_url=config.DATABASE_URL,
            default_db=config.DEFAULT_DB,
            image_endpoint=config.IMAGE_ENDPOINT,
        )
