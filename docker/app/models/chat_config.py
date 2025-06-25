from dataclasses import dataclass

import streamlit as st
from utils.config import config


@dataclass
class ChatConfig:
    """Configuration for chat application"""

    # Bot configuration
    bot_title: str
    assistant_avatar: str
    user_avatar: str

    # Model configuration
    fast_llm_model_name: str
    fast_llm_endpoint: str
    llm_model_name: str
    llm_endpoint: str
    intelligent_llm_model_name: str
    intelligent_llm_endpoint: str
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
        """Create configuration from environment variables using centralized config"""
        st.set_page_config(
            page_title=config.env.BOT_TITLE,
            page_icon=config.ui.ASSISTANT_AVATAR_PATH,
            # layout="wide",
            initial_sidebar_state="expanded",
        )
        return cls(
            bot_title=config.env.BOT_TITLE,
            assistant_avatar=config.ui.ASSISTANT_AVATAR_PATH,
            user_avatar=config.ui.USER_AVATAR_PATH,
            fast_llm_model_name=config.env.FAST_LLM_MODEL_NAME,
            fast_llm_endpoint=config.env.FAST_LLM_ENDPOINT,
            llm_model_name=config.env.LLM_MODEL_NAME,
            llm_endpoint=config.env.LLM_ENDPOINT,
            embedding_endpoint=config.env.EMBEDDING_ENDPOINT,
            intelligent_llm_model_name=config.env.INTELLIGENT_LLM_MODEL_NAME,
            intelligent_llm_endpoint=config.env.INTELLIGENT_LLM_ENDPOINT,
            api_key=config.env.NVIDIA_API_KEY,
            embedding_model=config.env.EMBEDDING_MODEL,
            collection_name=config.env.COLLECTION_NAME,
            database_url=config.env.DATABASE_URL,
            default_db=config.env.DEFAULT_DB,
            image_endpoint=config.env.IMAGE_ENDPOINT,
        )
