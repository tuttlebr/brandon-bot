from dataclasses import dataclass

import streamlit as st
from utils.config import config


@dataclass
class ChatConfig:
    """Configuration wrapper for chat application - uses centralized config"""

    @property
    def assistant_avatar(self) -> str:
        return config.ui.ASSISTANT_AVATAR_PATH

    @property
    def user_avatar(self) -> str:
        return config.ui.USER_AVATAR_PATH

    @property
    def fast_llm_model_name(self) -> str:
        return config.env.FAST_LLM_MODEL_NAME

    @property
    def fast_llm_endpoint(self) -> str:
        return config.env.FAST_LLM_ENDPOINT

    @property
    def llm_model_name(self) -> str:
        return config.env.LLM_MODEL_NAME

    @property
    def llm_endpoint(self) -> str:
        return config.env.LLM_ENDPOINT

    @property
    def intelligent_llm_model_name(self) -> str:
        return config.env.INTELLIGENT_LLM_MODEL_NAME

    @property
    def intelligent_llm_endpoint(self) -> str:
        return config.env.INTELLIGENT_LLM_ENDPOINT

    @property
    def vlm_model_name(self) -> str:
        return config.env.VLM_MODEL_NAME

    @property
    def vlm_endpoint(self) -> str:
        return config.env.VLM_ENDPOINT or config.env.LLM_ENDPOINT

    @property
    def api_key(self) -> str:
        return config.env.NVIDIA_API_KEY

    @property
    def collection_name(self) -> str:
        return config.env.COLLECTION_NAME

    @property
    def image_endpoint(self) -> str:
        return config.env.IMAGE_ENDPOINT

    @classmethod
    def from_environment(cls) -> "ChatConfig":
        """Create configuration from environment variables using centralized config"""
        st.set_page_config(
            page_title=config.env.BOT_TITLE,
            page_icon=config.ui.ASSISTANT_AVATAR_PATH,
            initial_sidebar_state="collapsed",
        )
        return cls()
