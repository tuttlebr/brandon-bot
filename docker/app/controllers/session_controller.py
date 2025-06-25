import logging
from typing import Any, Dict

import streamlit as st
from models.chat_config import ChatConfig
from utils.config import config
from utils.system_prompt import SYSTEM_PROMPT


class SessionController:
    """Controller for managing Streamlit session state and cleanup operations"""

    def __init__(self, config_obj: ChatConfig):
        """
        Initialize the session controller

        Args:
            config_obj: Application configuration
        """
        self.config_obj = config_obj

    def initialize_session_state(self):
        """Initialize Streamlit session state with default values"""
        if not hasattr(st.session_state, "initialized"):
            st.session_state.initialized = True
            st.session_state.fast_llm_model_name = self.config_obj.fast_llm_model_name
            st.session_state.llm_model_name = self.config_obj.llm_model_name
            st.session_state.intelligent_llm_model_name = self.config_obj.intelligent_llm_model_name
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state.current_page = config.ui.CURRENT_PAGE_DEFAULT
            st.session_state.processing = False

            # Initialize image storage
            if 'generated_images' not in st.session_state:
                st.session_state.generated_images = {}

        # Clean up old data periodically
        self.cleanup_old_images()

    def cleanup_old_images(self, max_images: int = None):
        """
        Clean up old image data from session state to prevent memory issues

        Args:
            max_images: Maximum number of images to keep in session state
        """
        max_images = max_images or config.session.MAX_IMAGES_IN_SESSION

        if hasattr(st.session_state, 'generated_images') and st.session_state.generated_images:
            current_count = len(st.session_state.generated_images)

            if current_count > max_images:
                # Sort by image_id (which includes timestamp) and keep the most recent ones
                sorted_images = sorted(st.session_state.generated_images.items(), key=lambda x: x[0], reverse=True)

                # Keep only the most recent max_images
                st.session_state.generated_images = dict(sorted_images[:max_images])

                removed_count = current_count - max_images
                logging.info(
                    f"Cleaned up {removed_count} old images from session state. "
                    f"Kept {max_images} most recent images."
                )

    def set_processing_state(self, processing: bool):
        """
        Set the processing state to prevent concurrent operations

        Args:
            processing: Whether the app is currently processing
        """
        st.session_state.processing = processing

    def is_processing(self) -> bool:
        """
        Check if the app is currently processing

        Returns:
            True if processing, False otherwise
        """
        return st.session_state.get("processing", False)

    def store_tool_context(self, context: str):
        """
        Store tool context in session state

        Args:
            context: Tool context to store
        """
        st.session_state.last_tool_context = context

    def clear_tool_context(self):
        """Clear previous tool context from session state"""
        if hasattr(st.session_state, 'last_tool_context'):
            st.session_state.last_tool_context = None

    def store_generated_image(self, image_data: str, enhanced_prompt: str, original_prompt: str) -> str:
        """
        Store generated image data and return image ID

        Args:
            image_data: Base64 encoded image data
            enhanced_prompt: Enhanced prompt used for generation
            original_prompt: Original user prompt

        Returns:
            Unique image ID for the stored image
        """
        import time

        # Initialize session state for storing image data if not exists
        if 'generated_images' not in st.session_state:
            st.session_state.generated_images = {}

        # Generate a unique ID for this image using configured prefix
        image_id = f"{config.session.IMAGE_ID_PREFIX}{int(time.time() * 1000)}"

        # Store image data in session state for visual persistence
        st.session_state.generated_images[image_id] = {
            'image_data': image_data,
            'enhanced_prompt': enhanced_prompt,
            'original_prompt': original_prompt,
        }

        return image_id
