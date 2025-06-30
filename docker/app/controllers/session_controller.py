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
        # Use atomic check-and-set to prevent race conditions in concurrent sessions
        if not getattr(st.session_state, "initialized", False):
            # Set initialization flag first to prevent multiple initializations
            st.session_state.initialized = True

            # Conditionally create a unique session ID if it doesn't exist
            if not hasattr(st.session_state, 'session_id') or not st.session_state.session_id:
                import random
                import time

                st.session_state.session_id = f"session_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

            st.session_state.fast_llm_model_name = self.config_obj.fast_llm_model_name
            st.session_state.llm_model_name = self.config_obj.llm_model_name
            st.session_state.intelligent_llm_model_name = self.config_obj.intelligent_llm_model_name
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state.current_page = config.ui.CURRENT_PAGE_DEFAULT
            st.session_state.processing = False

            # Initialize image storage with atomic check
            if not hasattr(st.session_state, 'generated_images'):
                st.session_state.generated_images = {}

            logging.info(f"Initialized new session state for session: {st.session_state.session_id}")

        # Clean up old data periodically (only for already initialized sessions)
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
        # Ensure session state is initialized before setting processing state
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()
        st.session_state.processing = processing

    def is_processing(self) -> bool:
        """
        Check if the app is currently processing

        Returns:
            True if processing, False otherwise
        """
        # Ensure session state is initialized before checking processing state
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()
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

    def get_model_name(self, model_type: str = "fast") -> str:
        """
        Safely get model name from session state with fallback to config

        Args:
            model_type: Type of model ('fast', 'llm', or 'intelligent')

        Returns:
            Model name string
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        # Map model types to session state keys and config fallbacks
        model_mapping = {
            "fast": ("fast_llm_model_name", self.config_obj.fast_llm_model_name),
            "llm": ("llm_model_name", self.config_obj.llm_model_name),
            "intelligent": ("intelligent_llm_model_name", self.config_obj.intelligent_llm_model_name),
        }

        if model_type not in model_mapping:
            logging.warning(f"Unknown model type '{model_type}', defaulting to fast model")
            model_type = "fast"

        session_key, config_fallback = model_mapping[model_type]
        model_name = st.session_state.get(session_key, config_fallback)

        if not model_name:
            logging.warning(f"No {model_type} model name found, using config fallback")
            model_name = config_fallback

        return model_name

    def get_messages(self):
        """
        Safely get messages from session state with initialization check

        Returns:
            List of messages from session state
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        return getattr(st.session_state, "messages", [])

    def add_message(self, role: str, content: any):
        """
        Safely add a message to session state

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        if not hasattr(st.session_state, "messages"):
            st.session_state.messages = []

        st.session_state.messages.append({"role": role, "content": content})

    def set_messages(self, messages: list):
        """
        Safely set messages in session state

        Args:
            messages: List of messages to set
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        st.session_state.messages = messages
