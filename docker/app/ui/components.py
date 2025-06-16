import logging

import streamlit as st
from models.chat_config import ChatConfig
from models.chat_message import ChatMessage
from utils.image import base64_to_pil_image
from utils.split_context import extract_context_regex


class ChatHistoryComponent:
    """Component for displaying chat history with pagination"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the chat history component

        Args:
            config: Configuration for avatars and display settings
        """
        self.config = config

    def display_chat_history(self, messages: list, messages_per_page: int = 10):
        """
        Display the chat history with pagination

        Args:
            messages: List of messages to display
            messages_per_page: Number of messages to show per page
        """
        current_page = st.session_state.get("current_page", 0)

        # Filter out system messages for display
        display_messages = [m for m in messages if m["role"] != "system"]
        total_pages = max(1, len(display_messages) // messages_per_page + 1)

        start_idx = current_page * messages_per_page
        end_idx = start_idx + messages_per_page

        for message in display_messages[start_idx:end_idx]:
            with st.chat_message(
                message["role"],
                avatar=(self.config.user_avatar if message["role"] == "user" else self.config.assistant_avatar),
            ):
                chat_message = ChatMessage(message["role"], message["content"])

                # Check if this is an image message
                if chat_message.is_image_message():
                    image_data, image_caption = chat_message.get_image_data()
                    if image_data:
                        try:
                            # Convert base64 back to PIL Image for display
                            image = base64_to_pil_image(image_data)
                            if image:
                                st.image(image, caption=f"Generated image: {image_caption}", use_container_width=True)
                        except Exception as e:
                            logging.error(f"Error displaying image: {e}")
                            st.error("Could not display image")

                    # Display the text content
                    content = chat_message.get_display_content()
                    if content:
                        st.markdown(content, unsafe_allow_html=True)
                else:
                    # Regular text message
                    content = chat_message.get_display_content()
                    st.markdown(content, unsafe_allow_html=True)

        # Display pagination controls if needed
        if total_pages > 1:
            self._display_pagination_controls(current_page, total_pages)

    def _display_pagination_controls(self, current_page: int, total_pages: int):
        """
        Display pagination controls

        Args:
            current_page: Current page number
            total_pages: Total number of pages
        """
        cols = st.columns(3)

        if current_page > 0:
            if cols[0].button("Previous"):
                st.session_state.current_page -= 1
                st.rerun()

        cols[1].write(f"Page {current_page + 1} of {total_pages}")

        if current_page < total_pages - 1:
            if cols[2].button("Next"):
                st.session_state.current_page += 1
                st.rerun()

    def display_context_expander(self, context: str):
        """
        Display context information in an expandable section

        Args:
            context: Context information to display
        """
        if context:
            with st.expander("Expand for context"):
                st.markdown(
                    extract_context_regex(context).replace("$", "\\$").replace("\\${", "${"), unsafe_allow_html=True,
                )
