import logging

import streamlit as st
from models.chat_config import ChatConfig
from models.chat_message import ChatMessage
from services.file_storage_service import FileStorageService
from utils.image import base64_to_pil_image
from utils.split_context import extract_context_regex
from utils.text_processing import strip_think_tags


class ChatHistoryComponent:
    """Component for displaying chat history with pagination"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the chat history component

        Args:
            config: Configuration for avatars and display settings
        """
        self.config = config
        self.file_storage = FileStorageService()

    def display_chat_history(self, messages: list, messages_per_page: int = 25):
        """
        Display the chat history with pagination

        Args:
            messages: List of messages to display
            messages_per_page: Number of messages to show per page
        """
        current_page = st.session_state.get("current_page", 0)

        # Filter out system messages and tool messages (PDF content is now automatically injected)
        display_messages = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "tool":
                # Skip tool messages - they're no longer needed for display
                continue
            else:
                display_messages.append(m)
        total_pages = max(1, len(display_messages) // messages_per_page + 1)

        start_idx = current_page * messages_per_page
        end_idx = start_idx + messages_per_page

        for i, message in enumerate(display_messages[start_idx:end_idx]):
            with st.chat_message(
                message["role"],
                avatar=(
                    self.config.user_avatar
                    if message["role"] == "user"
                    else self.config.assistant_avatar
                ),
            ):
                chat_message = ChatMessage(message["role"], message["content"])

                # Check if this is an image message
                if chat_message.is_image_message():
                    image_id, enhanced_prompt, original_prompt = (
                        chat_message.get_image_data()
                    )

                    # Retrieve image from file storage
                    if image_id:
                        try:
                            # Get image data from file storage
                            image_info = self.file_storage.get_image(image_id)

                            if image_info and 'image_data' in image_info:
                                image_data = image_info['image_data']

                                # Convert base64 back to PIL Image for display
                                image = base64_to_pil_image(image_data)
                                if image:
                                    st.image(
                                        image,
                                        caption=f"{enhanced_prompt}",
                                        use_container_width=True,
                                    )
                                else:
                                    logging.error(
                                        f"Failed to convert image data for image_id: {image_id}"
                                    )
                                    st.info(
                                        "🖼️ Image could not be displayed (conversion error)"
                                    )
                            else:
                                logging.warning(
                                    f"Image not found in storage: {image_id}"
                                )
                                st.info("🖼️ Image not available (may have been removed)")

                        except Exception as e:
                            logging.error(f"Error displaying image {image_id}: {e}")
                            st.info("🖼️ Image could not be displayed")
                    else:
                        # No image ID available
                        logging.warning("Image message without image_id")
                        st.info("🖼️ Image reference not available")

                    # Display the text content
                    content = chat_message.get_display_content()
                    if content:
                        st.markdown(content, unsafe_allow_html=True)
                else:
                    # Regular text message
                    content = chat_message.get_display_content()
                    st.markdown(content, unsafe_allow_html=True)

                # Display tool context if this is the last assistant message and context exists
                if (
                    message["role"] == "assistant"
                    and i == len(display_messages[start_idx:end_idx]) - 1
                    and hasattr(
                        st.session_state, 'last_tool_context'
                    )  # Last message in current page
                    and st.session_state.last_tool_context
                    and not st.session_state.get(
                        "processing", False
                    )  # Don't show during active processing
                ):
                    self.display_context_expander(st.session_state.last_tool_context)

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
            with st.expander(
                "📋 View Tool Data Sources (for verification)", expanded=False
            ):
                # Strip think tags before displaying
                cleaned_context = strip_think_tags(context)
                st.markdown(
                    extract_context_regex(cleaned_context)
                    .replace("$", "\\$")
                    .replace("\\${", "${"),
                    unsafe_allow_html=True,
                )
