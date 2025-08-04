import logging
from typing import Any, Dict, List

from models.chat_config import ChatConfig
from models.chat_message import ChatMessage
from utils.split_context import END_CONTEXT, START_CONTEXT, extract_context_regex
from utils.text_processing import strip_think_tags


class ChatService:
    """Service for handling chat processing operations"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the chat service

        Args:
            config: Configuration for the chat service
        """
        self.config = config
        self.verbose_messages = []

    def clean_chat_history_context(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clean up context information and thinking tags from all previous messages in chat history

        Args:
            messages: List of message dictionaries

        Returns:
            Cleaned list of message dictionaries
        """
        cleaned_messages = []

        for message in messages:
            if message["role"] != "system":  # Preserve system prompt as is
                # Check if this is an image message - if so, skip context cleaning
                if (
                    isinstance(message["content"], dict)
                    and message["content"].get("type") == "image"
                ):
                    # Image messages don't need context cleaning
                    cleaned_messages.append(message)
                    continue

                # Only clean string content
                if isinstance(message["content"], str):
                    # First remove context markers
                    cleaned_content = extract_context_regex(message["content"])

                    # Then remove any thinking tags that might be present
                    cleaned_content = strip_think_tags(cleaned_content)

                    cleaned_messages.append(
                        {"role": message["role"], "content": cleaned_content}
                    )
                else:
                    cleaned_messages.append(message)
            else:
                cleaned_messages.append(message)

        logging.debug("Cleaned context and thinking tags from previous chat history")
        return cleaned_messages

    def prepare_messages_for_api(
        self, messages: List[Dict[str, Any]], context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for API call, filtering out image data for LLM

        Args:
            messages: List of message dictionaries
            context: Context information to add to the prompt

        Returns:
            Prepared messages for API call
        """
        # Include all messages but filter out image data for LLM
        self.verbose_messages = []

        for msg in messages:
            chat_message = ChatMessage(msg["role"], msg["content"])

            # For image messages, only include the text content, not the image data
            if chat_message.is_image_message():
                # Only include the text confirmation message, not the image data
                text_content = chat_message.get_display_content()
                self.verbose_messages.append(
                    {"role": msg["role"], "content": text_content}
                )
            else:
                # Regular messages go through as normal
                self.verbose_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        # PDF context is now handled by LLM service via pdf_assistant tool calls
        # The tool response will be captured and displayed in the tool context expander

        # Add any explicitly provided context to the last user message if provided
        if context and self.verbose_messages:
            # Find the last user message
            for i in range(len(self.verbose_messages) - 1, -1, -1):
                if self.verbose_messages[i].get("role") == "user":
                    self.verbose_messages[i][
                        "content"
                    ] += f"{START_CONTEXT}{context}{END_CONTEXT}"
                    break

        return self.verbose_messages

    def drop_verbose_messages_context(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clean up context information from messages

        Args:
            messages: List of message dictionaries

        Returns:
            Cleaned list of message dictionaries
        """
        cleaned_messages = []

        for message in messages:
            # Handle different message content types
            if (
                isinstance(message["content"], dict)
                and message["content"].get("type") == "image"
            ):
                # Image messages don't need context extraction
                cleaned_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }  # Keep image content as-is
                )
            else:
                # Regular text messages get context cleaned
                cleaned_messages.append(
                    {
                        "role": message["role"],
                        "content": ChatMessage(
                            message["role"], extract_context_regex(message["content"])
                        ).get_display_content(),
                    }
                )

        return cleaned_messages
