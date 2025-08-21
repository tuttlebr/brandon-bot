import logging
import re
from typing import Any, Dict, List

import streamlit as st
from models.chat_config import ChatConfig
from services import ChatService
from utils.text_processing import sanitize_python_input, strip_think_tags


class MessageController:
    """Controller for handling message processing and validation"""

    def __init__(
        self,
        config_obj: ChatConfig,
        chat_service: ChatService,
        session_controller=None,
    ):
        """
        Initialize the message controller

        Args:
            config_obj: Application configuration
            chat_service: Chat service for message operations
            session_controller: Session controller for safe state management
                (optional)
        """
        self.config_obj = config_obj
        self.chat_service = chat_service
        self.session_controller = session_controller
        # Compile pattern once for performance
        self._toolcall_pattern = re.compile(
            r'<TOOLCALL(?:[-"\s])*\[.*?\]</TOOLCALL>',
            re.DOTALL | re.IGNORECASE,
        )

    def validate_prompt(self, prompt: str) -> tuple[bool, str]:
        """
        Enhanced validation for user prompts with Python syntax checking

        Args:
            prompt: User input prompt

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt or not prompt.strip():
            return False, "Input cannot be empty"

        # First check for tool call instructions
        if self.contains_tool_call_instructions(prompt):
            return False, "Tool call instructions are not allowed"

        return True, "Input is valid"

    def contains_tool_call_instructions(self, content: str) -> bool:
        """
        Check if content contains custom tool call instructions
        (optimized for speed)

        Args:
            content: The content to check

        Returns:
            True if content contains tool call instructions, False otherwise
        """
        if not isinstance(content, str):
            return False

        # Quick string check first - if no '<TOOLCALL' found, return early
        if '<TOOLCALL' not in content.upper():
            return False

        # Only do regex if the quick check passes
        return bool(self._toolcall_pattern.search(content))

    def clean_chat_history_of_tool_calls(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clean existing chat history to remove any messages containing
        tool call instructions

        Args:
            messages: List of messages to clean

        Returns:
            Cleaned list of messages
        """
        if not messages:
            return []

        original_count = len(messages)
        cleaned_messages = []

        for message in messages:
            content = message.get("content", "")

            # Keep system messages and non-string content as-is
            if message.get("role") == "system" or not isinstance(content, str):
                cleaned_messages.append(message)
                continue

            # Check if message contains tool call instructions
            if self.contains_tool_call_instructions(content):
                logging.warning(
                    f"Removing {message.get('role', 'unknown')} message "
                    f"with tool call instructions from chat history"
                )
                continue

            # Keep clean messages
            cleaned_messages.append(message)

        if len(cleaned_messages) != original_count:
            logging.info(
                f"Cleaned chat history: {original_count} -> "
                f"{len(cleaned_messages)} messages"
            )

        return cleaned_messages

    def safe_add_message_to_history(self, role: str, content: Any) -> bool:
        """
        Enhanced safe message addition with Python validation

        Args:
            role: The role of the message sender
            content: The content of the message (can be string, dict for
                images, etc.)

        Returns:
            True if message was added, False if rejected
        """
        # Handle different content types
        if isinstance(content, str):
            # Sanitize the content first
            sanitized_content = sanitize_python_input(content)

            if not sanitized_content:
                logging.warning(
                    f"Attempted to add empty {role} message to chat "
                    f"history, skipping"
                )
                return False

            # Check for tool call instructions (but allow tool responses)
            if role != "tool" and self.contains_tool_call_instructions(
                sanitized_content
            ):
                logging.warning(
                    f"Attempted to add {role} message with tool call "
                    f"instructions to chat history, skipping"
                )
                return False

            # Strip think tags before adding to history
            content = strip_think_tags(sanitized_content).strip()
        elif isinstance(content, dict):
            # Dict content (like image messages) - ensure it has
            # meaningful data
            if not content:
                logging.warning(
                    f"Attempted to add empty dict {role} message to "
                    f"chat history, skipping"
                )
                return False
        else:
            # Other content types - ensure they're truthy
            if not content:
                logging.warning(
                    f"Attempted to add empty {role} message to chat "
                    f"history, skipping"
                )
                return False

        # Add the validated message to history using session controller
        # if available
        if hasattr(self, 'session_controller') and self.session_controller:
            self.session_controller.add_message(role, content)
        else:
            # Fallback to direct access with safety check
            if not hasattr(st.session_state, "messages"):
                st.session_state.messages = []
            st.session_state.messages.append(
                {"role": role, "content": content}
            )
        logging.debug("Added %s message to chat history", role)
        return True

    def update_chat_history(self, text: str, role: str):
        """
        Update chat history with new response

        Args:
            text: The response text
            role: The role of the message sender
        """
        # Clean up message format before saving using session controller
        # if available
        if hasattr(self, 'session_controller') and self.session_controller:
            messages = self.session_controller.get_messages()
            cleaned_messages = self.chat_service.drop_verbose_messages_context(
                messages
            )
            self.session_controller.set_messages(cleaned_messages)
        else:
            # Fallback to direct access
            if hasattr(st.session_state, "messages"):
                st.session_state.messages = (
                    self.chat_service.drop_verbose_messages_context(
                        st.session_state.messages
                    )
                )

        # Use the safe method to add the message
        self.safe_add_message_to_history(role, text)

    def prepare_messages_for_processing(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages for LLM processing by cleaning and validating

        Args:
            messages: Raw messages from session state

        Returns:
            Cleaned and prepared messages
        """
        # Clean existing chat history of any tool call instructions
        cleaned_messages = self.clean_chat_history_of_tool_calls(messages)

        # Clean previous chat history from context
        cleaned_messages = self.chat_service.clean_chat_history_context(
            cleaned_messages
        )

        # Prepare messages for API
        return self.chat_service.prepare_messages_for_api(cleaned_messages)
