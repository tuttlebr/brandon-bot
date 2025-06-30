import json
import logging
import re
from typing import Any, Dict, List

import streamlit as st
from models.chat_config import ChatConfig
from services import ChatService
from utils.config import config


class MessageController:
    """Controller for handling message processing and validation"""

    def __init__(self, config_obj: ChatConfig, chat_service: ChatService, session_controller=None):
        """
        Initialize the message controller

        Args:
            config_obj: Application configuration
            chat_service: Chat service for message operations
            session_controller: Session controller for safe state management (optional)
        """
        self.config_obj = config_obj
        self.chat_service = chat_service
        self.session_controller = session_controller
        # Compile pattern once for performance
        self._toolcall_pattern = re.compile(r'<TOOLCALL(?:[-"\s])*\[.*?\]</TOOLCALL>', re.DOTALL | re.IGNORECASE)

    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate user prompt for safety and correctness

        Args:
            prompt: User input prompt

        Returns:
            True if valid, False if contains issues
        """
        if not prompt or not prompt.strip():
            return False

        return not self.contains_tool_call_instructions(prompt)

    def contains_tool_call_instructions(self, content: str) -> bool:
        """
        Check if content contains custom tool call instructions (optimized for speed)

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

    def clean_chat_history_of_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean existing chat history to remove any messages containing tool call instructions

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
                    f"Removing {message.get('role', 'unknown')} message with tool call instructions from chat history"
                )
                continue

            # Keep clean messages
            cleaned_messages.append(message)

        if len(cleaned_messages) != original_count:
            logging.info(f"Cleaned chat history: {original_count} -> {len(cleaned_messages)} messages")

        return cleaned_messages

    def safe_add_message_to_history(self, role: str, content: Any) -> bool:
        """
        Safely add a message to chat history with validation

        Args:
            role: The role of the message sender
            content: The content of the message (can be string, dict for images, etc.)

        Returns:
            True if message was added, False if rejected
        """
        # Handle different content types
        if isinstance(content, str):
            # String content - validate it's not empty and doesn't contain tool calls
            if not content or not content.strip():
                logging.warning(f"Attempted to add empty {role} message to chat history, skipping")
                return False

            # Check for tool call instructions
            if self.contains_tool_call_instructions(content):
                logging.warning(
                    f"Attempted to add {role} message with tool call instructions to chat history, skipping"
                )
                return False

            content = content.strip()
        elif isinstance(content, dict):
            # Dict content (like image messages) - ensure it has meaningful data
            if not content:
                logging.warning(f"Attempted to add empty dict {role} message to chat history, skipping")
                return False
        else:
            # Other content types - ensure they're truthy
            if not content:
                logging.warning(f"Attempted to add empty {role} message to chat history, skipping")
                return False

        # Add the validated message to history using session controller if available
        if hasattr(self, 'session_controller'):
            self.session_controller.add_message(role, content)
        else:
            # Fallback to direct access with safety check
            if not hasattr(st.session_state, "messages"):
                st.session_state.messages = []
            st.session_state.messages.append({"role": role, "content": content})
        logging.debug(f"Added {role} message to chat history")
        return True

    def update_chat_history(self, text: str, role: str):
        """
        Update chat history with new response

        Args:
            text: The response text
            role: The role of the message sender
        """
        # Clean up message format before saving using session controller if available
        if hasattr(self, 'session_controller') and self.session_controller:
            messages = self.session_controller.get_messages()
            cleaned_messages = self.chat_service.drop_verbose_messages_context(messages)
            self.session_controller.set_messages(cleaned_messages)
        else:
            # Fallback to direct access
            if hasattr(st.session_state, "messages"):
                st.session_state.messages = self.chat_service.drop_verbose_messages_context(st.session_state.messages)

        # Use the safe method to add the message
        self.safe_add_message_to_history(role, text)

    def truncate_long_prompt(self, prompt: str, max_length: int = None) -> str:
        """
        Truncate overly long prompts for display purposes

        Args:
            prompt: Original prompt
            max_length: Maximum allowed length (uses config default if None)

        Returns:
            Truncated prompt with ellipsis if needed
        """
        max_length = max_length or config.ui.MAX_PROMPT_DISPLAY_LENGTH
        if len(prompt) > max_length:
            return prompt[:max_length] + "..."
        return prompt

    def prepare_messages_for_processing(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        cleaned_messages = self.chat_service.clean_chat_history_context(cleaned_messages)

        # Inject PDF data from session state if available
        cleaned_messages = self._inject_pdf_data_if_available(cleaned_messages)

        # Prepare messages for API
        return self.chat_service.prepare_messages_for_api(cleaned_messages)

    def _inject_pdf_data_if_available(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Inject PDF data from session state into messages for tool access

        Args:
            messages: List of messages

        Returns:
            Messages with PDF data injected if available
        """
        try:
            if self.session_controller and self.session_controller.has_pdf_documents():
                # Get the latest PDF from session state
                pdf_data = self.session_controller.get_latest_pdf_document()
                if pdf_data:
                    # Prepare PDF data for injection
                    injection_data = {
                        "type": "pdf_data",
                        "tool_name": "process_pdf_document",
                        "filename": pdf_data.get('filename', 'Unknown'),
                        "total_pages": pdf_data.get('total_pages', 0),
                        "pages": pdf_data.get('pages', []),
                        "status": pdf_data.get('status', 'success'),
                    }

                    # Include summarization data if available
                    if pdf_data.get('summarization_complete', False):
                        injection_data['summarization_complete'] = True
                        injection_data['document_summary'] = pdf_data.get('document_summary', '')
                        injection_data['page_summaries'] = pdf_data.get('page_summaries', [])
                        logging.info(f"Including document summary for: {pdf_data.get('filename', 'Unknown')}")

                    # Add a special system message with PDF data for tools to access
                    pdf_message = {"role": "system", "content": json.dumps(injection_data)}
                    # Insert after the main system message
                    result = messages[:1] + [pdf_message] + messages[1:]
                    logging.info(f"Injected PDF data for: {pdf_data.get('filename', 'Unknown')}")
                    return result
        except Exception as e:
            logging.error(f"Error injecting PDF data: {e}")

        return messages
