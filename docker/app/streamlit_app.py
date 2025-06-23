import asyncio
import logging
from typing import Any, Dict, Generator, List

import streamlit as st
from models import ChatConfig
from services import ChatService, ImageService, LLMService
from ui import ChatHistoryComponent, apply_custom_styles, get_typing_indicator_html
from utils.image import pil_image_to_base64
from utils.system_prompt import SYSTEM_PROMPT, TOOL_PROMPT, greeting_prompt


class StreamlitChatApp:
    """Main Streamlit chat application that orchestrates UI and services"""

    def __init__(self):
        """Initialize the Streamlit chat application"""
        # Initialize configuration
        self.config = ChatConfig.from_environment()

        # Apply custom styling and setup UI
        apply_custom_styles()
        st.markdown(
            f'<h1 style="color: #76b900; text-align: center;">{greeting_prompt()}</h1>', unsafe_allow_html=True,
        )

        # Initialize services
        self.chat_service = ChatService(self.config)
        self.image_service = ImageService(self.config)
        self.llm_service = LLMService(self.config)

        # Initialize UI components
        self.chat_history_component = ChatHistoryComponent(self.config)

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state with default values"""
        if not hasattr(st.session_state, "initialized"):
            st.session_state.initialized = True
            st.session_state.fast_llm_model_name = self.config.fast_llm_model_name
            st.session_state.llm_model_name = self.config.llm_model_name
            st.session_state.intelligent_llm_model_name = self.config.intelligent_llm_model_name
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state.current_page = 0

    def display_chat_history(self):
        """Display the chat history using the chat history component"""
        self.chat_history_component.display_chat_history(st.session_state.messages)

    def display_tool_context_expander(self, context: str):
        """
        Display tool context in an expandable section for user verification

        Args:
            context: Tool response context to display
        """
        if context:
            self.chat_history_component.display_context_expander(context)

    def _clean_chat_history_of_tool_calls(self):
        """
        Clean existing chat history to remove any messages containing tool call instructions
        """
        if not hasattr(st.session_state, 'messages') or not st.session_state.messages:
            return

        original_count = len(st.session_state.messages)
        cleaned_messages = []

        for message in st.session_state.messages:
            content = message.get("content", "")

            # Keep system messages and non-string content as-is
            if message.get("role") == "system" or not isinstance(content, str):
                cleaned_messages.append(message)
                continue

            # Check if message contains tool call instructions
            if self._contains_tool_call_instructions(content):
                logging.warning(
                    f"Removing {message.get('role', 'unknown')} message with tool call instructions from chat history"
                )
                continue

            # Keep clean messages
            cleaned_messages.append(message)

        if len(cleaned_messages) != original_count:
            st.session_state.messages = cleaned_messages
            logging.info(f"Cleaned chat history: {original_count} -> {len(cleaned_messages)} messages")

    def process_prompt(self, prompt: str):
        """
        Process user prompt and generate response

        Args:
            prompt: The user's text prompt
        """
        # Clean the prompt
        prompt = prompt.strip()

        # Clear any previous tool context when processing a new user message
        if hasattr(st.session_state, 'last_tool_context'):
            st.session_state.last_tool_context = None

        # Clear previous tool responses from LLM service
        if hasattr(self.llm_service, 'last_tool_responses'):
            self.llm_service.last_tool_responses = []

        # Validate that the prompt doesn't contain tool call instructions (shouldn't happen from user input)
        if self._contains_tool_call_instructions(prompt):
            logging.error("User prompt contains tool call instructions - this should not happen")
            st.error("Invalid input detected. Please try again with a different message.")
            st.session_state.processing = False
            return

        # Clean existing chat history of any tool call instructions
        self._clean_chat_history_of_tool_calls()

        # Clean previous chat history from context
        st.session_state.messages = self.chat_service.clean_chat_history_context(st.session_state.messages)

        # Display user message immediately in the UI
        with st.chat_message("user", avatar=self.config.user_avatar):
            st.markdown(prompt[:2048] + "..." if len(prompt) > 2048 else prompt)

        # Add user message to chat history using safe method
        self._safe_add_message_to_history("user", prompt)

        # Check if this is an image generation request
        if self.image_service.detect_image_generation_request(prompt):
            self._handle_image_generation(prompt)
            return

        try:
            prepared_messages = self.chat_service.prepare_messages_for_api(st.session_state.messages)

            # Generate and display response
            self._generate_and_display_response(prepared_messages)

        except Exception as e:
            logging.error(f"Error processing prompt: {e}")
            st.error("An error occurred while processing your request. Please try again.")
            st.session_state.processing = False

    def _extract_tool_context_from_llm_responses(self) -> str:
        """
        Extract context from the LLM service's last tool responses

        Returns:
            Formatted context string from tool responses, empty if none found
        """
        if not hasattr(self.llm_service, 'last_tool_responses') or not self.llm_service.last_tool_responses:
            logging.debug("No tool responses found in LLM service")
            return ""

        logging.debug(f"Found {len(self.llm_service.last_tool_responses)} tool responses to process")

        tool_contexts = []

        for tool_response in self.llm_service.last_tool_responses:
            logging.debug(f"Processing tool response: {tool_response}")
            if tool_response.get("role") == "tool":
                try:
                    tool_content = tool_response.get("content", "")
                    logging.debug(f"Tool content type: {type(tool_content)}, length: {len(str(tool_content))}")
                    if isinstance(tool_content, str):
                        # Try to parse as JSON to extract formatted results
                        import json

                        try:
                            tool_data = json.loads(tool_content)
                            if isinstance(tool_data, dict):
                                # Check for formatted_results first (preferred format)
                                if "formatted_results" in tool_data:
                                    formatted_results = tool_data["formatted_results"]
                                    if formatted_results and formatted_results.strip():
                                        tool_contexts.append(f"**Tool Response Data:**\n{formatted_results}")
                                        logging.info(f"Found formatted_results from tool response")

                                # Check for other common tool response formats
                                elif "results" in tool_data:
                                    results = tool_data["results"]
                                    if isinstance(results, list) and results:
                                        tool_contexts.append(f"**Tool found {len(results)} results**")
                                        logging.info(f"Found {len(results)} results from tool response")
                                    elif isinstance(results, str) and results.strip():
                                        tool_contexts.append(f"**Tool Response:**\n{results}")
                                        logging.info(f"Found string results from tool response")

                                # Handle weather tool response format
                                elif "location" in tool_data and "current" in tool_data:
                                    location = tool_data.get("location", "Unknown")
                                    current = tool_data.get("current", {})
                                    temp = current.get("temperature", "N/A")
                                    tool_contexts.append(f"**Weather data for {location}** (Current: {temp}°F)")
                                    logging.info(f"Found weather tool response for {location}")

                                # Generic fallback for other structured data
                                else:
                                    # Try to create a summary of the tool response
                                    summary_parts = []
                                    if "query" in tool_data:
                                        summary_parts.append(f"Query: {tool_data['query']}")
                                    if "total_results" in tool_data:
                                        summary_parts.append(f"Results: {tool_data['total_results']}")
                                    if summary_parts:
                                        tool_contexts.append(f"**Tool Response:** {', '.join(summary_parts)}")

                        except json.JSONDecodeError:
                            # If not JSON, treat as plain text but format it nicely
                            if tool_content.strip():
                                tool_contexts.append(f"**Tool Response:**\n{tool_content.strip()}")
                                logging.info(f"Found plain text tool response")
                except Exception as e:
                    logging.error(f"Error extracting tool context: {e}")
                    continue

        # Combine all tool contexts with proper formatting
        if tool_contexts:
            combined_context = "\n\n---\n\n".join(tool_contexts)
            logging.info(
                f"Extracted tool context with {len(tool_contexts)} entries, total length: {len(combined_context)}"
            )
            return combined_context

        logging.debug("No tool context found in LLM responses")
        return ""

    def _extract_tool_context_from_messages(self, messages: List[Dict[str, Any]] = None) -> str:
        """
        Extract context from tool responses in the message history

        Args:
            messages: Optional list of messages to search, defaults to session state messages

        Returns:
            Formatted context string from tool responses, empty if none found
        """
        # Use provided messages or fall back to session state
        search_messages = messages if messages is not None else st.session_state.messages

        tool_contexts = []

        # Look for recent tool messages
        for message in reversed(search_messages):
            if message.get("role") == "tool":
                try:
                    # Parse tool response content
                    tool_content = message.get("content", "")
                    if isinstance(tool_content, str):
                        # Try to parse as JSON to extract formatted results
                        import json

                        try:
                            tool_data = json.loads(tool_content)
                            if isinstance(tool_data, dict):
                                # Check for formatted_results first (preferred format)
                                if "formatted_results" in tool_data:
                                    formatted_results = tool_data["formatted_results"]
                                    if formatted_results and formatted_results.strip():
                                        tool_contexts.append(f"**Tool Response Data:**\n{formatted_results}")
                                        logging.info(f"Found formatted_results from tool response")

                                # Check for other common tool response formats
                                elif "results" in tool_data:
                                    results = tool_data["results"]
                                    if isinstance(results, list) and results:
                                        tool_contexts.append(f"**Tool found {len(results)} results**")
                                        logging.info(f"Found {len(results)} results from tool response")
                                    elif isinstance(results, str) and results.strip():
                                        tool_contexts.append(f"**Tool Response:**\n{results}")
                                        logging.info(f"Found string results from tool response")

                                # Handle weather tool response format
                                elif "location" in tool_data and "current" in tool_data:
                                    location = tool_data.get("location", "Unknown")
                                    current = tool_data.get("current", {})
                                    temp = current.get("temperature", "N/A")
                                    tool_contexts.append(f"**Weather data for {location}** (Current: {temp}°F)")
                                    logging.info(f"Found weather tool response for {location}")

                                # Generic fallback for other structured data
                                else:
                                    # Try to create a summary of the tool response
                                    summary_parts = []
                                    if "query" in tool_data:
                                        summary_parts.append(f"Query: {tool_data['query']}")
                                    if "total_results" in tool_data:
                                        summary_parts.append(f"Results: {tool_data['total_results']}")
                                    if summary_parts:
                                        tool_contexts.append(f"**Tool Response:** {', '.join(summary_parts)}")

                        except json.JSONDecodeError:
                            # If not JSON, treat as plain text but format it nicely
                            if tool_content.strip():
                                tool_contexts.append(f"**Tool Response:**\n{tool_content.strip()}")
                                logging.info(f"Found plain text tool response")
                except Exception as e:
                    logging.error(f"Error extracting tool context: {e}")
                    continue

        # Combine all tool contexts with proper formatting
        if tool_contexts:
            combined_context = "\n\n---\n\n".join(tool_contexts)
            logging.info(
                f"Extracted tool context with {len(tool_contexts)} entries, total length: {len(combined_context)}"
            )
            return combined_context

        logging.debug("No tool context found in messages")
        return ""

    def _run_async_streaming_response(self, prepared_messages: list, model: str) -> Generator[str, None, str]:
        """
        Wrapper to run async streaming response in a synchronous context

        Args:
            prepared_messages: Prepared messages for API call
            model: Model name to use

        Yields:
            Filtered content chunks

        Returns:
            Complete filtered response
        """
        try:

            async def collect_and_yield():
                """Collect all chunks from async generator"""
                chunks = []
                async_gen = self.llm_service.generate_streaming_response(prepared_messages, model)
                async for chunk in async_gen:
                    chunks.append(chunk)
                return chunks

            # Run the async function
            chunks = asyncio.run(collect_and_yield())

            # Yield all chunks for streaming display
            full_response = ""
            for chunk in chunks:
                full_response += chunk
                yield chunk

            return full_response

        except Exception as e:
            error_msg = f"Error in async streaming wrapper: {e}"
            logging.error(error_msg)
            yield "I apologize, but I encountered an error while generating a response."
            return "Error occurred during response generation"

    def _generate_and_display_response(self, prepared_messages: list):
        """
        Generate and display streaming response from LLM

        Args:
            prepared_messages: Prepared messages for API call
        """
        # Create a placeholder for the typing indicator
        typing_indicator = st.empty()
        typing_indicator.markdown(get_typing_indicator_html(), unsafe_allow_html=True)

        try:
            # Clear the typing indicator once we're ready to stream
            typing_indicator.empty()

            # Generate streaming response and display it in real-time
            response_generator = self._run_async_streaming_response(
                prepared_messages, st.session_state["fast_llm_model_name"]
            )

            # Create a chat message container for the assistant response
            with st.spinner(""):
                with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                    # Use st.write_stream to handle the streaming display
                    full_response = st.write_stream(response_generator)

                    # Extract and display tool context from LLM service tool responses
                    tool_context = self._extract_tool_context_from_llm_responses()
                    if tool_context:
                        # Display the context expander for immediate user verification
                        self.display_tool_context_expander(tool_context)
                        # Store tool context in session state for chat history display
                        st.session_state.last_tool_context = tool_context
                        logging.info("Displayed and stored tool context for verification")
                    else:
                        logging.debug("No tool context found to display")

            # Add the complete response to chat history
            self._update_chat_history(full_response, "assistant")

            # Clear processing flag - no need to rerun since we've already displayed the response
            st.session_state.processing = False

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logging.error(error_msg)
            self._update_chat_history(
                "I apologize, but I encountered an error while generating a response.", "assistant"
            )
            # Clear processing flag even on error
            st.session_state.processing = False
            st.rerun()

    def _handle_image_generation(self, prompt: str):
        """
        Handle image generation requests from the user

        Args:
            prompt: The user's original prompt containing image generation request
        """
        try:
            # Extract the image description from the prompt
            image_prompt = self.image_service.extract_image_prompt(prompt)

            # Generate image with loading indicator
            with st.spinner("Generating image..."):
                generated_image, confirmation_message = self.image_service.generate_image_response(image_prompt)

            # Display the image response inline
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                if generated_image:
                    # Display the generated image
                    st.image(generated_image, caption=f"Generated image: {image_prompt}", use_container_width=True)
                    # Display the confirmation message
                    st.markdown(confirmation_message)

                    # Create a special message format for storing images in chat history
                    image_b64 = pil_image_to_base64(generated_image)
                    image_message = {
                        "type": "image",
                        "image_data": image_b64,
                        "image_caption": image_prompt,
                        "text": confirmation_message,
                    }
                    # Add image response to chat history using safe method
                    self._safe_add_message_to_history("assistant", image_message)
                else:
                    # Display error message
                    st.markdown(confirmation_message)
                    # Add error message to chat history using safe method
                    self._safe_add_message_to_history("assistant", confirmation_message)

            # Clear processing flag - no need to rerun since we've already displayed the response
            st.session_state.processing = False

        except Exception as e:
            logging.error(f"Error handling image generation: {e}")
            error_message = "I apologize, but I encountered an error while generating the image. Please try again."

            # Display error message inline
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.markdown(error_message)

            # Add error message to chat history using safe method
            self._safe_add_message_to_history("assistant", error_message)
            # Clear processing flag
            st.session_state.processing = False

    def _contains_tool_call_instructions(self, content: str) -> bool:
        """
        Check if content contains custom tool call instructions

        Args:
            content: The content to check

        Returns:
            True if content contains tool call instructions, False otherwise
        """
        if not isinstance(content, str):
            return False

        # Check for various custom tool call patterns
        import re

        patterns = [
            r'<TOOLCALL-\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\[.*?\]</TOOLCALL>',
            r'<TOOLCALL"\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\s*-\s*\[.*?\]</TOOLCALL>',
            r'<toolcall-\[.*?\]</toolcall>',
            r'<TOOLCALL.?\[.*?\]</TOOLCALL>',
        ]

        for pattern in patterns:
            if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
                return True

        return False

    def _safe_add_message_to_history(self, role: str, content: Any):
        """
        Safely add a message to chat history with validation

        Args:
            role: The role of the message sender
            content: The content of the message (can be string, dict for images, etc.)
        """
        # Handle different content types
        if isinstance(content, str):
            # String content - validate it's not empty and doesn't contain tool calls
            if not content or not content.strip():
                logging.warning(f"Attempted to add empty {role} message to chat history, skipping")
                return

            # Check for tool call instructions
            if self._contains_tool_call_instructions(content):
                logging.warning(
                    f"Attempted to add {role} message with tool call instructions to chat history, skipping"
                )
                return

            content = content.strip()
        elif isinstance(content, dict):
            # Dict content (like image messages) - ensure it has meaningful data
            if not content:
                logging.warning(f"Attempted to add empty dict {role} message to chat history, skipping")
                return
        else:
            # Other content types - ensure they're truthy
            if not content:
                logging.warning(f"Attempted to add empty {role} message to chat history, skipping")
                return

        # Add the validated message to history
        st.session_state.messages.append({"role": role, "content": content})
        logging.debug(f"Added {role} message to chat history")

    def _update_chat_history(self, text: str, role: str):
        """
        Update chat history with new response

        Args:
            text: The response text
            role: The role of the message sender
        """
        # Clean up message format before saving
        st.session_state.messages = self.chat_service.drop_verbose_messages_context(st.session_state.messages)

        # Use the safe method to add the message
        self._safe_add_message_to_history(role, text)

    def run(self):
        """Run the main application"""
        # Display chat history (will include any new messages)
        self.display_chat_history()

        # Check if we're currently processing a message to prevent concurrent processing
        if st.session_state.get("processing", False):
            st.info("Processing your message, please wait...")
            return

        # Handle user input - let Streamlit's natural refresh cycle with smooth transitions handle updates
        if prompt := st.chat_input("Hello, how are you?"):
            # Set processing flag to prevent concurrent requests
            st.session_state.processing = True
            self.process_prompt(prompt)


def main():
    """Main function to run the Streamlit app"""
    # Create and run the application
    app = StreamlitChatApp()
    app.run()


if __name__ == "__main__":
    main()
