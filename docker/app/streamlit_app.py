import logging
from typing import Any, Dict, List

import streamlit as st
from models import ChatConfig
from services import ChatService, ImageService, LLMService
from ui import ChatHistoryComponent, apply_custom_styles, get_typing_indicator_html
from utils.image import pil_image_to_base64
from utils.system_prompt import SYSTEM_PROMPT, greeting_prompt


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
            st.session_state.openai_model = self.config.llm_model_name
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state.current_page = 0

    def display_chat_history(self):
        """Display the chat history using the chat history component"""
        self.chat_history_component.display_chat_history(st.session_state.messages)

    def process_prompt(self, prompt: str):
        """
        Process user prompt and generate response

        Args:
            prompt: The user's text prompt
        """
        # Clean the prompt
        prompt = prompt.strip()

        # Clean previous chat history from context
        st.session_state.messages = self.chat_service.clean_chat_history_context(st.session_state.messages)

        # Display user message immediately in the UI
        with st.chat_message("user", avatar=self.config.user_avatar):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

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
                            if isinstance(tool_data, dict) and "formatted_results" in tool_data:
                                formatted_results = tool_data["formatted_results"]
                                if formatted_results and formatted_results.strip():
                                    tool_contexts.append(formatted_results)
                                    logging.info(f"Found formatted_results from tool response")
                            elif isinstance(tool_data, dict) and "results" in tool_data:
                                # Handle other tool response formats
                                results = tool_data["results"]
                                if results:
                                    tool_contexts.append(f"Found {len(results)} relevant results")
                                    logging.info(f"Found {len(results)} results from tool response")
                        except json.JSONDecodeError:
                            # If not JSON, treat as plain text
                            if tool_content.strip():
                                tool_contexts.append(tool_content)
                                logging.info(f"Found plain text tool response")
                except Exception as e:
                    logging.error(f"Error extracting tool context: {e}")
                    continue

        # Combine all tool contexts
        if tool_contexts:
            combined_context = "\n\n".join(tool_contexts)
            logging.info(
                f"Extracted tool context with {len(tool_contexts)} entries, total length: {len(combined_context)}"
            )
            return combined_context

        logging.info("No tool context found in messages")
        return ""

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
            response_generator = self.llm_service.generate_streaming_response(
                prepared_messages, st.session_state["openai_model"]
            )

            # Create a chat message container for the assistant response
            with st.spinner(""):
                with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                    # Use st.write_stream to handle the streaming display
                    full_response = st.write_stream(response_generator)

            # Add the complete response to chat history
            self._update_chat_history(full_response, "assistant")

            # Extract tool context from the prepared_messages that now contain tool responses
            tool_context = self._extract_tool_context_from_messages(prepared_messages)
            if tool_context:
                # Store tool context in session state to display it properly in history
                st.session_state.last_tool_context = tool_context
                logging.info("Stored tool context for display")

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
                    # Add image response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": image_message})
                else:
                    # Display error message
                    st.markdown(confirmation_message)
                    # Add error message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": confirmation_message})

            # Clear processing flag - no need to rerun since we've already displayed the response
            st.session_state.processing = False

        except Exception as e:
            logging.error(f"Error handling image generation: {e}")
            error_message = "I apologize, but I encountered an error while generating the image. Please try again."

            # Display error message inline
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.markdown(error_message)

            # Add error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            # Clear processing flag
            st.session_state.processing = False

    def _update_chat_history(self, text: str, role: str):
        """
        Update chat history with new response

        Args:
            text: The response text
            role: The role of the message sender
        """
        # Clean up message format before saving
        st.session_state.messages = self.chat_service.drop_verbose_messages_context(st.session_state.messages)

        # Add message to history
        st.session_state.messages.append({"role": role, "content": text})
        logging.debug(f"Updated chat history: {st.session_state.messages}")

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
    # Uncomment to enable password protection
    # if not check_password():
    #     st.stop()

    # Create and run the application
    app = StreamlitChatApp()
    app.run()


if __name__ == "__main__":
    main()
