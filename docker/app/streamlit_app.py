import logging

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

        # Display user message
        with st.chat_message("user", avatar=self.config.user_avatar):
            st.markdown(prompt, unsafe_allow_html=True)

        # Check if this is an image generation request
        if self.image_service.detect_image_generation_request(prompt):
            self._handle_image_generation(prompt)
            return

        try:
            # Enhance prompt with context if needed
            with st.spinner("Looking for relevant context..."):
                prompt, context = self.chat_service.enhance_prompt_with_context(prompt)

                # Display context if available
                self.chat_history_component.display_context_expander(context)

                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Prepare messages for API call
                prepared_messages = self.chat_service.prepare_messages_for_api(st.session_state.messages, context)

            # Generate and display response
            self._generate_and_display_response(prepared_messages)

        except Exception as e:
            logging.error(f"Error processing prompt: {e}")
            st.error("An error occurred while processing your request. Please try again.")

    def _generate_and_display_response(self, prepared_messages: list):
        """
        Generate and display streaming response from LLM

        Args:
            prepared_messages: Prepared messages for API call
        """
        # Create a placeholder for the typing indicator
        typing_indicator = st.empty()
        typing_indicator.markdown(get_typing_indicator_html(), unsafe_allow_html=True)

        with st.chat_message("assistant", avatar=self.config.assistant_avatar):
            try:
                # Clear the typing indicator once we're ready to stream
                typing_indicator.empty()

                # Generate streaming response
                with st.spinner("Thinking..."):
                    response_generator = self.llm_service.generate_streaming_response(
                        prepared_messages, st.session_state["openai_model"]
                    )

                    # Stream response to UI
                    full_response = st.write_stream(response_generator)

                    # Add response to chat history
                    self._update_chat_history(full_response, "assistant")

            except Exception as e:
                error_msg = f"Error generating response: {e}"
                logging.error(error_msg)
                st.error(f"I'm having trouble generating a response. Please try again. {error_msg}")
                self._update_chat_history(
                    "I apologize, but I encountered an error while generating a response.", "assistant"
                )

    def _handle_image_generation(self, prompt: str):
        """
        Handle image generation requests from the user

        Args:
            prompt: The user's original prompt containing image generation request
        """
        try:
            # Extract the image description from the prompt
            image_prompt = self.image_service.extract_image_prompt(prompt)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate image with loading indicator
            with st.spinner("Generating image..."):
                generated_image, confirmation_message = self.image_service.generate_image_response(image_prompt)

            # Display the response
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                if generated_image:
                    # Display the generated image
                    st.image(generated_image, caption=f"Generated image: {image_prompt}", use_container_width=True)
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
                    st.session_state.messages.append({"role": "assistant", "content": confirmation_message})

        except Exception as e:
            logging.error(f"Error handling image generation: {e}")
            error_message = "I apologize, but I encountered an error while generating the image. Please try again."

            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                st.error(error_message)

            st.session_state.messages.append({"role": "assistant", "content": error_message})

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
        # Display chat history
        self.display_chat_history()

        # Handle user input
        if prompt := st.chat_input("Hello, how are you?"):
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
