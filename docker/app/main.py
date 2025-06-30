import logging
import random

import streamlit as st

# Import the controller classes
from controllers import FileController, MessageController, ResponseController, SessionController
from models import ChatConfig
from services import ChatService, ImageService, LLMService
from tools.initialize_tools import initialize_all_tools
from ui import ChatHistoryComponent, apply_custom_styles
from utils.config import config
from utils.exceptions import ChatbotException, ConfigurationError
from utils.system_prompt import greeting_prompt


class ProductionStreamlitChatApp:
    def __init__(self):
        """Initialize the production-ready Streamlit chat application"""
        try:
            # Initialize tools first
            initialize_all_tools()

            # Initialize configuration using centralized system
            self.config_obj = ChatConfig.from_environment()

            # Apply custom styling using centralized configuration
            apply_custom_styles()
            st.markdown(
                f'<h1 style="color: {config.ui.BRAND_COLOR}; text-align: center;">{greeting_prompt()}</h1>',
                unsafe_allow_html=True,
            )

            # Initialize services
            self.chat_service = ChatService(self.config_obj)
            self.image_service = ImageService(self.config_obj)
            self.llm_service = LLMService(self.config_obj)

            # Initialize UI components
            self.chat_history_component = ChatHistoryComponent(self.config_obj)

            # Initialize controllers with dependency injection
            self.session_controller = SessionController(self.config_obj)
            self.message_controller = MessageController(self.config_obj, self.chat_service, self.session_controller)
            self.file_controller = FileController(self.config_obj, self.message_controller, self.session_controller)
            self.response_controller = ResponseController(
                self.config_obj,
                self.llm_service,
                self.message_controller,
                self.session_controller,
                self.chat_history_component,
            )

            # Initialize session state using controller
            self.session_controller.initialize_session_state()

        except Exception as e:
            logging.error(f"Failed to initialize application: {e}")
            raise ConfigurationError(f"Application initialization failed: {e}")

    def display_chat_history(self):
        """Display the chat history using the chat history component"""
        try:
            # Use session controller's safe message access
            messages = self.session_controller.get_messages()
            self.chat_history_component.display_chat_history(messages)
        except Exception as e:
            logging.error(f"Error displaying chat history: {e}")
            st.error("Failed to display chat history. Please refresh the page.")

    def process_prompt(self, prompt: str):
        """
        Process user prompt using production-ready controller pattern

        Args:
            prompt: The user's text prompt
        """
        try:
            # Set processing state using controller
            self.session_controller.set_processing_state(True)

            # Clean and validate the prompt
            prompt = prompt.strip()

            # Display user message with centralized configuration
            with st.chat_message("user", avatar=self.config_obj.user_avatar):
                truncated_prompt = self.message_controller.truncate_long_prompt(prompt)
                st.markdown(truncated_prompt)

            # Validate prompt using controller with proper error handling
            if not self.message_controller.validate_prompt(prompt):
                st.error("Invalid input detected. Please try again with a different message.")
                self.session_controller.set_processing_state(False)
                return

            # Clear previous context and tool responses using controllers
            self.session_controller.clear_tool_context()
            if hasattr(self.llm_service, 'last_tool_responses'):
                self.llm_service.last_tool_responses = []

            # Add user message to chat history using safe controller method
            self.message_controller.safe_add_message_to_history("user", prompt)

            # Prepare messages for processing using controller
            messages = self.session_controller.get_messages()
            prepared_messages = self.message_controller.prepare_messages_for_processing(messages)

            # Generate and display response using controller with centralized spinner
            random_icon = random.choice(config.ui.SPINNER_ICONS)
            with st.spinner(f"{random_icon} _Typing..._"):
                self.response_controller.generate_and_display_response_no_spinner(prepared_messages)

        except ChatbotException as e:
            logging.error(f"Chatbot error processing prompt: {e}")
            st.error(f"Error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error processing prompt: {e}")
            st.error("An unexpected error occurred. Please try again.")
        finally:
            self.session_controller.set_processing_state(False)

    def process_pdf_upload(self, uploaded_file):
        """
        Process uploaded PDF file using production-ready controller

        Args:
            uploaded_file: Streamlit uploaded file object
        """
        try:
            self.session_controller.set_processing_state(True)

            # Use file controller with centralized configuration and error handling
            success = self.file_controller.process_pdf_upload(uploaded_file)
            if success:
                self.file_controller.mark_file_as_processed(uploaded_file.name)

        except ChatbotException as e:
            logging.error(f"Chatbot error processing PDF: {e}")
            st.error(f"PDF Error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error processing PDF: {e}")
            st.error("An unexpected error occurred while processing the PDF.")
        finally:
            self.session_controller.set_processing_state(False)

    def run(self):
        """Run the production-ready application using controller pattern"""
        try:
            # Display chat history
            self.display_chat_history()

            # Check processing state using controller
            if self.session_controller.is_processing():
                st.info("Processing your message, please wait...")
                return

            # Handle user input with centralized configuration
            if prompt := st.chat_input("Hello, how are you?"):
                self.process_prompt(prompt)

            # Handle PDF upload with centralized configuration and validation
            uploaded_file = st.file_uploader(
                "📄 Upload PDF Document",
                type=self.file_controller.get_supported_file_types(),
                accept_multiple_files=False,
                help=f"Upload a PDF document to analyze and discuss its content (Max size: {self.file_controller.get_file_size_limit_mb()}MB)",
                key="pdf_uploader",
            )

            if uploaded_file is not None:
                # Check if this is a new upload using controller
                if self.file_controller.is_new_upload(uploaded_file):
                    self.process_pdf_upload(uploaded_file)
                    st.rerun()

        except Exception as e:
            logging.error(f"Error in application run loop: {e}")
            st.error("Application error. Please refresh the page.")


def main():
    """Main function to run the production-ready Streamlit app"""
    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('/tmp/chatbot.log', mode='a')],
    )

    # Validate environment configuration on startup
    try:
        config.validate_environment()
        logging.info("Environment configuration validated successfully")
    except ConfigurationError as e:
        logging.error(f"Configuration error: {e}")
        st.error(f"Configuration error: {e}")
        st.stop()
    except Exception as e:
        logging.error(f"Unexpected validation error: {e}")
        st.error("Failed to validate configuration. Please check your environment variables.")
        st.stop()

    # Create and run the production application
    try:
        app = ProductionStreamlitChatApp()
        app.run()
    except ConfigurationError as e:
        logging.error(f"Application configuration error: {e}")
        st.error(f"Configuration error: {e}")
    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        st.error("Application failed to start. Please check the logs.")


if __name__ == "__main__":
    main()
