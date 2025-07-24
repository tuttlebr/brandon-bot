import logging
import time

import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml

# Import the controller classes
from controllers.file_controller import FileController
from controllers.image_controller import ImageController
from controllers.message_controller import MessageController
from controllers.response_controller import ResponseController
from controllers.session_controller import SessionController
from models import ChatConfig
from services import ChatService, ImageService, LLMService
from services.pdf_context_service import PDFContextService
from tools.initialize_tools import initialize_all_tools
from ui import ChatHistoryComponent
from utils.animated_loading import get_galaxy_animation_html
from utils.config import config
from utils.exceptions import ChatbotException, ConfigurationError
from yaml.loader import SafeLoader


class ProductionStreamlitChatApp:
    def __init__(self):
        """Initialize the production-ready Streamlit chat application"""
        try:
            # Initialize configuration using centralized system
            self.config_obj = ChatConfig.from_environment()

            # Apply mobile optimization styles
            # apply_mobile_styles() # Removed as per edit hint

            # Tools and LLM client service are already initialized in startup.initialize_app()
            # Just verify they're available
            from tools.registry import get_all_tool_definitions

            if len(get_all_tool_definitions()) == 0:
                logging.warning("No tools found, attempting initialization")
                initialize_all_tools()

            # Apply custom styling using centralized configuration
            custom_galaxy = get_galaxy_animation_html(
                center_dot_size=100,  # Larger galactic core
                container_size=250,  # Bigger galaxy
                animation_duration=12.0,  # Slower, more majestic rotation
                enable_3d_depth=True,  # Keep the 3D effects
            )

            components.html(custom_galaxy, height=300)

            # Initialize services
            self.chat_service = ChatService(self.config_obj)
            self.image_service = ImageService(self.config_obj)
            self.llm_service = LLMService(self.config_obj)
            self.pdf_context_service = PDFContextService(self.config_obj)

            # Initialize UI components
            self.chat_history_component = ChatHistoryComponent(self.config_obj)

            # Initialize controllers with dependency injection
            self.session_controller = SessionController(self.config_obj)
            self.message_controller = MessageController(
                self.config_obj, self.chat_service, self.session_controller
            )
            self.file_controller = FileController(
                self.config_obj, self.message_controller, self.session_controller
            )
            self.image_controller = ImageController(
                self.config_obj, self.message_controller, self.session_controller
            )
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
            # Show PDF info if available
            if hasattr(self, "pdf_context_service"):
                pdf_info = self.pdf_context_service.get_pdf_info_for_display()
                if pdf_info:
                    st.info(pdf_info)

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
                st.markdown(prompt)

            # Validate prompt using controller with proper error handling
            if not self.message_controller.validate_prompt(prompt):
                st.error(
                    "Invalid input detected. Please try again with a different message."
                )
                self.session_controller.set_processing_state(False)
                return

            # Clear previous context and tool responses using controllers
            self.session_controller.clear_tool_context()
            if hasattr(self.llm_service, "last_tool_responses"):
                self.llm_service.last_tool_responses = []

            # Add user message to chat history using safe controller method
            self.message_controller.safe_add_message_to_history("user", prompt)

            # Prepare messages for processing using controller
            messages = self.session_controller.get_messages()
            prepared_messages = self.message_controller.prepare_messages_for_processing(
                messages
            )

            # Inject PDF context if available
            prepared_messages = self.pdf_context_service.inject_pdf_context(
                prepared_messages, prompt
            )

            # Generate and display response using controller with centralized spinner
            cleanup_fn = None
            cleanup_fn = (
                self.response_controller.generate_response_with_cleanup_separation(
                    prepared_messages
                )
            )

            # Execute cleanup
            if cleanup_fn:
                cleanup_fn()

        except ChatbotException as e:
            logging.error(f"Chatbot error processing prompt: {e}")
            st.error(f"Error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error processing prompt: {e}")
            st.error("An unexpected error occurred. Please try again.")
        finally:
            self.session_controller.set_processing_state(False)

    @st.fragment(run_every=1)
    def pdf_analysis_progress_fragment(self):
        """
        Fragment to show real-time PDF analysis progress
        """
        if (
            hasattr(st.session_state, "pdf_analysis_progress")
            and st.session_state.pdf_analysis_progress
        ):
            progress_info = st.session_state.pdf_analysis_progress

            if progress_info.get("status") in ["starting", "analyzing"]:
                # st.info("🔍 **Intelligent PDF Analysis in Progress**")

                message = progress_info.get("message", "Processing...")

                logging.debug(f"PDF analysis message: {message}")

            elif progress_info.get("status") == "completed":
                logging.debug("PDF analysis completed")
                # Clear progress after a brief moment
                if not hasattr(st.session_state, "analysis_completion_shown"):
                    st.session_state.analysis_completion_shown = True
                else:
                    # Clear the progress info
                    st.session_state.pdf_analysis_progress = None
                    del st.session_state.analysis_completion_shown

    @st.fragment(run_every=1)
    def pdf_processing_fragment(self):
        """
        Self-contained PDF processing fragment that runs independently
        Uses st.fragment to poll every second and update status
        """
        st.subheader("📄 PDF Document Upload")

        # Get current processing status
        processing_status = getattr(st.session_state, "pdf_processing_status", None)

        if processing_status == "processing":
            # Show processing status
            getattr(st.session_state, "pdf_processing_file", "Unknown")
            getattr(st.session_state, "pdf_processing_start_time", time.time())

        elif processing_status == "completed":
            # Show completion message
            message = getattr(
                st.session_state, "pdf_processing_message", "✅ Processing completed"
            )
            st.success(message)

            # Clear processing status after brief display
            if (
                not hasattr(st.session_state, "completion_shown")
                or not st.session_state.completion_shown
            ):
                st.session_state.completion_shown = True
                time.sleep(0.01)
            else:
                # Reset after showing completion
                st.session_state.pdf_processing_status = None
                st.session_state.pdf_processing_file = None
                st.session_state.pdf_processing_message = None
                st.session_state.completion_shown = False

        elif processing_status == "error":
            # Show error message
            message = getattr(
                st.session_state, "pdf_processing_message", "❌ Processing failed"
            )
            st.error(message)

            # Clear error status after brief display
            if (
                not hasattr(st.session_state, "error_shown")
                or not st.session_state.error_shown
            ):
                st.session_state.error_shown = True
                time.sleep(0.01)
            else:
                # Reset after showing error
                st.session_state.pdf_processing_status = None
                st.session_state.pdf_processing_file = None
                st.session_state.pdf_processing_message = None
                st.session_state.error_shown = False

        else:
            # Normal state - show file uploader
            uploaded_file = st.file_uploader(
                "Choose PDF file",
                type=self.file_controller.get_supported_file_types(),
                accept_multiple_files=False,
                help=f"Upload a PDF document to analyze and discuss its content (Max size: {self.file_controller.get_file_size_limit_mb()}MB)",
                key="pdf_uploader",
            )

            # Handle new uploads - process synchronously within fragment
            if (
                uploaded_file is not None
                and not self.session_controller.is_processing()
            ):
                if self.file_controller.is_new_upload(uploaded_file):
                    # Mark as processing immediately
                    st.session_state.pdf_processing_status = "processing"
                    st.session_state.pdf_processing_start_time = time.time()
                    st.session_state.pdf_processing_file = uploaded_file.name

                    # Process PDF synchronously with spinner
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            logging.debug(
                                f"Starting synchronous PDF processing for: {uploaded_file.name}"
                            )

                            # Use file controller for processing - this is blocking
                            success = self.file_controller.process_pdf_upload(
                                uploaded_file
                            )

                            if success:
                                self.file_controller.mark_file_as_processed(
                                    uploaded_file.name
                                )
                                st.session_state.pdf_processing_status = "completed"
                                st.session_state.pdf_processing_message = f"✅ Successfully processed PDF: {uploaded_file.name}"
                                logging.debug(
                                    f"Synchronous PDF processing completed for: {uploaded_file.name}"
                                )
                            else:
                                st.session_state.pdf_processing_status = "error"
                                st.session_state.pdf_processing_message = (
                                    f"❌ Failed to process PDF: {uploaded_file.name}"
                                )
                                logging.error(
                                    f"Synchronous PDF processing failed for: {uploaded_file.name}"
                                )

                        except Exception as e:
                            logging.error(f"Synchronous PDF processing error: {e}")
                            st.session_state.pdf_processing_status = "error"
                            st.session_state.pdf_processing_message = (
                                f"❌ PDF processing error: {str(e)}"
                            )

            # Show current PDF status if available
            if self.session_controller.has_pdf_documents():
                latest_pdf = self.session_controller.get_latest_pdf_document()
                logging.debug(f"Latest PDF: {latest_pdf}")
                if latest_pdf:
                    filename = latest_pdf.get("filename", "Unknown")
                    pages = latest_pdf.get("total_pages", 0)
                    st.success(f"✅ Current PDF: {filename} ({pages} pages)")
                    if pages > config.file_processing.PDF_SUMMARIZATION_THRESHOLD:
                        st.markdown(
                            "💡 This is a large document. You can ask me to 'summarize the PDF' for a quick overview!"
                        )

                    if st.button(
                        "🗑️ Remove Current PDF",
                        help="Remove the current PDF from session",
                    ):
                        self.session_controller.clear_pdf_documents()
                        st.rerun()

            # Image Upload Section
            st.markdown("---")
            st.markdown("### 📷 Image Upload")

            # Check image processing status
            processing_status = getattr(
                st.session_state, "image_processing_status", None
            )

            if processing_status == "processing":
                # Show processing message
                message = getattr(
                    st.session_state,
                    "image_processing_message",
                    "🔄 Processing image...",
                )
                st.info(message)

                # Clear processing status after brief display
                if (
                    not hasattr(st.session_state, "image_processing_shown")
                    or not st.session_state.image_processing_shown
                ):
                    st.session_state.image_processing_shown = True
                    time.sleep(0.01)
                else:
                    # Reset after showing processing
                    st.session_state.image_processing_status = None
                    st.session_state.image_processing_file = None
                    st.session_state.image_processing_message = None
                    st.session_state.image_processing_shown = False

            elif processing_status == "error":
                # Show error message
                message = getattr(
                    st.session_state, "image_processing_message", "❌ Processing failed"
                )
                st.error(message)

                # Clear error status after brief display
                if (
                    not hasattr(st.session_state, "image_error_shown")
                    or not st.session_state.image_error_shown
                ):
                    st.session_state.image_error_shown = True
                    time.sleep(0.01)
                else:
                    # Reset after showing error
                    st.session_state.image_processing_status = None
                    st.session_state.image_processing_file = None
                    st.session_state.image_processing_message = None
                    st.session_state.image_error_shown = False

            else:
                # Normal state - show image uploader
                uploaded_image = st.file_uploader(
                    "Choose image file",
                    type=self.image_controller.get_supported_file_types(),
                    accept_multiple_files=False,
                    help=f"Upload an image to analyze and discuss its content (Max size: {self.image_controller.get_file_size_limit_mb()}MB)",
                    key="image_uploader",
                )

                if uploaded_image and self.image_controller.is_new_upload(
                    uploaded_image
                ):
                    try:
                        # Mark as processing
                        st.session_state.image_processing_status = "processing"
                        st.session_state.image_processing_file = uploaded_image.name
                        st.session_state.image_processing_message = (
                            f"🔄 Processing image: {uploaded_image.name}"
                        )

                        # Process the image
                        success = self.image_controller.process_image_upload(
                            uploaded_image
                        )

                        if success:
                            st.session_state.image_processing_status = None
                            st.session_state.image_processing_file = None
                            st.session_state.image_processing_message = None
                            st.rerun()
                        else:
                            st.session_state.image_processing_status = "error"
                            st.session_state.image_processing_message = (
                                f"❌ Failed to process image: {uploaded_image.name}"
                            )

                    except Exception as e:
                        logging.error(f"Image processing error: {e}")
                        st.session_state.image_processing_status = "error"
                        st.session_state.image_processing_message = (
                            f"❌ Image processing error: {str(e)}"
                        )

            # Show current image status if available
            if self.session_controller.has_uploaded_images():
                latest_image = self.session_controller.get_latest_uploaded_image()
                if latest_image:
                    filename = latest_image.get("filename", "Unknown")
                    st.success(f"✅ Current Image: {filename}")

                    # Remove verbose logging that runs every second
                    # Only log once when the image changes
                    if (
                        not hasattr(st.session_state, "_last_displayed_image")
                        or st.session_state._last_displayed_image != filename
                    ):
                        st.session_state._last_displayed_image = filename
                        if "image_data" in latest_image:
                            import base64

                            img_bytes = base64.b64decode(latest_image["image_data"])
                            logging.info(
                                f"Displaying new image {filename}: {len(img_bytes) / 1024:.2f} KB"
                            )

                            # Check actual dimensions only for new images
                            try:
                                from io import BytesIO

                                from PIL import Image

                                img = Image.open(BytesIO(img_bytes))
                                width, height = img.size
                                logging.info(f"Image dimensions: {width}x{height}")
                            except Exception as e:
                                logging.error(f"Failed to check image dimensions: {e}")

                    st.image(latest_image["file_path"])

                    if st.button(
                        "🗑️ Remove Current Image",
                        help="Remove the current image from session",
                    ):
                        self.session_controller.clear_uploaded_images()
                        st.rerun()

    def run(self):
        """Run the production-ready application using controller pattern"""
        try:
            # Display chat history
            self.display_chat_history()

            # Show PDF analysis progress if active
            self.pdf_analysis_progress_fragment()

            # Handle user input with centralized configuration
            if prompt := st.chat_input():
                self.process_prompt(prompt)

            # Handle PDF upload and processing via fragment
            with st.sidebar:
                self.pdf_processing_fragment()

        except Exception as e:
            logging.error(f"Error in application run loop: {e}")
            st.error("Application error. Please refresh the page.")


def main():
    """Main function to run the production-ready Streamlit app"""

    # Initialize app startup settings (including warning suppression)
    from utils.startup import initialize_app

    initialize_app()

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
        st.error(
            "Failed to validate configuration. Please check your environment variables."
        )
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
    with open(".streamlit/auth.yaml") as file:
        auth_config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        auth_config["credentials"],
        auth_config["cookie"]["name"],
        auth_config["cookie"]["key"],
        auth_config["cookie"]["expiry_days"],
    )

    try:
        authenticator.login(location="sidebar")
        if st.session_state.get("authentication_status"):
            authenticator.logout(location="sidebar")
            main()
        elif st.session_state.get("authentication_status") is False:
            st.error("Username/password is incorrect")
        elif st.session_state.get("authentication_status") is None:
            st.warning("Please enter your username and password")
    except Exception as e:
        st.error(e)
