import base64
import logging
import os
import tempfile
from typing import Tuple

import streamlit as st
from controllers.message_controller import MessageController
from models.chat_config import ChatConfig
from utils.config import config


class ImageController:
    """Controller for handling image upload operations"""

    def __init__(
        self, config_obj: ChatConfig, message_controller: MessageController, session_controller=None,
    ):
        """
        Initialize the image controller

        Args:
            config_obj: Application configuration
            message_controller: Message controller for adding responses to history
            session_controller: Session controller for image storage
        """
        self.config_obj = config_obj
        self.message_controller = message_controller
        self.session_controller = session_controller

    def process_image_upload(self, uploaded_file) -> bool:
        """
        Process uploaded image file and store it for analysis

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Mark file as being processed immediately to prevent duplicates
            self.mark_file_as_processing(uploaded_file.name)

            # Process the image file
            success, result = self._process_image_file(uploaded_file)

            if success:
                self._handle_successful_processing(uploaded_file.name, result)
                return True
            else:
                self._handle_processing_error(result)
                return False

        except Exception as e:
            error_msg = f"❌ **Image Processing Error:** An unexpected error occurred while processing your image. Please try again."
            logging.error(f"Unexpected image processing error: {e}")
            self._display_error_message(error_msg)
            return False
        finally:
            # Always clear the processing marker when done
            self.clear_processing_file()

    def _process_image_file(self, uploaded_file) -> Tuple[bool, dict]:
        """
        Process the image file and convert to base64

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Tuple of (success: bool, result: dict)
        """
        # Create temporary file with configured suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            # Check file size limits
            file_size = os.path.getsize(temp_file_path)
            if file_size > config.file_processing.MAX_IMAGE_SIZE:
                return (
                    False,
                    {
                        "error": f"Image file too large. Maximum size: {config.file_processing.MAX_IMAGE_SIZE // (1024*1024)}MB"
                    },
                )

            # Read the image file and convert to base64
            with open(temp_file_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Get file type from uploaded file
            file_type = uploaded_file.type if uploaded_file.type else "image/png"

            return (
                True,
                {
                    "image_data": image_base64,
                    "filename": uploaded_file.name,
                    "file_type": file_type,
                    "size_bytes": file_size,
                },
            )

        except Exception as e:
            error_msg = f"Failed to process image file: {str(e)}"
            logging.error(f"Image processing error: {e}")
            return False, {"error": error_msg}

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    def _handle_successful_processing(self, filename: str, image_data: dict):
        """
        Handle successful image processing

        Args:
            filename: Name of the processed file
            image_data: Processed image data
        """
        try:
            # Store image in session controller
            if self.session_controller:
                image_id = self.session_controller.store_uploaded_image(
                    image_data["image_data"], image_data["filename"], image_data["file_type"],
                )

                # Get the stored image data to retrieve the file path
                stored_image_data = self.session_controller.get_latest_uploaded_image()
                if stored_image_data and "file_path" in stored_image_data:
                    # Add the file path to the image_data dict
                    image_data["file_object"] = stored_image_data["file_path"]
                    logging.info(f"Added file path to image_data: {stored_image_data['file_path']}")
                else:
                    logging.warning("Could not retrieve file path for stored image")

                # Store the image data in session state for easy access
                st.session_state.current_image_base64 = image_data["image_data"]
                st.session_state.current_image_filename = filename
                st.session_state.current_image_id = image_id

                logging.info(
                    f"Stored image in session state - filename: {filename}, data length: {len(image_data['image_data'])}"
                )

                # Add user notification message (without base64 data)
                user_message = f"📷 Uploaded image: **{filename}**"
                self.message_controller.safe_add_message_to_history("user", user_message)

                # Display the user message
                with st.chat_message("user", avatar=self.config_obj.user_avatar):
                    st.markdown(user_message)

                # Add assistant response
                assistant_message = f"I've received your image **{filename}**. What would you like to know about it?"
                self.message_controller.safe_add_message_to_history("assistant", assistant_message)

                # Display the assistant message
                with st.chat_message("assistant", avatar=self.config_obj.assistant_avatar):
                    st.markdown(assistant_message)

                # Mark file as processed
                self.mark_file_as_processed(filename)

                logging.info(f"Successfully processed and stored image: {filename}")

            else:
                logging.error("Session controller not available for image storage")

        except Exception as e:
            logging.error(f"Error handling successful image processing: {e}")
            raise

    def _handle_processing_error(self, error_result: dict):
        """
        Handle image processing errors

        Args:
            error_result: Error information dictionary
        """
        error_msg = f"❌ **Image Processing Error:** {error_result.get('error', 'Unknown error')}"
        self._display_error_message(error_msg)

    def _display_error_message(self, error_msg: str):
        """
        Display error message in chat UI and add to history

        Args:
            error_msg: Error message to display
        """
        with st.chat_message("assistant", avatar=self.config_obj.assistant_avatar):
            st.error(error_msg)

        self.message_controller.safe_add_message_to_history("assistant", error_msg)

    def is_new_upload(self, uploaded_file) -> bool:
        """
        Check if this is a new file upload

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            True if this is a new upload, False otherwise
        """
        # Check if we're currently processing this file
        if (
            hasattr(st.session_state, "currently_processing_image")
            and st.session_state.currently_processing_image == uploaded_file.name
        ):
            logging.info(f"Image '{uploaded_file.name}' is already being processed, skipping duplicate processing")
            return False

        # Check if this file was already processed successfully
        return (
            not hasattr(st.session_state, "last_uploaded_image")
            or st.session_state.last_uploaded_image != uploaded_file.name
        )

    def mark_file_as_processing(self, filename: str):
        """
        Mark a file as currently being processed to prevent duplicates

        Args:
            filename: Name of the file being processed
        """
        st.session_state.currently_processing_image = filename
        logging.info(f"Marked image '{filename}' as currently being processed")

    def clear_processing_file(self):
        """Clear the currently processing file marker"""
        if hasattr(st.session_state, "currently_processing_image"):
            filename = st.session_state.currently_processing_image
            st.session_state.currently_processing_image = None
            logging.info(f"Cleared processing marker for image '{filename}'")

    def mark_file_as_processed(self, filename: str):
        """
        Mark a file as processed to prevent reprocessing

        Args:
            filename: Name of the processed file
        """
        st.session_state.last_uploaded_image = filename
        logging.info(f"Marked image '{filename}' as successfully processed")

    def get_supported_file_types(self) -> list:
        """
        Get list of supported file types for upload

        Returns:
            List of supported file extensions
        """
        return config.file_processing.SUPPORTED_IMAGE_TYPES

    def get_file_size_limit_mb(self) -> int:
        """
        Get the file size limit in megabytes

        Returns:
            File size limit in MB
        """
        return config.file_processing.MAX_IMAGE_SIZE // (1024 * 1024)
