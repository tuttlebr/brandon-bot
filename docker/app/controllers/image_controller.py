import base64
import logging
import os
import tempfile
from io import BytesIO
from typing import Tuple

import streamlit as st
from controllers.message_controller import MessageController
from models.chat_config import ChatConfig
from PIL import Image
from utils.config import config


class ImageController:
    """Controller for handling image upload operations"""

    def __init__(
        self,
        config_obj: ChatConfig,
        message_controller: MessageController,
        session_controller=None,
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
        logging.info(
            f"Starting to process image file: {uploaded_file.name}, type: {uploaded_file.type}"
        )

        # Create temporary file with configured suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}"
        ) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            # Check file size limits
            file_size = os.path.getsize(temp_file_path)
            logging.debug(f"Original file size: {file_size / 1024:.2f} KB")

            if file_size > config.file_processing.MAX_IMAGE_SIZE:
                return (
                    False,
                    {
                        "error": f"Image file too large. Maximum size: {config.file_processing.MAX_IMAGE_SIZE // (1024*1024)}MB"
                    },
                )

            # Open image with PIL for resizing
            with Image.open(temp_file_path) as img:
                original_width, original_height = img.size
                max_dimension = 150

                logging.debug(
                    f"Original image dimensions: {original_width}x{original_height}"
                )

                # Check if resizing is needed
                if original_width > max_dimension or original_height > max_dimension:
                    # Calculate scale factor to fit longest side within max_dimension
                    scale = max_dimension / max(original_width, original_height)

                    # Calculate new dimensions
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)

                    logging.debug(f"Image needs resizing. Scale factor: {scale:.3f}")

                    # Resize using high-quality Lanczos resampling
                    resized_img = img.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )

                    logging.info(
                        f"Resized uploaded image '{uploaded_file.name}' from {original_width}x{original_height} to {new_width}x{new_height}"
                    )

                    # Save resized image to bytes
                    img_buffer = BytesIO()
                    # Determine format from file type or default to PNG
                    img_format = uploaded_file.type.split('/')[-1].upper()
                    if img_format == 'JPG':
                        img_format = 'JPEG'
                    if img_format not in ['JPEG', 'PNG', 'GIF', 'BMP']:
                        img_format = 'PNG'

                    logging.debug(f"Saving resized image as format: {img_format}")

                    resized_img.save(img_buffer, format=img_format, optimize=True)
                    img_buffer.seek(0)  # Reset buffer position to beginning
                    image_bytes = img_buffer.getvalue()

                    logging.debug(
                        f"Resized image size: {len(image_bytes) / 1024:.2f} KB"
                    )
                else:
                    logging.debug(
                        f"Uploaded image '{uploaded_file.name}' already within size limit ({original_width}x{original_height})"
                    )
                    # Read original image bytes
                    with open(temp_file_path, "rb") as image_file:
                        image_bytes = image_file.read()

                # Convert to base64
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # Verify the dimensions of the final image
                from io import BytesIO as IOByteIO

                from PIL import Image as PILImage

                verify_img = PILImage.open(IOByteIO(image_bytes))
                verify_width, verify_height = verify_img.size
                logging.debug(
                    f"Final image dimensions after processing: {verify_width}x{verify_height}"
                )

                # Update file size after potential resizing
                file_size = len(image_bytes)

                logging.info(
                    f"Processed image '{uploaded_file.name}': {file_size / 1024:.2f} KB, {verify_width}x{verify_height}"
                )

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
            logging.error(f"Image processing error: {e}", exc_info=True)
            return False, {"error": error_msg}

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
                logging.debug(f"Cleaned up temporary file: {temp_file_path}")
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
                    image_data["image_data"],
                    image_data["filename"],
                    image_data["file_type"],
                )

                # Get the stored image data to retrieve the file path
                stored_image_data = self.session_controller.get_latest_uploaded_image()
                if stored_image_data and "file_path" in stored_image_data:
                    # Add the file path to the image_data dict
                    image_data["file_object"] = stored_image_data["file_path"]
                    logging.info(
                        f"Added file path to image_data: {stored_image_data['file_path']}"
                    )
                else:
                    logging.warning("Could not retrieve file path for stored image")

                # Store the image data in session state for easy access
                st.session_state.current_image_base64 = image_data["image_data"]
                st.session_state.current_image_filename = filename
                st.session_state.current_image_id = image_id

                logging.debug(
                    f"Stored image in session state - filename: {filename}, data length: {len(image_data['image_data'])}"
                )

                # Verify the size of what we stored in session state
                import base64 as b64

                stored_bytes = b64.b64decode(st.session_state.current_image_base64)
                logging.debug(
                    f"Session state image size: {len(stored_bytes) / 1024:.2f} KB"
                )

                # Add user notification message (without base64 data)
                user_message = f"📷 Uploaded image: **{filename}**"
                self.message_controller.safe_add_message_to_history(
                    "user", user_message
                )

                # Display the user message
                with st.chat_message("user", avatar=self.config_obj.user_avatar):
                    st.markdown(user_message)

                # Add assistant response
                assistant_message = f"I've received your image **{filename}**. What would you like to know about it?"
                self.message_controller.safe_add_message_to_history(
                    "assistant", assistant_message
                )

                # Display the assistant message
                with st.chat_message(
                    "assistant", avatar=self.config_obj.assistant_avatar
                ):
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
            logging.info(
                f"Image '{uploaded_file.name}' is already being processed, skipping duplicate processing"
            )
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
