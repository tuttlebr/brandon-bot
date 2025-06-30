import json
import logging
import os
import tempfile
from typing import Optional, Tuple

import requests
import streamlit as st
from controllers.message_controller import MessageController
from models.chat_config import ChatConfig
from utils.config import config


class FileController:
    """Controller for handling file operations, primarily PDF processing"""

    def __init__(self, config_obj: ChatConfig, message_controller: MessageController):
        """
        Initialize the file controller

        Args:
            config_obj: Application configuration
            message_controller: Message controller for adding responses to history
        """
        self.config_obj = config_obj
        self.message_controller = message_controller

    def process_pdf_upload(self, uploaded_file) -> bool:
        """
        Process uploaded PDF file and extract text content

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Display user action in chat
            self._display_upload_message(uploaded_file.name)

            # Add user action to chat history
            self.message_controller.safe_add_message_to_history("user", f"📄 Uploaded PDF: {uploaded_file.name}")

            # Process the PDF file
            with st.spinner("🔍 Processing PDF..."):
                success, result = self._process_pdf_file(uploaded_file)

            if success:
                self._handle_successful_processing(uploaded_file.name, result)
                return True
            else:
                self._handle_processing_error(result)
                return False

        except Exception as e:
            error_msg = f"❌ **PDF Processing Error:** An unexpected error occurred while processing your PDF. Please try again."
            logging.error(f"Unexpected PDF processing error: {e}")
            self._display_error_message(error_msg)
            return False

    def _display_upload_message(self, filename: str):
        """Display upload message in chat UI"""
        with st.chat_message("user", avatar=self.config_obj.user_avatar):
            st.markdown(f"📄 **Uploaded PDF:** {filename}")

    def _process_pdf_file(self, uploaded_file) -> Tuple[bool, dict]:
        """
        Process the PDF file using the external service

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Tuple of (success: bool, result: dict)
        """
        # Create temporary file with configured suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=config.file_processing.PDF_TEMP_FILE_SUFFIX
        ) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            # Check file size limits
            file_size = os.path.getsize(temp_file_path)
            if file_size > config.file_processing.MAX_PDF_SIZE:
                return (
                    False,
                    {
                        "error": f"PDF file too large. Maximum size: {config.file_processing.MAX_PDF_SIZE // (1024*1024)}MB"
                    },
                )

            # Make request to PDF processing server using configured endpoint and timeout
            nvingest_endpoint = config.env.NVINGEST_ENDPOINT
            if not nvingest_endpoint:
                return False, {"error": "PDF processing service not configured (NVINGEST_ENDPOINT not set)"}

            with open(temp_file_path, 'rb') as pdf_file:
                files = {'file': pdf_file}
                response = requests.post(nvingest_endpoint, files=files, timeout=config.get_api_timeout("pdf"))

            # Check if request was successful
            response.raise_for_status()

            # Parse JSON response
            pdf_data = response.json()

            # Validate response structure
            if not isinstance(pdf_data, dict) or 'pages' not in pdf_data:
                return False, {"error": "Invalid response format from PDF processing server"}

            pages = pdf_data.get('pages', [])
            if not pages:
                return False, {"error": "No pages found in PDF response"}

            return True, pdf_data

        except requests.exceptions.RequestException as e:
            error_msg = f"Unable to connect to the PDF processing server. Please ensure the server is running at {config.env.NVINGEST_ENDPOINT}"
            logging.error(f"PDF processing request error: {e}")
            return False, {"error": error_msg}

        except requests.exceptions.Timeout:
            error_msg = f"The PDF processing took too long (timeout: {config.file_processing.PDF_PROCESSING_TIMEOUT}s). Please try with a smaller document."
            logging.error("PDF processing timeout")
            return False, {"error": error_msg}

        except ValueError as e:
            error_msg = str(e)
            logging.error(f"PDF processing validation error: {e}")
            return False, {"error": error_msg}

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    def _handle_successful_processing(self, filename: str, pdf_data: dict):
        """
        Handle successful PDF processing

        Args:
            filename: Name of the processed file
            pdf_data: Processed PDF data
        """
        pages = pdf_data.get('pages', [])

        # Display processing results
        with st.chat_message("assistant", avatar=self.config_obj.assistant_avatar):
            st.success(f"✅ Successfully processed PDF: **{filename}**")
            st.info(f"📊 Extracted text from **{len(pages)} pages**")
            st.markdown("📝 I can now answer questions about this document when you ask me about it!")

        # Create tool response for chat history (stores PDF data for later retrieval)
        tool_response = {
            "role": "tool",
            "content": json.dumps(
                {
                    "tool_name": "process_pdf_document",
                    "filename": filename,
                    "total_pages": len(pages),
                    "pages": pages,
                    "status": "success",
                }
            ),
        }

        # Add tool response to chat history (this stores the PDF data for the retrieval tool)
        # Use safe access with fallback
        if not hasattr(st.session_state, "messages"):
            st.session_state.messages = []
        st.session_state.messages.append(tool_response)

        # Add assistant confirmation message
        confirmation_msg = (
            f"I've successfully processed your PDF document **{filename}** and extracted text from "
            f"{len(pages)} pages. I can now answer questions about the document content when you ask me "
            f"about it. What would you like to know?"
        )

        self.message_controller.safe_add_message_to_history("assistant", confirmation_msg)
        logging.info(f"Successfully processed PDF: {filename} ({len(pages)} pages)")

    def _handle_processing_error(self, error_result: dict):
        """
        Handle PDF processing errors

        Args:
            error_result: Error information dictionary
        """
        error_msg = f"❌ **PDF Processing Error:** {error_result.get('error', 'Unknown error')}"
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
        return (
            not hasattr(st.session_state, 'last_uploaded_pdf')
            or st.session_state.last_uploaded_pdf != uploaded_file.name
        )

    def mark_file_as_processed(self, filename: str):
        """
        Mark a file as processed to prevent reprocessing

        Args:
            filename: Name of the processed file
        """
        st.session_state.last_uploaded_pdf = filename

    def get_supported_file_types(self) -> list:
        """
        Get list of supported file types for upload

        Returns:
            List of supported file extensions
        """
        return config.file_processing.SUPPORTED_PDF_TYPES

    def get_file_size_limit_mb(self) -> int:
        """
        Get the file size limit in megabytes

        Returns:
            File size limit in MB
        """
        return config.file_processing.MAX_PDF_SIZE // (1024 * 1024)
