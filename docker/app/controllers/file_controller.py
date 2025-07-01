import asyncio
import json
import logging
import os
import tempfile
import threading
from typing import List, Optional, Tuple

import requests
import streamlit as st
from controllers.message_controller import MessageController
from models.chat_config import ChatConfig
from services.pdf_summarization_service import PDFSummarizationService
from utils.config import config


class FileController:
    """Controller for handling file operations, primarily PDF processing"""

    def __init__(self, config_obj: ChatConfig, message_controller: MessageController, session_controller=None):
        """
        Initialize the file controller

        Args:
            config_obj: Application configuration
            message_controller: Message controller for adding responses to history
            session_controller: Session controller for PDF storage
        """
        self.config_obj = config_obj
        self.message_controller = message_controller
        self.session_controller = session_controller
        self.pdf_summarization_service = PDFSummarizationService(config_obj)

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
        summarization_threshold = config.file_processing.PDF_SUMMARIZATION_THRESHOLD
        summarization_enabled = config.file_processing.PDF_SUMMARIZATION_ENABLED

        # Display processing results
        with st.chat_message("assistant", avatar=self.config_obj.assistant_avatar):
            st.success(f"✅ Successfully processed PDF: **{filename}**")
            st.info(f"📊 Extracted text from **{len(pages)} pages**")

            # For large documents, mention summarization capability
            if len(pages) > summarization_threshold:
                st.markdown("💡 This is a large document. You can ask me to 'summarize the PDF' for a quick overview!")

            st.markdown("📝 I can now answer questions about this document!")

        # Store PDF data in session state via session controller
        if self.session_controller:
            pdf_id = self.session_controller.store_pdf_document(filename, pdf_data)
            logging.info(f"Stored PDF '{filename}' with ID '{pdf_id}' in session state")

            # Add PDF content directly to message history as a tool response
            self._add_pdf_content_to_history(filename, pdf_data)

            # No automatic summarization - this is now user-driven

        else:
            # Log warning but don't add complex tool responses
            logging.warning("Session controller not available for PDF storage")

        # The detailed confirmation is now handled by _add_pdf_content_to_history
        logging.info(f"Successfully processed PDF: {filename} ({len(pages)} pages)")

    def _run_async_summarization(self, pdf_id: str, pdf_data: dict):
        """
        Run async summarization in a new event loop in a background thread

        This method creates a new event loop in a background thread to handle
        async operations, which is necessary because Streamlit runs in a
        synchronous context without an event loop.

        Args:
            pdf_id: ID of the stored PDF
            pdf_data: PDF data to summarize

        Note:
            This runs in a daemon thread, so it will be terminated when the
            main program exits. Session state updates need to be handled
            carefully as Streamlit's session state is not thread-safe.
        """
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the async summarization
            loop.run_until_complete(self._async_summarize_pdf(pdf_id, pdf_data))

        except Exception as e:
            logging.error(f"Error in background summarization thread: {e}")
        finally:
            loop.close()

    async def _async_summarize_pdf(self, pdf_id: str, pdf_data: dict):
        """
        Asynchronously summarize PDF for large documents

        Args:
            pdf_id: ID of the stored PDF
            pdf_data: PDF data to summarize
        """
        try:
            filename = pdf_data.get('filename', 'Unknown')
            logging.info(f"Starting async summarization for PDF: {filename}")

            # Perform recursive summarization
            enhanced_pdf_data = await self.pdf_summarization_service.summarize_pdf_recursive(pdf_data)

            # Update the stored PDF data with summaries
            # Note: We need to be careful with session state in background threads
            if self.session_controller:
                try:
                    # Instead of directly accessing session state from thread,
                    # we'll use a thread-safe approach
                    self._update_pdf_with_summary(pdf_id, enhanced_pdf_data)

                    # Notify in chat that summarization is complete
                    summary_complete_msg = (
                        f"✨ Document summary complete for **{filename}**! "
                        f"I now have a comprehensive understanding of the document's content, "
                        f"which will help me respond more quickly to your questions.\n\n"
                        f"💡 You can ask me to 'show the summary of the document' to see the AI-generated overview."
                    )
                    self.message_controller.safe_add_message_to_history("assistant", summary_complete_msg)

                except Exception as e:
                    logging.error(f"Error updating PDF data with summary: {e}")

        except Exception as e:
            logging.error(f"Error in async PDF summarization: {e}")

    def _update_pdf_with_summary(self, pdf_id: str, enhanced_pdf_data: dict):
        """
        Thread-safe method to update PDF data with summary

        Args:
            pdf_id: ID of the PDF to update
            enhanced_pdf_data: PDF data with summaries
        """
        try:
            # This method should be called from the main thread or use proper synchronization
            # For now, we'll log the update and let the session controller handle it
            # when it's safe to do so
            logging.info(f"PDF summarization complete for ID: {pdf_id}")

            # Store the enhanced data in a way that's safe for the session controller to pick up
            # You could use a queue, file, or database for thread-safe communication
            # For simplicity, we'll just log it for now

            # In a production system, you might want to:
            # 1. Use a thread-safe queue to communicate back to the main thread
            # 2. Store the summary in a database that the main thread can poll
            # 3. Use Redis or another external store

            # For now, we'll update if we can safely access the session state
            if hasattr(st.session_state, 'uploaded_pdfs') and pdf_id in st.session_state.uploaded_pdfs:
                st.session_state.uploaded_pdfs[pdf_id] = enhanced_pdf_data
                logging.info(f"Updated PDF '{enhanced_pdf_data.get('filename')}' with summarization data")
            else:
                logging.warning(f"Could not update PDF {pdf_id} - session state not accessible")

        except Exception as e:
            logging.error(f"Error in _update_pdf_with_summary: {e}")

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

    def _add_pdf_content_to_history(self, filename: str, pdf_data: dict):
        """
        Add a simple confirmation message to history

        Args:
            filename: Name of the PDF file
            pdf_data: Processed PDF data
        """
        pages = pdf_data.get('pages', [])

        # Just add a confirmation message - the actual PDF content will be injected automatically
        confirmation_msg = (
            f"✅ I've successfully loaded your PDF document **{filename}** ({len(pages)} pages). "
            f"The full document content is now available for me to reference when answering your questions.\n\n"
        )

        # Add tips for large documents
        if len(pages) > 10:
            confirmation_msg += (
                f"💡 **Tips for working with this document:**\n"
                f"- Ask me to 'summarize the document' for a quick overview\n"
                f"- Request specific information like 'What does the document say about X?'\n"
                f"- Ask about specific pages or sections\n"
                f"- I can translate, proofread, or rewrite sections\n\n"
            )

        confirmation_msg += "What would you like to know about this document?"

        self.message_controller.safe_add_message_to_history("assistant", confirmation_msg)
        logging.info(f"PDF '{filename}' is now available for automatic context injection")
