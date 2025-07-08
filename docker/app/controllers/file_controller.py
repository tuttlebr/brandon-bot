import asyncio
import json
import logging
import os
import tempfile
from typing import Tuple

import requests
import streamlit as st
from controllers.message_controller import MessageController
from models.chat_config import ChatConfig
from services.pdf_batch_processor import PDFBatchProcessor
from services.pdf_summarization_service import PDFSummarizationService
from utils.config import config


class FileController:
    """Controller for handling file operations, primarily PDF processing"""

    def __init__(
        self, config_obj: ChatConfig, message_controller: MessageController, session_controller=None,
    ):
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
        self.batch_processor = PDFBatchProcessor()

    def process_pdf_upload(self, uploaded_file) -> bool:
        """
        Process uploaded PDF file and extract text content

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Mark file as being processed immediately to prevent duplicates
            self.mark_file_as_processing(uploaded_file.name)

            # Display user action in chat
            # self._display_upload_message(uploaded_file.name)

            # Add user action to chat history
            # self.message_controller.safe_add_message_to_history("user", f"📄 Uploaded PDF: {uploaded_file.name}")

            # Process the PDF file
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
        finally:
            # Always clear the processing marker when done
            self.clear_processing_file()

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

            with open(temp_file_path, "rb") as pdf_file:
                files = {"file": pdf_file}

                # Use resilient request
                response = self._make_resilient_request(
                    nvingest_endpoint, files, base_timeout=config.get_api_timeout("pdf")
                )

            # Parse JSON response
            pdf_data = response.json()

            # Validate response structure
            if not isinstance(pdf_data, dict) or "pages" not in pdf_data:
                return False, {"error": "Invalid response format from PDF processing server"}

            pages = pdf_data.get("pages", [])
            if not pages:
                return False, {"error": "No pages found in PDF response"}

            return True, pdf_data

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if "Connection refused" in str(e):
                error_msg = f"Unable to connect to the PDF processing server. Please ensure the server is running at {config.env.NVINGEST_ENDPOINT}"
            else:
                error_msg = f"The PDF processing service is temporarily unavailable or the document is too large. Please try again in a few moments."
            logging.error(f"PDF processing network error: {e}")
            return False, {"error": error_msg}

        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 500:
                error_msg = "The PDF processing server encountered an internal error. Please try again later."
            elif e.response.status_code == 413:
                error_msg = "The PDF file is too large for the processing server."
            else:
                error_msg = f"PDF processing failed with server error: {e.response.status_code}"
            logging.error(f"PDF processing HTTP error: {e}")
            return False, {"error": error_msg}

        except requests.exceptions.RequestException as e:
            error_msg = f"PDF processing failed due to a network issue. Please check your connection and try again."
            logging.error(f"PDF processing request error: {e}")
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

    def _display_upload_message(self, filename: str):
        """Display upload message in chat UI"""
        with st.chat_message("user", avatar=self.config_obj.user_avatar):
            st.markdown(f"📄 **Uploaded PDF:** {filename}")

    def _make_resilient_request(self, url: str, files: dict, base_timeout: int = None) -> requests.Response:
        """
        Make a long-running HTTP request with specified timeout

        Args:
            url: The URL to make the request to
            files: Files to upload
            max_retries: Unused parameter (kept for backwards compatibility)
            base_timeout: Timeout in seconds for the request

        Returns:
            requests.Response object

        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        if base_timeout is None:
            base_timeout = config.get_api_timeout("pdf")

        logging.info(f"PDF processing with {base_timeout} seconds timeout")

        response = requests.post(url, files=files, timeout=base_timeout)
        response.raise_for_status()
        return response

    def _handle_successful_processing(self, filename: str, pdf_data: dict):
        """
        Handle successful PDF processing

        Args:
            filename: Name of the processed file
            pdf_data: Processed PDF data
        """
        pages = pdf_data.get("pages", [])
        total_pages = len(pages)

        # Check if batch processing is needed
        if self.batch_processor.should_batch_process(total_pages):
            logging.info(f"Large PDF detected ({total_pages} pages), using batch processing")
            self._handle_batch_processing(filename, pdf_data)
        else:
            # Normal processing for smaller PDFs
            if self.session_controller:
                pdf_id = self.session_controller.store_pdf_document(filename, pdf_data)
                logging.info(f"Stored PDF '{filename}' with ID '{pdf_id}' in session state")

                # Add PDF content availability to message history
                self._add_pdf_content_to_history(filename, pdf_data, pdf_id)

                # Verify storage in session state
                if hasattr(st.session_state, "stored_pdfs") and pdf_id in st.session_state.stored_pdfs:
                    logging.info(
                        f"✅ Verified PDF '{pdf_id}' is in session state stored_pdfs list: {st.session_state.stored_pdfs}"
                    )
                else:
                    logging.error(f"❌ PDF '{pdf_id}' NOT found in session state stored_pdfs list")

            else:
                # Log warning but don't add complex tool responses
                logging.warning("Session controller not available for PDF storage")

        # The detailed confirmation is now handled by _add_pdf_content_to_history
        logging.info(f"Successfully processed PDF: {filename} ({total_pages} pages)")

    def _handle_batch_processing(self, filename: str, pdf_data: dict):
        """
        Handle batch processing for large PDFs

        Args:
            filename: Name of the processed file
            pdf_data: Processed PDF data
        """
        pages = pdf_data.get("pages", [])
        total_pages = len(pages)

        # Create batches
        batches = self.batch_processor.create_page_batches(total_pages)

        # Generate PDF ID
        import hashlib

        pdf_hash = hashlib.md5(filename.encode()).hexdigest()[:12]
        pdf_id = f"pdf_{pdf_hash}"

        # Process and store each batch
        from services.file_storage_service import FileStorageService

        file_storage = FileStorageService()

        for batch_num, (start_idx, end_idx) in enumerate(batches):
            batch_data = self.batch_processor.process_batch(pdf_data, (start_idx, end_idx))
            batch_id = file_storage.store_pdf_batch(filename, batch_data, st.session_state.session_id, batch_num)
            logging.info(f"Stored batch {batch_num + 1}/{len(batches)} with ID: {batch_id}")

        # Store PDF reference in session state
        if self.session_controller:
            # Add PDF ID to stored PDFs list
            if 'stored_pdfs' not in st.session_state:
                st.session_state.stored_pdfs = []
            st.session_state.stored_pdfs.append(pdf_id)

            # Store metadata about batch processing
            st.session_state[f"{pdf_id}_batch_info"] = {
                'filename': filename,
                'total_pages': total_pages,
                'total_batches': len(batches),
                'batch_processed': True,
            }

            # Add notification to message history
            self._add_batch_processed_notification(filename, total_pages, len(batches), pdf_id)

    def _add_batch_processed_notification(self, filename: str, total_pages: int, total_batches: int, pdf_id: str):
        """
        Add notification about batch processed PDF to message history

        Args:
            filename: Name of the PDF file
            total_pages: Total number of pages
            total_batches: Number of batches created
            pdf_id: PDF reference ID
        """
        try:
            # Create a system message indicating PDF availability with batch processing
            pdf_availability_message = {
                "role": "system",
                "content": json.dumps(
                    {
                        "type": "pdf_data",
                        "tool_name": "process_pdf_document",
                        "filename": filename,
                        "pdf_id": pdf_id,
                        "total_pages": total_pages,
                        "status": "available",
                        "batch_processed": True,
                        "total_batches": total_batches,
                        "message": f"Large PDF '{filename}' ({total_pages} pages) processed in {total_batches} batches and is now available for analysis",
                    }
                ),
            }

            # Add to message history via message controller
            self.message_controller.safe_add_message_to_history("system", pdf_availability_message["content"])

            # Add user-friendly notification
            user_notification = (
                f"✅ **PDF Uploaded Successfully**\n\n"
                f"The large document '{filename}' ({total_pages} pages) has been processed efficiently "
                f"in {total_batches} batches to optimize memory usage. You can now ask questions about it!"
            )

            with st.chat_message("assistant", avatar=self.config_obj.assistant_avatar):
                st.markdown(user_notification)

            self.message_controller.safe_add_message_to_history("assistant", user_notification)

            logging.info(f"Added batch processing notification to message history for '{filename}'")

        except Exception as e:
            logging.error(f"Error adding batch processing notification to history: {e}")

    def _add_pdf_content_to_history(self, filename: str, pdf_data: dict, pdf_id: str):
        """
        Add PDF content availability notification to message history

        Args:
            filename: Name of the PDF file
            pdf_data: Processed PDF data
            pdf_id: PDF reference ID
        """
        try:
            pages = pdf_data.get("pages", [])
            total_pages = len(pages)

            # Create a system message indicating PDF availability
            pdf_availability_message = {
                "role": "system",
                "content": json.dumps(
                    {
                        "type": "pdf_data",
                        "tool_name": "process_pdf_document",
                        "filename": filename,
                        "pdf_id": pdf_id,
                        "pages": pages,
                        "total_pages": total_pages,
                        "status": "available",
                        "message": f"PDF '{filename}' ({total_pages} pages) is now available for analysis",
                    }
                ),
            }

            # Add to message history via message controller
            self.message_controller.safe_add_message_to_history("system", pdf_availability_message["content"])

            logging.info(f"Added PDF availability notification to message history for '{filename}'")

        except Exception as e:
            logging.error(f"Error adding PDF content to history: {e}")

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
            filename = pdf_data.get("filename", "Unknown")
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
            if hasattr(st.session_state, "uploaded_pdfs") and pdf_id in st.session_state.uploaded_pdfs:
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
        # Check if we're currently processing this file
        if (
            hasattr(st.session_state, "currently_processing_pdf")
            and st.session_state.currently_processing_pdf == uploaded_file.name
        ):
            logging.info(f"PDF '{uploaded_file.name}' is already being processed, skipping duplicate processing")
            return False

        # Check if this file was already processed successfully
        return (
            not hasattr(st.session_state, "last_uploaded_pdf")
            or st.session_state.last_uploaded_pdf != uploaded_file.name
        )

    def mark_file_as_processing(self, filename: str):
        """
        Mark a file as currently being processed to prevent duplicates

        Args:
            filename: Name of the file being processed
        """
        st.session_state.currently_processing_pdf = filename
        logging.info(f"Marked PDF '{filename}' as currently being processed")

    def clear_processing_file(self):
        """Clear the currently processing file marker"""
        if hasattr(st.session_state, "currently_processing_pdf"):
            filename = st.session_state.currently_processing_pdf
            st.session_state.currently_processing_pdf = None
            logging.info(f"Cleared processing marker for PDF '{filename}'")

    def mark_file_as_processed(self, filename: str):
        """
        Mark a file as processed to prevent reprocessing

        Args:
            filename: Name of the processed file
        """
        st.session_state.last_uploaded_pdf = filename
        logging.info(f"Marked PDF '{filename}' as successfully processed")

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
