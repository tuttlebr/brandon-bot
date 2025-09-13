import json
import logging
import os
import tempfile
from typing import Tuple

import requests
from models.chat_config import ChatConfig
from utils.config import config
from utils.text_processing import TextProcessor


class FileController:
    """
    Handles file upload requests and extracts text content using
    external services
    """

    def __init__(
        self,
        chat_config: ChatConfig,
        message_controller=None,
        session_controller=None,
    ):
        self.config = chat_config
        self.message_controller = message_controller
        self.session_controller = session_controller
        self.processing_files = set()
        self.processed_files = set()

    def normalize_filename(self, filename: str) -> str:
        """
        Normalize filename for consistency in tracking

        Args:
            filename: Original filename

        Returns:
            Normalized filename
        """
        # Remove path components if any
        filename = os.path.basename(filename)

        # Use TextProcessor to create a safe filename
        # This handles Unicode properly and creates ASCII-safe filenames
        normalized = TextProcessor.for_filename(filename)

        # Convert to lowercase for consistency in tracking
        normalized = normalized.lower()

        return normalized

    def mark_file_as_processing(self, filename: str):
        """Mark a file as being processed"""
        normalized = self.normalize_filename(filename)
        self.processing_files.add(normalized)

    def is_file_processing(self, filename: str) -> bool:
        """Check if a file is currently being processed"""
        normalized = self.normalize_filename(filename)
        return normalized in self.processing_files

    def clear_processing_file(self):
        """Clear all processing markers"""
        self.processing_files.clear()

    def process_pdf_upload(self, uploaded_file) -> Tuple[bool, dict]:
        """
        Process uploaded PDF file and extract text content using NVIngest

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            Tuple of (success: bool, result: dict) where result contains
            either the extracted PDF data or error information
        """
        try:
            # Mark file as being processed immediately to prevent duplicates
            self.mark_file_as_processing(uploaded_file.name)

            # Process the PDF file
            success, result = self._process_pdf_file(uploaded_file)

            if success:
                # Just return the extracted data
                return True, result
            else:
                return False, result

        except Exception as e:
            logging.error("Unexpected PDF processing error: %s", e)
            return False, {"error": str(e)}
        finally:
            # Always clear the processing marker when done
            self.clear_processing_file()

    def _process_pdf_file(self, uploaded_file) -> Tuple[bool, dict]:
        """
        Process the PDF file using the external NVIngest service

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
                        "error": (
                            "PDF file too large. Maximum size: "
                            f"{config.file_processing.MAX_PDF_SIZE // (1024*1024)}MB"
                        )
                    },
                )

            # Make request to PDF processing server using configured
            # endpoint and timeout
            nvingest_endpoint = config.env.NVINGEST_ENDPOINT
            if not nvingest_endpoint:
                return False, {
                    "error": (
                        "PDF processing service not configured "
                        "(NVINGEST_ENDPOINT not set)"
                    )
                }

            with open(temp_file_path, "rb") as pdf_file:
                files = {"file": pdf_file}

                # Use resilient request
                response = self._make_resilient_request(
                    nvingest_endpoint,
                    files,
                    base_timeout=config.get_api_timeout("pdf"),
                )

            # Parse JSON response
            pdf_data = response.json()

            # Validate response structure
            if not isinstance(pdf_data, dict) or "pages" not in pdf_data:
                return False, {
                    "error": (
                        "Invalid response format from PDF processing server"
                    )
                }

            pages = pdf_data.get("pages", [])
            if not pages:
                return False, {
                    "error": "No content could be extracted from the PDF"
                }

            # Calculate total text length
            total_chars = sum(len(page.get("text", "")) for page in pages)
            logging.info(
                "Successfully extracted %d pages (%d characters) from PDF",
                len(pages),
                total_chars,
            )

            return True, pdf_data

        except requests.exceptions.Timeout:
            return False, {
                "error": (
                    "PDF processing timed out. The file may be too large "
                    "or complex. Please try a smaller file."
                )
            }
        except requests.exceptions.RequestException as e:
            logging.error("Request error during PDF processing: %s", e)
            return False, {
                "error": (
                    "Failed to connect to PDF processing server. Please try"
                    " again later."
                )
            }
        except json.JSONDecodeError as e:
            logging.error("JSON decode error: %s", e)
            return False, {
                "error": "Invalid response from PDF processing server"
            }
        except Exception as e:
            logging.error("Unexpected error processing PDF: %s", e)
            return False, {"error": f"PDF processing failed: {str(e)}"}
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def _make_resilient_request(
        self, url: str, files: dict, base_timeout: int = None
    ) -> requests.Response:
        """
        Make a resilient request to external service with exponential backoff

        Args:
            url: The endpoint URL
            files: Files dictionary for the request
            base_timeout: Base timeout in seconds

        Returns:
            Response object

        Raises:
            Various requests exceptions if all retries fail
        """
        if base_timeout is None:
            base_timeout = config.get_api_timeout("pdf")

        # Simple request without retries for now
        response = requests.post(url, files=files, timeout=base_timeout)

        response.raise_for_status()
        return response

    def get_supported_file_types(self):
        """Get list of supported file types for upload"""
        return ["pdf"]

    def get_file_size_limit_mb(self):
        """Get file size limit in MB"""
        return config.file_processing.MAX_PDF_SIZE // (1024 * 1024)

    def is_new_upload(self, uploaded_file) -> bool:
        """Check if this is a new file that hasn't been processed yet"""
        if not uploaded_file:
            return False

        normalized = self.normalize_filename(uploaded_file.name)

        # Check if already processed
        if normalized in self.processed_files:
            return False

        # Check if currently processing
        if normalized in self.processing_files:
            return False

        return True

    def mark_file_as_processed(self, filename: str):
        """Mark a file as successfully processed"""
        normalized = self.normalize_filename(filename)
        self.processed_files.add(normalized)
        # Remove from processing set if present
        self.processing_files.discard(normalized)
