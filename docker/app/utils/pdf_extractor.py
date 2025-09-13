"""
PDF Data Extractor Utility

This module provides a centralized way to extract PDF data from various sources,
eliminating duplicate code across multiple tools.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PDFDataExtractor:
    """Utility class for extracting PDF data from messages and other sources"""

    @staticmethod
    def extract_from_messages(
        messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract PDF data from message history

        This method searches through messages to find PDF data that was
        injected by tools or system messages.

        Args:
            messages: List of message dictionaries

        Returns:
            PDF data dictionary or None if not found
        """
        # Search for PDF data in reverse order (most recent first)
        for message in reversed(messages):
            content = message.get("content", "")

            # Handle system messages with PDF data
            if message.get("role") == "system":
                # First try to parse as JSON
                try:
                    data = json.loads(content)
                    if data.get("type") == "pdf_data" and data.get("pages"):
                        # Found PDF data in system message
                        logger.debug(
                            "Found PDF data in system message:"
                            f" {data.get('filename')}"
                        )
                        return {
                            "filename": data.get("filename", "Unknown"),
                            "pages": data.get("pages", []),
                            "pdf_id": data.get("pdf_id"),
                            "total_pages": data.get(
                                "total_pages", len(data.get("pages", []))
                            ),
                        }
                    # Check for batch-processed PDF
                    elif data.get("type") == "pdf_data" and data.get(
                        "batch_processed"
                    ):
                        logger.debug(
                            "Found batch-processed PDF in system message:"
                            f" {data.get('filename')}"
                        )
                        return {
                            "filename": data.get("filename", "Unknown"),
                            "pdf_id": data.get("pdf_id"),
                            "total_pages": data.get("total_pages", 0),
                            "batch_processed": True,
                            "total_batches": data.get("total_batches", 0),
                            "pages": [],  # Empty for batch-processed
                        }
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, try parsing as context message format
                    pass

                # Try to parse as context message format (PDF content with markers)
                if (
                    "---BEGIN PDF CONTENT---" in content
                    and "---END PDF CONTENT---" in content
                ):
                    pdf_data = PDFDataExtractor._parse_pdf_context_message(
                        content
                    )
                    if pdf_data:
                        logger.debug(
                            "Found PDF data in context message:"
                            f" {pdf_data.get('filename')}"
                        )
                        return pdf_data

        logger.debug("No PDF data found in system messages")
        return None

    @staticmethod
    def _parse_pdf_context_message(content: str) -> Optional[Dict[str, Any]]:
        """Parse PDF content from context message format"""
        try:
            lines = content.split("\n")
            filename = None
            pages = []
            in_pdf_content = False
            current_page = None
            current_page_text = []

            for line in lines:
                if "The user has uploaded a PDF document:" in line:
                    # Extract filename
                    match = re.search(r"'([^']+)'", line)
                    if match:
                        filename = match.group(1)
                elif line.strip() == "---BEGIN PDF CONTENT---":
                    in_pdf_content = True
                elif line.strip() == "---END PDF CONTENT---":
                    in_pdf_content = False
                    # Save last page if any
                    if current_page is not None and current_page_text:
                        pages.append(
                            {
                                "page": current_page,
                                "text": "\n".join(current_page_text).strip(),
                            }
                        )
                elif in_pdf_content:
                    if line.startswith("### Page "):
                        # Save previous page if any
                        if current_page is not None and current_page_text:
                            pages.append(
                                {
                                    "page": current_page,
                                    "text": (
                                        "\n".join(current_page_text).strip()
                                    ),
                                }
                            )
                        # Start new page
                        try:
                            current_page = int(
                                line.replace("### Page ", "").strip()
                            )
                            current_page_text = []
                        except ValueError:
                            pass
                    elif current_page is not None:
                        current_page_text.append(line)

            if pages:
                logger.info(
                    f"Extracted PDF data from context message: {filename} with"
                    f" {len(pages)} pages"
                )
                return {"filename": filename or "Unknown", "pages": pages}

        except Exception as e:
            logger.error(f"Error parsing PDF context message: {e}")

        return None

    @staticmethod
    def extract_text_from_pdf_data(
        pdf_data: Dict[str, Any], max_pages: Optional[int] = None
    ) -> Optional[str]:
        """
        Extract text content from PDF data

        Args:
            pdf_data: PDF data dictionary with pages
            max_pages: Maximum number of pages to extract (None for all)

        Returns:
            Extracted text or None
        """
        if not pdf_data:
            return None

        pages = pdf_data.get("pages", [])
        if not pages:
            return None

        document_text = []
        pages_to_process = pages[:max_pages] if max_pages else pages

        for page in pages_to_process:
            page_text = page.get("text", "")
            if page_text:
                document_text.append(
                    f"[Page {page.get('page', '?')}]\n{page_text}"
                )

        if document_text:
            return "\n\n".join(document_text)

        return None
