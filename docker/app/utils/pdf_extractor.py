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
    def extract_from_messages(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract PDF data from injected system messages

        Args:
            messages: List of conversation messages

        Returns:
            Extracted PDF data dictionary or None
        """
        if not messages:
            logger.debug("No messages provided to extract PDF data from")
            return None

        # Look for PDF content in system messages
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, str) and "## PDF Document Context" in content:
                    return PDFDataExtractor._parse_pdf_context_message(content)
                elif isinstance(content, str) and PDFDataExtractor._is_json_pdf_data(content):
                    return PDFDataExtractor._parse_json_pdf_data(content)

        logger.debug("No PDF data found in system messages")
        return None

    @staticmethod
    def _is_json_pdf_data(content: str) -> bool:
        """Check if content is JSON-formatted PDF data"""
        if not content.strip().startswith("{") or not content.strip().endswith("}"):
            return False

        try:
            data = json.loads(content)
            return (
                isinstance(data, dict)
                and data.get("type") == "pdf_data"
                and data.get("tool_name") == "process_pdf_document"
            )
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def _parse_json_pdf_data(content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON-formatted PDF data"""
        try:
            data = json.loads(content)
            pages = data.get("pages", [])
            filename = data.get("filename", "Unknown")

            if pages:
                logger.info(f"Extracted PDF data from JSON: {filename} with {len(pages)} pages")
                return {"filename": filename, "pages": pages}
        except Exception as e:
            logger.error(f"Error parsing JSON PDF data: {e}")

        return None

    @staticmethod
    def _parse_pdf_context_message(content: str) -> Optional[Dict[str, Any]]:
        """Parse PDF content from context message format"""
        try:
            lines = content.split('\n')
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
                        pages.append({"page": current_page, "text": '\n'.join(current_page_text).strip()})
                elif in_pdf_content:
                    if line.startswith("### Page "):
                        # Save previous page if any
                        if current_page is not None and current_page_text:
                            pages.append({"page": current_page, "text": '\n'.join(current_page_text).strip()})
                        # Start new page
                        try:
                            current_page = int(line.replace("### Page ", "").strip())
                            current_page_text = []
                        except ValueError:
                            pass
                    elif current_page is not None:
                        current_page_text.append(line)

            if pages:
                logger.info(f"Extracted PDF data from context message: {filename} with {len(pages)} pages")
                return {"filename": filename or "Unknown", "pages": pages}

        except Exception as e:
            logger.error(f"Error parsing PDF context message: {e}")

        return None

    @staticmethod
    def extract_text_from_pdf_data(pdf_data: Dict[str, Any], max_pages: Optional[int] = None) -> Optional[str]:
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
                document_text.append(f"[Page {page.get('page', '?')}]\n{page_text}")

        if document_text:
            return "\n\n".join(document_text)

        return None
