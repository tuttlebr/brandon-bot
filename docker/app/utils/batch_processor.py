"""
Batch Processing Utility

This module provides a reusable batch processing framework for handling
large documents and datasets efficiently.
"""

from typing import Any, Dict, List, Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Specialized processor for document analysis tasks"""

    @staticmethod
    def categorize_document_size(total_pages: int) -> str:
        """
        Categorize document by size

        Args:
            total_pages: Number of pages in document

        Returns:
            Size category: "small", "medium", or "large"
        """
        if total_pages <= 5:
            return "small"
        elif total_pages <= 15:
            return "medium"
        else:
            return "large"

    @staticmethod
    def format_pages_for_analysis(
        pages: List[Dict[str, Any]], max_chars_per_page: Optional[int] = None
    ) -> str:
        """
        Format pages into text suitable for analysis

        Args:
            pages: List of page dictionaries
            max_chars_per_page: Optional limit on characters per page

        Returns:
            Formatted text string
        """
        formatted_pages = []

        for i, page in enumerate(pages):
            page_num = page.get("page")
            if page_num is None:
                page_num = i + 1
            page_text = page.get("text", "")

            if max_chars_per_page and len(page_text) > max_chars_per_page:
                page_text = page_text[:max_chars_per_page] + "..."

            formatted_pages.append(f"Page {page_num}:\n{page_text}")

        return "\n\n".join(formatted_pages)
