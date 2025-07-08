"""
Batch Processing Utility

This module provides a reusable batch processing framework for handling
large documents and datasets efficiently.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BatchProcessor:
    """Generic batch processor for handling large datasets"""

    def __init__(self, batch_size: int = 5, delay_between_batches: float = 0.3):
        """
        Initialize batch processor

        Args:
            batch_size: Number of items to process in each batch
            delay_between_batches: Delay in seconds between batch processing
        """
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches

    async def process_in_batches(
        self,
        items: List[T],
        process_func: Callable[[List[T], int, int], Any],
        combine_func: Optional[Callable[[List[Any]], Any]] = None,
    ) -> Any:
        """
        Process items in batches with optional result combination

        Args:
            items: List of items to process
            process_func: Async function to process a batch (batch, start_idx, end_idx)
            combine_func: Optional function to combine all batch results

        Returns:
            Combined results or list of batch results
        """
        results = []
        total_items = len(items)

        for i in range(0, total_items, self.batch_size):
            batch_end = min(i + self.batch_size, total_items)
            batch = items[i:batch_end]

            logger.debug(f"Processing batch {i+1}-{batch_end} of {total_items}")

            try:
                result = await process_func(batch, i, batch_end)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch {i+1}-{batch_end}: {e}")
                results.append(None)

            # Add delay between batches (except for the last one)
            if batch_end < total_items:
                await asyncio.sleep(self.delay_between_batches)

        # Combine results if function provided
        if combine_func:
            return combine_func(results)

        return results


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
            page_num = page.get('page')
            if page_num is None:
                page_num = i + 1
            page_text = page.get('text', '')

            if max_chars_per_page and len(page_text) > max_chars_per_page:
                page_text = page_text[:max_chars_per_page] + "..."

            formatted_pages.append(f"Page {page_num}:\n{page_text}")

        return "\n\n".join(formatted_pages)

    @staticmethod
    def extract_page_numbers_from_text(
        text: str, valid_range: Optional[tuple] = None
    ) -> List[int]:
        """
        Extract page numbers mentioned in text

        Args:
            text: Text to search for page numbers
            valid_range: Optional (start, end) tuple for valid page numbers

        Returns:
            List of unique page numbers found
        """
        import re

        if 'none' in text.lower() or 'no pages' in text.lower():
            return []

        # Find all numbers that could be page numbers
        numbers = re.findall(r'\b(\d+)\b', text)
        page_numbers = []

        for num_str in numbers:
            num = int(num_str)

            # Check if in valid range
            if valid_range:
                start, end = valid_range
                if start <= num <= end:
                    page_numbers.append(num)
            else:
                page_numbers.append(num)

        return list(set(page_numbers))  # Remove duplicates
