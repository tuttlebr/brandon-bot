"""
PDF Batch Processing Service

This service handles processing of large PDFs in batches to avoid memory issues.
It splits PDFs into manageable chunks and processes them sequentially.
"""

import logging
from typing import Dict, List, Tuple

from utils.config import config

logger = logging.getLogger(__name__)


class PDFBatchProcessor:
    """Service for processing large PDFs in batches"""

    def __init__(self):
        """Initialize the PDF batch processor"""
        self.batch_threshold = config.file_processing.PDF_BATCH_PROCESSING_THRESHOLD
        self.pages_per_batch = config.file_processing.PDF_PAGES_PER_BATCH

    def should_batch_process(self, total_pages: int) -> bool:
        """
        Determine if a PDF should be processed in batches

        Args:
            total_pages: Total number of pages in the PDF

        Returns:
            True if batch processing should be used
        """
        return total_pages > self.batch_threshold

    def create_page_batches(self, total_pages: int) -> List[Tuple[int, int]]:
        """
        Create batches of page ranges for processing

        Args:
            total_pages: Total number of pages in the PDF

        Returns:
            List of tuples (start_page, end_page) for each batch
        """
        batches = []
        for i in range(0, total_pages, self.pages_per_batch):
            start_page = i
            end_page = min(i + self.pages_per_batch, total_pages)
            batches.append((start_page, end_page))

        logger.info(f"Created {len(batches)} batches for {total_pages} pages")
        return batches

    def process_batch(self, pdf_data: Dict, batch_range: Tuple[int, int]) -> Dict:
        """
        Process a specific batch of pages from PDF data

        Args:
            pdf_data: Complete PDF data
            batch_range: Tuple of (start_page, end_page)

        Returns:
            Processed batch data
        """
        start_idx, end_idx = batch_range
        pages = pdf_data.get('pages', [])

        # Extract batch of pages
        batch_pages = pages[start_idx:end_idx]

        # Create batch data structure
        batch_data = {
            'pages': batch_pages,
            'batch_info': {
                'start_page': start_idx + 1,  # Convert to 1-based indexing
                'end_page': end_idx,
                'total_pages': len(pages),
            },
        }

        logger.debug(f"Processing batch: pages {start_idx + 1} to {end_idx}")
        return batch_data
