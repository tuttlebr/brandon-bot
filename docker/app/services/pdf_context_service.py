"""
PDF Context Service

This service handles automatic injection of PDF content into LLM context.
It ensures PDF content is ALWAYS available when users ask questions about documents.
"""

import logging
from typing import Any, Dict, List, Optional

import streamlit as st
from models.chat_config import ChatConfig
from utils.config import config

logger = logging.getLogger(__name__)


class PDFContextService:
    """Service for managing PDF context injection into LLM conversations"""

    def __init__(self, config: ChatConfig):
        """
        Initialize PDF context service

        Args:
            config: Application configuration
        """
        self.config = config
        # Import here to avoid circular imports
        from services.file_storage_service import FileStorageService

        self.file_storage = FileStorageService()

    def should_inject_pdf_context(self, user_message: str) -> bool:
        """
        Determine if PDF context should be injected based on user message

        Args:
            user_message: The user's query

        Returns:
            True if PDF context should be injected
        """
        # Always inject if there's a PDF in session
        if not self.has_pdf_in_session():
            return False

        message_lower = user_message.lower()

        # Messages that clearly indicate NOT wanting PDF context
        non_pdf_indicators = [
            'thanks',
            'thank you',
            'goodbye',
            'bye',
            'hello',
            'hi',
            'how are you',
            'what\'s up',
            'weather',
            'news',
            'image',
            'picture',
            'generate',
            'create',
            'draw',
        ]

        # Check if message is clearly NOT about the PDF
        for indicator in non_pdf_indicators:
            if indicator in message_lower and len(message_lower.split()) <= 5:
                logger.debug(f"Message '{user_message}' identified as non-PDF query")
                return False

        # Keywords that strongly indicate PDF-related questions
        strong_pdf_keywords = [
            'pdf',
            'document',
            'file',
            'paper',
            'uploaded',
            'page',
            'summary',
            'summarize',
        ]

        # Keywords that might indicate PDF-related questions (need context)
        weak_pdf_keywords = [
            'text',
            'content',
            'says',
            'mentions',
            'according to',
            'in the',
            'what does',
            'find',
            'search',
            'quote',
            'section',
            'chapter',
            'paragraph',
            'analyze',
            'explain',
            'describe',
        ]

        # Check for strong PDF keywords
        for keyword in strong_pdf_keywords:
            if keyword in message_lower:
                logger.debug(f"Strong PDF keyword '{keyword}' found in message")
                return True

        # For weak keywords, only inject if the message seems to be asking a question
        # and is substantial enough (not just "explain" or "what")
        question_indicators = [
            'what',
            'how',
            'why',
            'when',
            'where',
            'who',
            'which',
            'can you',
            'could you',
            'would you',
            '?',
        ]
        is_question = any(indicator in message_lower for indicator in question_indicators)

        if is_question and len(message_lower.split()) >= 3:
            for keyword in weak_pdf_keywords:
                if keyword in message_lower:
                    logger.debug(f"Weak PDF keyword '{keyword}' found in question context")
                    return True

        # For very short messages, default to NOT injecting PDF context
        # unless they contain strong PDF keywords (already checked above)
        if len(message_lower.split()) < 3:
            logger.debug(f"Short message '{user_message}' - not injecting PDF context")
            return False

        # For ambiguous medium-length messages, check if they seem to be continuing
        # a conversation about the PDF (this is a conservative approach)
        # Only inject if there are clear contextual clues
        if 3 <= len(message_lower.split()) < 10:
            # Check for pronouns or references that might refer to the document
            contextual_references = ['it', 'this', 'that', 'the above', 'the text']
            has_reference = any(ref in message_lower for ref in contextual_references)

            if has_reference and is_question:
                logger.debug(f"Message contains contextual reference and is a question - injecting PDF context")
                return True

        # Default: don't inject PDF context for general conversation
        logger.debug(f"Message '{user_message}' does not require PDF context")
        return False

    def has_pdf_in_session(self) -> bool:
        """
        Check if there's a PDF available in the current session

        Returns:
            True if PDF is available
        """
        has_pdfs = hasattr(st.session_state, 'stored_pdfs') and len(st.session_state.stored_pdfs) > 0
        logger.debug(f"has_pdf_in_session: {has_pdfs}, stored_pdfs: {getattr(st.session_state, 'stored_pdfs', [])}")
        return has_pdfs

    def get_latest_pdf_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recently uploaded PDF data

        Returns:
            PDF data dictionary or None
        """
        if not self.has_pdf_in_session():
            return None

        # Get the latest PDF ID
        latest_pdf_id = st.session_state.stored_pdfs[-1]

        # Check if this is a batch-processed PDF
        batch_info_key = f"{latest_pdf_id}_batch_info"
        if hasattr(st.session_state, batch_info_key):
            batch_info = getattr(st.session_state, batch_info_key)
            if batch_info.get('batch_processed', False):
                # Handle batch-processed PDF
                return self._get_batch_processed_pdf(latest_pdf_id, batch_info)

        # Regular PDF
        pdf_data = self.file_storage.get_pdf(latest_pdf_id)

        if pdf_data:
            logger.info(f"Retrieved PDF data for: {pdf_data.get('filename', 'Unknown')}")
            return pdf_data
        else:
            logger.error(f"Failed to retrieve PDF data for ID: {latest_pdf_id}")
            return None

    def _get_batch_processed_pdf(self, pdf_id: str, batch_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get batch-processed PDF data

        Args:
            pdf_id: PDF reference ID
            batch_info: Batch processing information

        Returns:
            PDF data with batch information
        """
        # For batch-processed PDFs, we don't load all batches at once
        # Instead, we return metadata that allows lazy loading
        return {
            'pdf_id': pdf_id,
            'filename': batch_info.get('filename', 'Unknown'),
            'total_pages': batch_info.get('total_pages', 0),
            'batch_processed': True,
            'total_batches': batch_info.get('total_batches', 0),
            'pages': [],  # Empty for now, will be loaded on demand
        }

    def inject_pdf_context(self, messages: List[Dict[str, Any]], user_message: str) -> List[Dict[str, Any]]:
        """
        Inject PDF context into messages for LLM processing

        Args:
            messages: Current conversation messages
            user_message: The user's current query

        Returns:
            Messages with PDF context injected
        """
        # Check if we should inject PDF context
        if not self.should_inject_pdf_context(user_message):
            logger.debug(f"PDF context injection not needed for query: '{user_message}'")
            return messages

        logger.info(f"PDF context injection triggered for query: '{user_message}'")

        # Get PDF data
        pdf_data = self.get_latest_pdf_data()
        if not pdf_data:
            logger.warning("PDF context injection requested but no PDF data available")
            return messages

        # Create system message with PDF content
        pdf_system_message = self._create_pdf_system_message(pdf_data, user_message)

        # Create a new message list with PDF context injected
        enhanced_messages = []

        # First, add any existing system messages
        for msg in messages:
            if msg.get("role") == "system":
                enhanced_messages.append(msg)

        # Add our PDF context system message
        enhanced_messages.append(pdf_system_message)

        # Add all other messages
        for msg in messages:
            if msg.get("role") != "system":
                enhanced_messages.append(msg)

        logger.info(f"Injected PDF context for '{pdf_data.get('filename')}' into conversation")
        return enhanced_messages

    def _create_pdf_system_message(self, pdf_data: Dict[str, Any], user_query: str) -> Dict[str, str]:
        """
        Create a system message containing PDF content

        Args:
            pdf_data: The PDF data to inject
            user_query: The user's query for context

        Returns:
            System message with PDF content
        """
        # Ensure we have a proper filename
        filename = pdf_data.get('filename', '')
        if not filename or filename == 'Unknown':
            # Try to get from metadata
            filename = pdf_data.get('pdf_id', 'document').replace('pdf_', '') + '.pdf'

        total_pages = pdf_data.get('total_pages', len(pdf_data.get('pages', [])))
        is_batch_processed = pdf_data.get('batch_processed', False)

        # Start building the system message
        content_parts = [
            f"## PDF Document Context",
            f"The user has uploaded a PDF document: '{filename}' ({total_pages} pages)",
            f"",
            f"The user is asking: {user_query}",
            f"",
        ]

        if is_batch_processed:
            # For batch-processed PDFs, load only relevant pages
            content_parts.extend(self._get_relevant_pages_content(pdf_data, user_query))
        else:
            # For regular PDFs, use the existing approach with better limits
            content_parts.extend(self._get_regular_pdf_content(pdf_data))

        content_parts.extend(
            [
                "",
                "Please use this PDF content to answer the user's question. Be specific and cite page numbers when referencing information.",
            ]
        )

        return {"role": "system", "content": "\n".join(content_parts)}

    def _get_regular_pdf_content(self, pdf_data: Dict[str, Any]) -> List[str]:
        """
        Get content from a regular (non-batch) PDF with improved limits

        Args:
            pdf_data: PDF data

        Returns:
            List of content parts
        """
        pages = pdf_data.get('pages', [])
        content_parts = ["Here is the complete content of the PDF document:", "", "---BEGIN PDF CONTENT---"]

        # Use configured limits instead of half of MAX_CONTEXT_LENGTH
        max_chars = config.file_processing.PDF_CONTEXT_MAX_CHARS
        max_pages = config.file_processing.PDF_CONTEXT_MAX_PAGES
        current_chars = 0
        pages_included = 0

        for page in pages[:max_pages]:  # Limit pages upfront
            page_num = page.get('page', pages_included + 1)
            page_text = page.get('text', '').strip()

            if page_text:
                # Check if adding this page would exceed our character limit
                if current_chars + len(page_text) > max_chars and pages_included > 0:
                    remaining_pages = len(pages) - pages_included
                    if remaining_pages > 0:
                        content_parts.append(
                            f"\n[... {remaining_pages} more pages available but not included to optimize memory usage ...]"
                        )
                    break

                content_parts.append(f"\n### Page {page_num}")
                content_parts.append(page_text)
                content_parts.append("")  # Empty line between pages

                current_chars += len(page_text)
                pages_included += 1

        content_parts.append("---END PDF CONTENT---")

        if pages_included < len(pages):
            content_parts.append(
                f"\nNote: Showing {pages_included} of {len(pages)} pages. "
                f"For specific information from other pages, please mention the page numbers in your question."
            )

        return content_parts

    def _get_relevant_pages_content(self, pdf_data: Dict[str, Any], user_query: str) -> List[str]:
        """
        Get relevant pages for a batch-processed PDF based on the user query

        Args:
            pdf_data: PDF metadata
            user_query: User's query

        Returns:
            List of content parts
        """
        content_parts = ["Loading relevant pages based on your query...", "", "---BEGIN PDF CONTENT---"]

        # Extract page numbers mentioned in the query
        import re

        page_numbers = []

        # Look for patterns like "page 5", "pages 3-7", "p. 10", etc.
        page_patterns = [
            r'page[s]?\s+(\d+)(?:\s*-\s*(\d+))?',
            r'p\.\s*(\d+)(?:\s*-\s*(\d+))?',
            r'pg\s+(\d+)(?:\s*-\s*(\d+))?',
        ]

        query_lower = user_query.lower()
        for pattern in page_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if match[1]:  # Range
                    start, end = int(match[0]), int(match[1])
                    page_numbers.extend(range(start, end + 1))
                else:  # Single page
                    page_numbers.append(int(match[0]))

        # If no specific pages mentioned, load first few pages and any with keywords
        if not page_numbers:
            # Default to first few pages
            page_numbers = list(range(1, min(11, pdf_data.get('total_pages', 10) + 1)))

            # TODO: In a more advanced implementation, we could:
            # 1. Use embeddings to find relevant pages
            # 2. Search for keywords across batches
            # 3. Use the PDF summarization to identify relevant sections

        # Remove duplicates and sort
        page_numbers = sorted(set(page_numbers))

        # Load the specific pages from batches
        loaded_pages = self._load_pages_from_batches(pdf_data['pdf_id'], page_numbers)

        current_chars = 0
        max_chars = config.file_processing.PDF_CONTEXT_MAX_CHARS

        for page in loaded_pages:
            page_num = page.get('page', 1)
            page_text = page.get('text', '').strip()

            if page_text:
                if current_chars + len(page_text) > max_chars:
                    content_parts.append(
                        f"\n[... Additional pages available but truncated to optimize memory usage ...]"
                    )
                    break

                content_parts.append(f"\n### Page {page_num}")
                content_parts.append(page_text)
                content_parts.append("")

                current_chars += len(page_text)

        content_parts.append("---END PDF CONTENT---")

        if page_numbers:
            content_parts.append(f"\nShowing pages: {', '.join(map(str, page_numbers[:10]))}")
            if len(page_numbers) > 10:
                content_parts.append(f"and {len(page_numbers) - 10} more...")

        return content_parts

    def _load_pages_from_batches(self, pdf_id: str, page_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Load specific pages from PDF batches

        Args:
            pdf_id: PDF reference ID
            page_numbers: List of page numbers to load (1-indexed)

        Returns:
            List of page data
        """
        # Get all batches for this PDF
        batches = self.file_storage.get_pdf_batches(pdf_id)

        loaded_pages = []
        pages_per_batch = config.file_processing.PDF_PAGES_PER_BATCH

        for page_num in page_numbers:
            # Calculate which batch contains this page
            batch_idx = (page_num - 1) // pages_per_batch

            if batch_idx < len(batches):
                batch = batches[batch_idx]
                batch_pages = batch.get('pages', [])

                # Find the page within the batch
                page_idx_in_batch = (page_num - 1) % pages_per_batch
                if page_idx_in_batch < len(batch_pages):
                    loaded_pages.append(batch_pages[page_idx_in_batch])

        return loaded_pages

    def get_pdf_info_for_display(self) -> Optional[str]:
        """
        Get a brief info string about the current PDF for UI display

        Returns:
            Info string or None
        """
        pdf_data = self.get_latest_pdf_data()
        if not pdf_data:
            return None

        filename = pdf_data.get('filename', 'Unknown')
        total_pages = len(pdf_data.get('pages', []))

        return f"📄 Active document: {filename} ({total_pages} pages)"
