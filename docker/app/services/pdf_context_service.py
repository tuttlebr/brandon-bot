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
        is_question = any(
            indicator in message_lower for indicator in question_indicators
        )

        if is_question and len(message_lower.split()) >= 3:
            for keyword in weak_pdf_keywords:
                if keyword in message_lower:
                    logger.debug(
                        f"Weak PDF keyword '{keyword}' found in question context"
                    )
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
                logger.debug(
                    f"Message contains contextual reference and is a question - injecting PDF context"
                )
                return True

        # Default: don't inject PDF context for general conversation
        logger.debug(f"Message '{user_message}' does not require PDF context")
        return False

    def should_inject_pdf_context_after_tool(self, user_message: str) -> bool:
        """
        Determine if PDF context should be injected after tool execution

        This method is more permissive than should_inject_pdf_context and is used
        after tool execution to ensure PDF content is available for follow-up responses.

        Args:
            user_message: The user's query

        Returns:
            True if PDF context should be injected
        """
        # Always inject if there's a PDF in session and the message isn't clearly non-PDF
        if not self.has_pdf_in_session():
            return False

        message_lower = user_message.lower()

        # Messages that clearly indicate NOT wanting PDF context (even after tools)
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
        ]

        # Check if message is clearly NOT about the PDF
        for indicator in non_pdf_indicators:
            if indicator in message_lower and len(message_lower.split()) <= 3:
                logger.debug(
                    f"Message '{user_message}' identified as non-PDF query after tool execution"
                )
                return False

        # After tool execution, be more permissive - inject context for most messages
        # unless they're clearly unrelated to the PDF
        logger.debug(
            f"After tool execution, injecting PDF context for message: '{user_message}'"
        )
        return True

    def has_pdf_in_session(self) -> bool:
        """
        Check if there's a PDF available in the current session

        Returns:
            True if PDF is available
        """
        has_pdfs = (
            hasattr(st.session_state, 'stored_pdfs')
            and len(st.session_state.stored_pdfs) > 0
        )
        logger.debug(
            f"has_pdf_in_session: {has_pdfs}, stored_pdfs: {getattr(st.session_state, 'stored_pdfs', [])}"
        )
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
            logger.info(
                f"Retrieved PDF data for: {pdf_data.get('filename', 'Unknown')}"
            )
            return pdf_data
        else:
            logger.error(f"Failed to retrieve PDF data for ID: {latest_pdf_id}")
            return None

    def _get_batch_processed_pdf(
        self, pdf_id: str, batch_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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

    def get_merged_batch_pdf(self, pdf_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a merged version of a batch-processed PDF (fallback method)

        Args:
            pdf_id: PDF reference ID

        Returns:
            Merged PDF data or None
        """
        try:
            merged_data = self.file_storage.merge_pdf_batches(pdf_id)
            if merged_data:
                logger.info(
                    f"Successfully merged {len(merged_data.get('pages', []))} pages from batches for PDF {pdf_id}"
                )
                return merged_data
            else:
                logger.warning(f"Failed to merge batches for PDF {pdf_id}")
                return None
        except Exception as e:
            logger.error(f"Error merging batches for PDF {pdf_id}: {e}")
            return None

    def inject_pdf_context(
        self, messages: List[Dict[str, Any]], user_message: str
    ) -> List[Dict[str, Any]]:
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
            logger.debug(
                f"PDF context injection not needed for query: '{user_message}'"
            )
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

        logger.info(
            f"Injected PDF context for '{pdf_data.get('filename')}' into conversation"
        )
        return enhanced_messages

    def _create_pdf_system_message(
        self, pdf_data: Dict[str, Any], user_query: str
    ) -> Dict[str, str]:
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
                "Please use this PDF content to answer the user's question. Be specific and cite page numbers when referencing information at the end of the document.",
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
        content_parts = [
            "Here is the complete content of the PDF document:",
            "",
            "---BEGIN PDF CONTENT---",
        ]

        current_chars = 0
        # Use reasonable character limit to fit within LLM context window
        max_chars = config.file_processing.PDF_CONTEXT_MAX_CHARS
        pages_included = 0

        # Process all pages and let the character limit handle truncation for context injection.
        # The analysis tools will process the full document regardless.
        total_pages = len(pages)
        pages_to_process = pages
        logger.info(
            f"Preparing to process all {total_pages} pages for regular PDF context."
        )

        for page in pages_to_process:
            page_num = page.get('page', pages_included + 1)
            page_text = page.get('text', '').strip()

            if page_text:
                # Check if adding this page would exceed our character limit
                if current_chars + len(page_text) > max_chars and pages_included > 0:
                    remaining_pages = len(pages_to_process) - pages_included
                    if remaining_pages > 0:
                        content_parts.append(
                            f"\n[... {remaining_pages} more pages available. The analysis tools will process the complete document ...]"
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
                f"\nNote: Showing {pages_included} of {len(pages)} pages for context. "
                f"The analysis tools will process the complete document."
            )

        return content_parts

    def _get_relevant_pages_content(
        self, pdf_data: Dict[str, Any], user_query: str
    ) -> List[str]:
        """
        Get relevant pages for a batch-processed PDF based on the user query

        Args:
            pdf_data: PDF metadata
            user_query: User's query

        Returns:
            List of content parts
        """
        content_parts = [
            "Loading relevant pages based on your query...",
            "",
            "---BEGIN PDF CONTENT---",
        ]

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

        # If no specific pages mentioned, load all pages for analysis.
        if not page_numbers:
            # For batch-processed PDFs, we load all pages for analysis.
            total_pages = pdf_data.get('total_pages', 0)
            page_numbers = list(range(1, total_pages + 1))


        # Remove duplicates and sort
        page_numbers = sorted(set(page_numbers))
        logger.info(f"Requesting pages: {page_numbers}")

        # Load the specific pages from batches
        loaded_pages = self._load_pages_from_batches(pdf_data['pdf_id'], page_numbers)

        if not loaded_pages:
            logger.warning(
                f"No pages loaded from batches for PDF {pdf_data['pdf_id']}, trying fallback merge"
            )

            # Try to get merged PDF as fallback
            merged_pdf = self.get_merged_batch_pdf(pdf_data['pdf_id'])
            if merged_pdf:
                logger.info("Using merged PDF as fallback")
                # Use the merged PDF data instead
                merged_pages = merged_pdf.get('pages', [])
                if merged_pages:
                    # Limit to requested pages if possible
                    if page_numbers:
                        # Filter to requested pages
                        loaded_pages = [
                            p for p in merged_pages if p.get('page', 0) in page_numbers
                        ]
                    else:
                        # Use the entire document for analysis
                        loaded_pages = merged_pages

            if not loaded_pages:
                content_parts.append(
                    "⚠️ **Warning:** Unable to load any pages from the batch-processed PDF."
                )
                content_parts.append("This could be due to:")
                content_parts.append("- Batch files not found or corrupted")
                content_parts.append("- Incorrect page number calculations")
                content_parts.append("- Storage issues")
                content_parts.append("")
                content_parts.append("Please try:")
                content_parts.append("1. Re-uploading the PDF")
                content_parts.append("2. Asking about specific page numbers")
                content_parts.append("3. Using the PDF summary tool instead")
                content_parts.append("---END PDF CONTENT---")
                return content_parts

        current_chars = 0
        # For batch-processed PDFs, be more permissive with character limits
        # since the analysis tools can handle complete documents through their own batch processing
        is_batch_processed = pdf_data.get('batch_processed', False)
        if is_batch_processed:
            # Allow more content for batch-processed PDFs since analysis tools handle complete documents
            max_chars = (
                config.file_processing.PDF_CONTEXT_MAX_CHARS * 2
            )  # Double the limit for batch-processed
            logger.info(
                f"Using increased character limit ({max_chars}) for batch-processed PDF"
            )
        else:
            # Use standard limit for regular PDFs
            max_chars = config.file_processing.PDF_CONTEXT_MAX_CHARS
        pages_included = 0

        for page in loaded_pages:
            page_num = page.get('page', pages_included + 1)
            page_text = page.get('text', '').strip()

            if page_text:
                if current_chars + len(page_text) > max_chars and pages_included > 0:
                    if is_batch_processed:
                        # For batch-processed PDFs, note that analysis tools will process the complete document
                        content_parts.append(
                            f"\n[... Additional pages available. The analysis tools will process the complete document through batch processing ...]"
                        )
                    else:
                        # For regular PDFs, use the standard truncation message
                        content_parts.append(
                            f"\n[... Additional pages available but truncated to optimize memory usage ...]"
                        )
                    break

                content_parts.append(f"\n### Page {page_num}")
                content_parts.append(page_text)
                content_parts.append("")

                current_chars += len(page_text)
                pages_included += 1

        content_parts.append("---END PDF CONTENT---")

        if page_numbers:
            content_parts.append(
                f"\nShowing pages: {', '.join(map(str, page_numbers[:10]))}"
            )
            if len(page_numbers) > 10:
                content_parts.append(f"and {len(page_numbers) - 10} more...")

            if pages_included < len(loaded_pages):
                if is_batch_processed:
                    content_parts.append(
                        f"\n**Note:** Loaded {pages_included} of {len(loaded_pages)} requested pages for context. The analysis tools will process the complete document through batch processing."
                    )
                else:
                    content_parts.append(
                        f"\n**Note:** Loaded {pages_included} of {len(loaded_pages)} requested pages due to character limits."
                    )

        return content_parts

    def _load_pages_from_batches(
        self, pdf_id: str, page_numbers: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Load specific pages from PDF batches

        Args:
            pdf_id: PDF reference ID
            page_numbers: List of page numbers to load (1-indexed)

        Returns:
            List of page data
        """
        logger.info(f"Loading pages {page_numbers} from batches for PDF {pdf_id}")

        # Get all batches for this PDF
        batches = self.file_storage.get_pdf_batches(pdf_id)
        logger.info(f"Found {len(batches)} batches for PDF {pdf_id}")

        if not batches:
            logger.warning(f"No batches found for PDF {pdf_id}")
            return []

        loaded_pages = []
        pages_per_batch = config.file_processing.PDF_PAGES_PER_BATCH

        for page_num in page_numbers:
            # Calculate which batch contains this page
            batch_idx = (page_num - 1) // pages_per_batch

            logger.debug(f"Page {page_num} should be in batch {batch_idx}")

            if batch_idx < len(batches):
                batch = batches[batch_idx]
                batch_pages = batch.get('pages', [])

                logger.debug(f"Batch {batch_idx} has {len(batch_pages)} pages")

                # Find the page within the batch
                page_idx_in_batch = (page_num - 1) % pages_per_batch

                logger.debug(
                    f"Page {page_num} should be at index {page_idx_in_batch} within batch {batch_idx}"
                )

                if page_idx_in_batch < len(batch_pages):
                    page_data = batch_pages[page_idx_in_batch]
                    # Ensure the page has the correct page number
                    if 'page' not in page_data:
                        page_data['page'] = page_num
                    loaded_pages.append(page_data)
                    logger.debug(f"Successfully loaded page {page_num}")
                else:
                    logger.warning(
                        f"Page index {page_idx_in_batch} out of range for batch {batch_idx} (batch has {len(batch_pages)} pages)"
                    )
            else:
                logger.warning(
                    f"Batch index {batch_idx} out of range (only {len(batches)} batches available)"
                )

        logger.info(f"Successfully loaded {len(loaded_pages)} pages from batches")
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

        return f"📄 Active document: {filename}"

    def debug_batch_processing(self, pdf_id: str) -> str:
        """
        Debug batch processing for a PDF

        Args:
            pdf_id: PDF reference ID

        Returns:
            Debug information string
        """
        debug_info = []
        debug_info.append(f"## Debug Information for PDF: {pdf_id}")
        debug_info.append("")

        # Check if PDF exists in session
        if not self.has_pdf_in_session():
            debug_info.append("❌ No PDFs found in session")
            return "\n".join(debug_info)

        # Get latest PDF data
        pdf_data = self.get_latest_pdf_data()
        if not pdf_data:
            debug_info.append("❌ No PDF data available")
            return "\n".join(debug_info)

        debug_info.append(f"✅ PDF found: {pdf_data.get('filename', 'Unknown')}")
        debug_info.append(f"📄 Total pages: {pdf_data.get('total_pages', 0)}")
        debug_info.append(
            f"🔄 Batch processed: {pdf_data.get('batch_processed', False)}"
        )

        if pdf_data.get('batch_processed'):
            debug_info.append(f"📦 Total batches: {pdf_data.get('total_batches', 0)}")

            # Try to load batches
            batches = self.file_storage.get_pdf_batches(pdf_id)
            debug_info.append(f"📁 Found {len(batches)} batch files")

            for i, batch in enumerate(batches):
                pages = batch.get('pages', [])
                debug_info.append(f"  Batch {i}: {len(pages)} pages")

                if pages:
                    first_page = pages[0]
                    last_page = pages[-1]
                    debug_info.append(
                        f"    Pages {first_page.get('page', 'unknown')} to {last_page.get('page', 'unknown')}"
                    )

        return "\n".join(debug_info)

    def inject_pdf_context_forced(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Force PDF context injection regardless of user message content

        This method is used when we know we need PDF context (e.g., after PDF tool execution)
        and don't want to check the user message content.

        Args:
            messages: Current conversation messages

        Returns:
            Messages with PDF context injected
        """
        # Check if we have a PDF in session
        if not self.has_pdf_in_session():
            logger.debug("No PDF in session for forced context injection")
            return messages

        # Get PDF data
        pdf_data = self.get_latest_pdf_data()
        if not pdf_data:
            logger.warning(
                "Forced PDF context injection requested but no PDF data available"
            )
            return messages

        # Create system message with PDF content using a generic query
        pdf_system_message = self._create_pdf_system_message(
            pdf_data, "the PDF document"
        )

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

        logger.info(f"Forced PDF context injection for '{pdf_data.get('filename')}'")
        return enhanced_messages
