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

        # Keywords that indicate PDF-related questions
        pdf_keywords = [
            'pdf',
            'document',
            'file',
            'paper',
            'text',
            'page',
            'summary',
            'summarize',
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
        ]

        message_lower = user_message.lower()

        # Check for any PDF-related keywords
        for keyword in pdf_keywords:
            if keyword in message_lower:
                return True

        # If the message is short and doesn't specify what to do,
        # assume it's about the PDF if one was recently uploaded
        if len(message_lower.split()) < 10:
            return True

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
        pdf_data = self.file_storage.get_pdf(latest_pdf_id)

        if pdf_data:
            logger.info(f"Retrieved PDF data for: {pdf_data.get('filename', 'Unknown')}")
            return pdf_data
        else:
            logger.error(f"Failed to retrieve PDF data for ID: {latest_pdf_id}")
            return None

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

        pages = pdf_data.get('pages', [])
        total_pages = len(pages)

        # Start building the system message
        content_parts = [
            f"## PDF Document Context",
            f"The user has uploaded a PDF document: '{filename}' ({total_pages} pages)",
            f"",
            f"The user is asking: {user_query}",
            f"",
            f"Here is the complete content of the PDF document:",
            f"",
            "---BEGIN PDF CONTENT---",
        ]

        # Add page content (with smart truncation for very large documents)
        max_chars = config.llm.MAX_CONTEXT_LENGTH // 2  # Use half of max context for PDF
        current_chars = 0
        pages_included = 0

        for page in pages:
            page_num = page.get('page', pages_included + 1)
            page_text = page.get('text', '').strip()

            if page_text:
                # Check if adding this page would exceed our limit
                if current_chars + len(page_text) > max_chars and pages_included > 0:
                    content_parts.append(
                        f"\n[... {total_pages - pages_included} more pages truncated due to length ...]"
                    )
                    break

                content_parts.append(f"\n### Page {page_num}")
                content_parts.append(page_text)
                content_parts.append("")  # Empty line between pages

                current_chars += len(page_text)
                pages_included += 1

        content_parts.extend(
            [
                "---END PDF CONTENT---",
                "",
                "Please use this PDF content to answer the user's question. Be specific and cite page numbers when referencing information.",
            ]
        )

        return {"role": "system", "content": "\n".join(content_parts)}

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
