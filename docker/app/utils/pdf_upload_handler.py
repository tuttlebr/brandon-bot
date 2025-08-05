"""
PDF Upload Handler

Handles PDF file uploads for Streamlit, using NVIngest for extraction
and PDFIngestionService for chunking and embedding.
"""

import logging
from typing import Any, Dict, Optional

import streamlit as st
from controllers.file_controller import FileController
from models.chat_config import ChatConfig
from services.pdf_ingestion_service import PDFIngestionService
from services.session_state import set_active_pdf_id, set_session_id

logger = logging.getLogger(__name__)


def handle_pdf_upload(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Handle PDF file upload from Streamlit file uploader.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Dict with ingestion results or None if failed
    """
    if not uploaded_file:
        return None

    try:
        # Get session ID from Streamlit
        session_id = st.session_state.get('session_id', 'default')
        set_session_id(session_id)

        filename = uploaded_file.name
        logger.info(f"Processing PDF upload: {filename}")

        # First, use FileController to extract PDF via NVIngest
        config = ChatConfig.from_environment()
        file_controller = FileController(config)

        with st.spinner(f"Extracting text from {filename}..."):
            success, pdf_data = file_controller.process_pdf_upload(uploaded_file)

        if not success:
            error_msg = pdf_data.get('error', 'Failed to extract PDF text')
            st.error(f"❌ {error_msg}")
            return None

        # Now ingest the extracted data with content-based ID
        ingestion_service = PDFIngestionService(config)

        # Use configuration to determine whether to check for existing PDFs
        from utils.config import config as app_config

        check_existing = app_config.file_processing.PDF_REUPLOAD_EXISTING

        result = ingestion_service.ingest(
            pdf_data=pdf_data,
            filename=filename,
            session_id=session_id,
            pdf_content=uploaded_file,  # Pass the file for content-based ID generation
            check_existing=check_existing,  # Use configurable setting
        )

        # Set as active PDF
        pdf_id = result['pdf_id']
        set_active_pdf_id(pdf_id)

        # Store in session state for UI
        st.session_state['active_pdf'] = {
            'pdf_id': pdf_id,
            'filename': filename,
            'total_pages': result['total_pages'],
            'char_count': result['char_count'],
            'chunk_count': result['chunk_count'],
        }

        logger.info(f"PDF ingestion complete: {result}")

        # Display success message
        if result.get('skipped_existing'):
            st.success(f"✅ PDF already exists, using existing: {filename}")
            st.info(
                f"📄 **{filename}** (Already Processed)\n"
                f"- Pages: {result['total_pages']}\n"
                f"- Characters: {result['char_count']:,}\n"
                f"- Chunks: {result['chunk_count']}\n"
                f"- PDF ID: `{pdf_id}`\n"
                f"- Status: Using existing document chunks"
            )
        elif result.get('replaced_existing'):
            st.success(f"✅ Successfully replaced existing PDF: {filename}")
            st.info(
                f"📄 **{filename}** (Updated)\n"
                f"- Pages: {result['total_pages']}\n"
                f"- Characters: {result['char_count']:,}\n"
                f"- Chunks: {result['chunk_count']}\n"
                f"- PDF ID: `{pdf_id}`"
            )
        else:
            st.success(f"✅ Successfully processed new PDF: {filename}")
            st.info(
                f"📄 **{filename}**\n"
                f"- Pages: {result['total_pages']}\n"
                f"- Characters: {result['char_count']:,}\n"
                f"- Chunks: {result['chunk_count']}\n"
                f"- PDF ID: `{pdf_id}`"
            )

        return result

    except Exception as e:
        logger.error(f"PDF upload failed: {e}", exc_info=True)
        st.error(f"❌ Failed to process PDF: {str(e)}")
        return None


def clear_active_pdf():
    """Clear the active PDF from session state"""
    set_active_pdf_id(None)
    if 'active_pdf' in st.session_state:
        del st.session_state['active_pdf']


def get_active_pdf_info() -> Optional[Dict[str, Any]]:
    """Get information about the currently active PDF"""
    return st.session_state.get('active_pdf')
