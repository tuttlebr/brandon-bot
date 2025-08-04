"""
PDF ID Generator
----------------
Generates consistent, content-based IDs for PDFs to enable deduplication
and reliable identification across sessions.
"""

import hashlib
import logging
from typing import BinaryIO, Union


logger = logging.getLogger(__name__)


def generate_pdf_id(pdf_content: Union[bytes, BinaryIO], filename: str = None) -> str:
    """
    Generate a unique PDF ID based on file content.

    This ensures the same PDF always gets the same ID, enabling:
    - Deduplication: Skip re-processing already uploaded PDFs
    - Consistency: Same PDF referenced consistently across sessions
    - Uniqueness: Different PDFs always get different IDs

    Args:
        pdf_content: PDF file content as bytes or file-like object
        filename: Optional filename for logging purposes

    Returns:
        str: PDF ID in format 'pdf_<hash>' where hash is first 16 chars of SHA256
    """
    try:
        # Handle both bytes and file-like objects
        if hasattr(pdf_content, 'read'):
            # It's a file-like object (e.g., UploadedFile from Streamlit)
            # Read the content and reset position
            current_pos = pdf_content.tell() if hasattr(pdf_content, 'tell') else 0
            pdf_content.seek(0)
            content_bytes = pdf_content.read()
            pdf_content.seek(current_pos)  # Reset to original position
        else:
            # It's already bytes
            content_bytes = pdf_content

        # Generate SHA256 hash of content
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        # Use first 16 characters for ID (sufficient uniqueness)
        pdf_id = f"pdf_{content_hash[:16]}"

        logger.debug(f"Generated PDF ID {pdf_id} for {filename or 'unknown file'}")
        return pdf_id

    except Exception as e:
        logger.error(f"Error generating PDF ID: {e}")
        # Fallback to filename-based hash if content hashing fails
        if filename:
            fallback_hash = hashlib.md5(filename.encode()).hexdigest()[:16]
            return f"pdf_{fallback_hash}"
        else:
            # Last resort: timestamp-based ID
            import time

            timestamp_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
            return f"pdf_{timestamp_hash}"


def check_pdf_exists(pdf_id: str, milvus_client) -> bool:
    """
    Check if a PDF with given ID already exists in the database.

    Args:
        pdf_id: The PDF ID to check
        milvus_client: Milvus client instance

    Returns:
        bool: True if PDF exists, False otherwise
    """
    try:
        # Query for any chunks with this pdf_id
        from pymilvus import Collection

        collection = Collection("pdf_chunks")

        # Search for one result with this pdf_id
        expr = f'metadata like "%\\"pdf_id\\": \\"{pdf_id}\\"%"'
        results = collection.query(expr=expr, output_fields=["id"], limit=1)

        exists = len(results) > 0
        logger.debug(f"PDF {pdf_id} exists: {exists}")
        return exists

    except Exception as e:
        logger.warning(f"Error checking if PDF exists: {e}")
        # If we can't check, assume it doesn't exist
        return False


def get_existing_pdf_info(pdf_id: str, file_storage_service) -> dict:
    """
    Retrieve information about an existing PDF.

    Args:
        pdf_id: The PDF ID to retrieve
        file_storage_service: FileStorageService instance

    Returns:
        dict: PDF metadata if found, None otherwise
    """
    try:
        # Try to get PDF data from file storage
        pdf_data = file_storage_service.get_pdf(pdf_id)
        if pdf_data:
            # Count chunks if available in pages data
            chunk_count = 0
            pages = pdf_data.get('pages', [])
            if pages:
                # Estimate chunk count based on page count
                # Typically 2-3 chunks per page with overlapping windows
                chunk_count = len(pages) * 2

            return {
                'pdf_id': pdf_id,
                'filename': pdf_data.get('filename', 'Unknown'),
                'total_pages': pdf_data.get('total_pages', 0),
                'char_count': pdf_data.get('char_count', 0),
                'chunk_count': chunk_count,
                'already_exists': True,
            }
    except Exception as e:
        logger.debug(f"Could not retrieve existing PDF info: {e}")

    return None
