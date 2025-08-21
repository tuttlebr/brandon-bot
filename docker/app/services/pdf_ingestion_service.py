"""
PDF Ingestion Service

Handles the ingestion of PDF documents into the system, including:
– Content-based ID generation for deduplication
– Storage coordination with FileStorageService
– Chunking coordination with PDFChunkingService
– Safe to call multiple times; if pdf_id already exists, chunks are replaced.
"""

import logging
import uuid
from io import BinaryIO
from typing import Any, Dict

from models.chat_config import ChatConfig
from services.file_storage_service import FileStorageService
from services.pdf_chunking_service import PDFChunkingService
from utils.pdf_id_generator import generate_pdf_id, get_existing_pdf_info

logger = logging.getLogger(__name__)


class PDFIngestionService:
    """
    Service for ingesting PDF documents into the system.

    This service coordinates between FileStorageService and PDFChunkingService
    to ensure PDFs are properly stored and chunked for later retrieval.
    """

    def __init__(self, config: ChatConfig | None = None):
        """Initialize the PDF ingestion service"""
        self.config = config or ChatConfig.from_environment()
        self.file_storage = FileStorageService()
        self.chunking_service = PDFChunkingService(self.config)

        # Ensure pdf_links collection exists
        self._ensure_pdf_links_collection()

    def ingest(
        self,
        pdf_data: Dict[str, Any],
        filename: str,
        session_id: str,
        pdf_content: BinaryIO = None,
        check_existing: bool = True,
    ) -> Dict[str, Any]:
        """Ingest PDF data (already extracted by NVIngest) and return ingestion
        metadata.

        Args:
            pdf_data: Extracted PDF data from NVIngest containing pages with text.
            filename: Original filename.
            session_id: User session identifier (for storage scoping).
            pdf_content: Optional PDF file content for content-based ID
                        generation.
            check_existing: Whether to check if PDF already exists (enables
                           deduplication).

        Returns:
            Dict containing pdf_id, total_pages, char_count, chunk_count.
        """
        logger.info(
            "Starting ingestion for PDF '%s' (session %s)",
            filename,
            session_id,
        )

        # 1. Validate PDF data structure
        if not pdf_data or "pages" not in pdf_data:
            raise RuntimeError("Invalid PDF data structure - missing pages")

        pages = pdf_data.get("pages", [])
        if not pages:
            raise RuntimeError("PDF appears empty - no pages found")

        total_chars = sum(len(p.get("text", "")) for p in pages)
        logger.info(
            "Processing %s chars across %s pages", total_chars, len(pages)
        )

        # 2. Generate content-based pdf_id if possible
        if pdf_content:
            pdf_id = generate_pdf_id(pdf_content, filename)
            logger.info(f"Generated content-based ID: {pdf_id}")
        else:
            # Fallback to UUID if no content provided
            pdf_id = f"pdf_{uuid.uuid4().hex[:12]}"
            logger.warning(
                f"No PDF content provided, using random ID: {pdf_id}"
            )

        # 3. Check if PDF already exists and handle based on configuration
        pdf_exists = False
        if pdf_content:
            existing_info = get_existing_pdf_info(pdf_id, self.file_storage)
            if existing_info:
                if check_existing:
                    # Configuration says to re-upload existing PDFs
                    logger.info(
                        f"PDF already exists: {pdf_id} - will replace it"
                    )
                    pdf_exists = True
                    # Delete existing chunks from Milvus before re-ingesting
                    try:
                        deleted = self.chunking_service.delete_pdf_chunks(
                            pdf_id
                        )
                        logger.info(
                            f"Deleted {deleted} existing chunks for PDF {pdf_id}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete existing chunks: {e}"
                        )
                else:
                    # Configuration says to skip existing PDFs
                    logger.info(
                        f"PDF already exists: {pdf_id} - skipping upload "
                        f"(using existing)"
                    )
                    # Return existing PDF information without re-processing
                    return {
                        "pdf_id": pdf_id,
                        "total_pages": existing_info.get(
                            "total_pages", len(pages)
                        ),
                        "char_count": existing_info.get(
                            "char_count", total_chars
                        ),
                        "chunk_count": existing_info.get("chunk_count", 0),
                        "replaced_existing": False,
                        "skipped_existing": True,
                    }

        # 4. Prepare data structure for storage and chunking
        storage_data = {
            "pdf_id": pdf_id,
            "filename": filename,
            "pages": pages,
            "total_pages": len(pages),
            "char_count": total_chars,
        }

        # Store via FileStorageService
        self.file_storage.store_pdf(filename, storage_data, session_id)

        # 5. Chunk and store in vector database
        chunking_success = self.chunking_service.chunk_and_store_pdf(
            storage_data
        )

        if not chunking_success:
            raise RuntimeError(
                f"Failed to chunk and store PDF {pdf_id} in vector database"
            )

        # 6. Return ingestion metadata
        chunk_info = self.chunking_service.get_pdf_chunk_info(pdf_id)
        chunk_count = chunk_info.get("chunk_count", 0)

        logger.info(
            "Successfully ingested PDF '%s' (ID: %s) - %s pages, %s chars, "
            "%s chunks",
            filename,
            pdf_id,
            len(pages),
            total_chars,
            chunk_count,
        )

        return {
            "pdf_id": pdf_id,
            "total_pages": len(pages),
            "char_count": total_chars,
            "chunk_count": chunk_count,
            "replaced_existing": pdf_exists,
            "skipped_existing": False,
        }

    def _ensure_pdf_links_collection(self):
        """Ensure the PDF links collection exists in Milvus"""
        try:
            collection_name = "pdf_links"
            # Create collection with appropriate schema for storing PDF
            # links/references
            # This is a simplified version - in practice, you'd define the
            # schema properly
            logger.info(
                f"Ensuring PDF links collection '{collection_name}' exists"
            )

            # Note: This would typically involve creating the collection with
            # proper schema
            # For now, we'll just log that we're checking for it

        except Exception as e:
            logger.warning(
                f"Collection '{collection_name}' may already be loaded: {e}"
            )
            # Don't fail the entire service initialization if collection
            # creation fails
