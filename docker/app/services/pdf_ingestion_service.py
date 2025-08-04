"""
PDFIngestionService
--------------------
A single-step pipeline that:
1. Accepts an uploaded PDF (already processed by NVIngest)
2. Generates a unique pdf_id & stores the extracted data via FileStorageService
3. Creates semantic chunks and embeddings via PDFChunkingService
4. Uploads chunks to Milvus `pdf_chunks` collection
5. Returns the `pdf_id` and basic stats (pages, char_count, chunk_count)

This consolidates the previous fragmented PDF tools into one simple service.

NOTE:   – Expects PDF data already extracted by NVIngest service
        – Relies on existing PDFChunkingService for chunking + Milvus upload.
        – Safe to call multiple times; if pdf_id already exists, chunks are replaced.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, BinaryIO, Dict

from models.chat_config import ChatConfig
from pymilvus import MilvusClient
from services.file_storage_service import FileStorageService
from services.pdf_chunking_service import PDFChunkingService
from utils.config import config
from utils.pdf_id_generator import generate_pdf_id, get_existing_pdf_info

logger = logging.getLogger(__name__)


class PDFIngestionService:
    """High-level orchestrator for PDF ingestion."""

    def __init__(self, config: ChatConfig | None = None):
        self.config = config or ChatConfig.from_environment()
        self.file_storage = FileStorageService()
        self.chunking_service = PDFChunkingService(self.config)

        # Ensure pdf_links collection exists
        self._ensure_pdf_links_collection()

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------
    def ingest(
        self,
        pdf_data: Dict[str, Any],
        filename: str,
        session_id: str,
        pdf_content: BinaryIO = None,
        check_existing: bool = True,
    ) -> Dict[str, Any]:
        """Ingest PDF data (already extracted by NVIngest) and return ingestion metadata.

        Args:
            pdf_data: Extracted PDF data from NVIngest containing pages with text.
            filename: Original filename.
            session_id: User session identifier (for storage scoping).
            pdf_content: Optional PDF file content for content-based ID generation.
            check_existing: Whether to check if PDF already exists (enables deduplication).

        Returns:
            Dict containing pdf_id, total_pages, char_count, chunk_count.
        """
        logger.info(
            "Starting ingestion for PDF '%s' (session %s)", filename, session_id
        )

        # 1. Validate PDF data structure
        if not pdf_data or "pages" not in pdf_data:
            raise RuntimeError("Invalid PDF data structure - missing pages")

        pages = pdf_data.get("pages", [])
        if not pages:
            raise RuntimeError("PDF appears empty - no pages found")

        total_chars = sum(len(p.get("text", "")) for p in pages)
        logger.info("Processing %s chars across %s pages", total_chars, len(pages))

        # 2. Generate content-based pdf_id if possible
        if pdf_content:
            pdf_id = generate_pdf_id(pdf_content, filename)
            logger.info(f"Generated content-based ID: {pdf_id}")
        else:
            # Fallback to UUID if no content provided
            pdf_id = f"pdf_{uuid.uuid4().hex[:12]}"
            logger.warning(f"No PDF content provided, using random ID: {pdf_id}")

        # 3. Check if PDF already exists and optionally replace it
        pdf_exists = False
        if check_existing and pdf_content:
            existing_info = get_existing_pdf_info(pdf_id, self.file_storage)
            if existing_info:
                logger.info(f"PDF already exists: {pdf_id} - will replace it")
                pdf_exists = True
                # Delete existing chunks from Milvus before re-ingesting
                try:
                    deleted = self.chunking_service.delete_pdf_chunks(pdf_id)
                    logger.info(f"Deleted {deleted} existing chunks for PDF {pdf_id}")
                except Exception as e:
                    logger.warning(f"Failed to delete existing chunks: {e}")

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

        # 5. Chunk + embed
        chunks = self.chunking_service.chunk_pdf_document(storage_data)
        if not chunks:
            raise RuntimeError(
                "Chunking service returned no chunks – aborting ingestion"
            )

        logger.info("Created %s chunks – uploading to Milvus", len(chunks))
        success = self.chunking_service.store_chunks_with_embeddings(chunks)
        if not success:
            raise RuntimeError("Failed to upload PDF chunks to Milvus")

        logger.info("PDF '%s' ingestion completed (pdf_id=%s)", filename, pdf_id)
        return {
            "pdf_id": pdf_id,
            "total_pages": len(pages),
            "char_count": total_chars,
            "chunk_count": len(chunks),
            "replaced_existing": pdf_exists,
        }

    def _ensure_pdf_links_collection(self):
        """Ensure the pdf_links collection exists in Milvus."""
        try:
            # Initialize Milvus client
            milvus_client = MilvusClient(
                uri=config.env.DATABASE_URL,
                db_name=config.env.DEFAULT_DB,
            )

            collection_name = "pdf_links"

            # Check if collection already exists
            if not milvus_client.has_collection(collection_name):
                logger.info(f"Creating collection '{collection_name}'...")

                # Create collection with appropriate schema for storing PDF links/references
                # Adjust dimensions based on your embedding model if needed
                milvus_client.create_collection(
                    collection_name=collection_name,
                    dimension=2048,  # Adjust based on your embedding model
                    metric_type="L2",
                    consistency_level="Strong",
                )

                logger.info(f"Successfully created collection '{collection_name}'")
            else:
                logger.debug(f"Collection '{collection_name}' already exists")

            # Ensure collection is loaded
            try:
                milvus_client.load_collection(collection_name=collection_name)
                logger.debug(f"Collection '{collection_name}' loaded successfully")
            except Exception as e:
                logger.debug(
                    f"Collection '{collection_name}' may already be loaded: {e}"
                )

        except Exception as e:
            logger.error(f"Failed to ensure pdf_links collection exists: {e}")
            # Don't fail the entire service initialization if collection creation fails
            # The service can still work for other operations
