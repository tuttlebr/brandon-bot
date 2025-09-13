"""
PDF Chunking Service

This service handles intelligent chunking of PDF documents with embeddings
for efficient retrieval and query processing. Uses a simplified approach
that's compatible with MilvusClient's auto-schema generation.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from openai import OpenAI
from pymilvus import MilvusClient
from utils.config import config

logger = logging.getLogger(__name__)


class PDFChunkingService:
    """Service for intelligent PDF chunking with embeddings"""

    def __init__(self, config_obj: ChatConfig):
        """Initialize the PDF chunking service"""
        self.config = config_obj
        self.chunk_size = config.file_processing.PDF_CHUNK_SIZE
        self.chunk_overlap = config.file_processing.PDF_CHUNK_OVERLAP
        self.min_chunk_size = config.file_processing.PDF_MIN_CHUNK_SIZE

        # Initialize embedding client
        self.embedding_client = OpenAI(
            base_url=config.env.EMBEDDING_ENDPOINT,
            api_key=config.env.EMBEDDING_API_KEY,
        )
        self.embedding_model = config.env.EMBEDDING_MODEL

        # Initialize Milvus client
        self.collection_name = "pdf_chunks"
        self.embedding_dim = 2048
        self._initialize_milvus()

    def _initialize_milvus(self):
        """Initialize Milvus with simplified approach"""
        try:
            # Ensure database exists
            self._ensure_database_exists()

            # Create client
            self.milvus_client = MilvusClient(
                uri=config.env.DATABASE_URL,
                db_name=config.env.DEFAULT_DB,
            )

            # Check if collection exists
            if not self.milvus_client.has_collection(self.collection_name):
                logger.info(f"Creating collection '{self.collection_name}'...")

                # Use the simplest possible creation method
                # MilvusClient will auto-generate schema on first insert
                self.milvus_client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.embedding_dim,
                    metric_type="L2",
                    consistency_level="Strong",
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(
                    f"Collection '{self.collection_name}' already exists"
                )

            # Try to load collection
            try:
                self.milvus_client.load_collection(
                    collection_name=self.collection_name
                )
                logger.info(f"Loaded collection: {self.collection_name}")
            except Exception as e:
                logger.warning(
                    f"Could not load collection (might need data first): {e}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            self.milvus_client = None

    def _ensure_database_exists(self):
        """Ensure database exists"""
        try:
            from pymilvus import connections, db

            uri = config.env.DATABASE_URL
            db_name = config.env.DEFAULT_DB

            # Try simple connection first
            test_client = MilvusClient(uri=uri)

            # Check if database exists by trying to use it
            try:
                test_client = MilvusClient(uri=uri, db_name=db_name)
                logger.info(f"Database '{db_name}' exists")
                return True
            except:
                # Create database
                logger.info(f"Creating database '{db_name}'...")
                connections.connect(alias="default", uri=uri)
                db.create_database(db_name)
                connections.disconnect("default")
                logger.info(f"Created database '{db_name}'")
                return True

        except Exception as e:
            logger.error(f"Error with database: {e}")
            return False

    def chunk_and_store_pdf(self, pdf_data: Dict[str, Any]) -> bool:
        """
        Chunk PDF and store in Milvus in one operation

        Args:
            pdf_data: PDF data containing pages and metadata

        Returns:
            True if successful
        """
        if not self.milvus_client:
            logger.error("Milvus client not initialized")
            return False

        filename = pdf_data.get("filename", "Unknown")
        pdf_id = pdf_data.get("pdf_id")
        if not pdf_id:
            raise ValueError("pdf_id must be provided in pdf_data")
        pages = pdf_data.get("pages", [])

        logger.info(f"Processing {filename} ({len(pages)} pages)")

        # Create chunks
        chunks = self._create_simple_chunks(pages, pdf_id)

        # Prepare data for insertion
        data_to_insert = []

        for i, chunk in enumerate(chunks):
            # Create embedding
            embedding = self._create_embedding(chunk["text"])
            if not embedding:
                continue

            # Generate unique int64 ID
            # Use hash of pdf_id and chunk_index to ensure uniqueness
            id_str = f"{pdf_id}_{i}"
            chunk_id = abs(hash(id_str)) % (10**15)  # Ensure it fits in int64

            # Prepare simplified data structure
            # Use the fields that MilvusClient auto-schema expects
            chunk_data = {
                "id": chunk_id,
                "vector": (
                    embedding
                ),  # MilvusClient often expects 'vector' not 'embedding'
                "text": chunk["text"][:60000],  # Limit text size
                "metadata": json.dumps(
                    {
                        "pdf_id": pdf_id,
                        "filename": filename,
                        "chunk_index": i,
                        "pages": chunk["pages"],
                        "total_chunks": len(chunks),
                    }
                ),
            }

            data_to_insert.append(chunk_data)

        if data_to_insert:
            try:
                logger.info(
                    f"📤 Starting Milvus upload for {filename}:"
                    f" {len(data_to_insert)} chunks"
                )

                # Log details about chunks being uploaded
                total_text_size = sum(
                    len(chunk["text"]) for chunk in data_to_insert
                )
                logger.info(
                    "📊 Chunk details: Total text size:"
                    f" {total_text_size:,} chars, Average chunk size:"
                    f" {total_text_size // len(data_to_insert):,} chars"
                )

                # Insert all at once
                self.milvus_client.insert(
                    collection_name=self.collection_name, data=data_to_insert
                )

                logger.info(
                    f"✅ Successfully stored {len(data_to_insert)} chunks for"
                    f" {filename} in Milvus collection"
                    f" '{self.collection_name}'"
                )

                # Log individual chunk info for debugging
                for i, chunk in enumerate(
                    data_to_insert[:3]
                ):  # Log first 3 chunks
                    logger.debug(
                        f"  Chunk {i}: ID={chunk['id']}, Text preview:"
                        f" {chunk['text'][:100]}..."
                    )
                if len(data_to_insert) > 3:
                    logger.debug(
                        f"  ... and {len(data_to_insert) - 3} more chunks"
                    )

                # Try to load collection after first insert
                try:
                    self.milvus_client.load_collection(
                        collection_name=self.collection_name
                    )
                    logger.info(
                        f"📚 Milvus collection '{self.collection_name}' loaded"
                        " successfully"
                    )
                except:
                    pass  # Ignore if already loaded

                # Update session state to indicate Milvus upload complete
                import streamlit as st

                if hasattr(st, "session_state"):
                    st.session_state.pdf_milvus_upload_complete = True
                    st.session_state.pdf_milvus_upload_filename = filename
                    st.session_state.pdf_milvus_upload_chunks = len(
                        data_to_insert
                    )

                return True
            except Exception as e:
                logger.error(f"❌ Failed to store chunks in Milvus: {e}")
                return False
        else:
            logger.warning("⚠️ No chunks to store in Milvus")
            return False

    def _create_simple_chunks(
        self, pages: List[Dict[str, Any]], pdf_id: str
    ) -> List[Dict[str, Any]]:
        """Create simple chunks using sliding window"""
        chunks = []

        # Combine all text
        full_text = ""
        page_boundaries = {}

        for page in pages:
            page_num = page.get("page", len(page_boundaries) + 1)
            page_text = page.get("text", "").strip()

            if page_text:
                page_start = len(full_text)
                full_text += f"\n\n[Page {page_num}]\n{page_text}"
                page_boundaries[page_num] = page_start

        # Create chunks
        start = 0
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))

            # Try to end at paragraph
            if end < len(full_text):
                para_end = full_text.rfind("\n\n", start, end)
                if para_end > start + self.min_chunk_size:
                    end = para_end

            chunk_text = full_text[start:end].strip()

            # Find which pages this chunk covers
            chunk_pages = []
            for page_num, pos in page_boundaries.items():
                if start <= pos < end:
                    chunk_pages.append(page_num)

            if not chunk_pages and page_boundaries:
                # Assign to nearest page
                for page_num, pos in sorted(
                    page_boundaries.items(), key=lambda x: x[1]
                ):
                    if pos > start:
                        chunk_pages = [page_num - 1] if page_num > 1 else [1]
                        break
                if not chunk_pages:
                    chunk_pages = [max(page_boundaries.keys())]

            chunks.append(
                {
                    "text": chunk_text,
                    "pages": sorted(set(chunk_pages)),
                    "pdf_id": pdf_id,
                }
            )

            # Move with overlap
            start = (
                end - self.chunk_overlap
                if end - self.chunk_overlap > start
                else end
            )

        return chunks

    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for text"""
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float",
                extra_body={
                    "input_type": "passage",
                    "encoding_format": "float",
                    "truncate": "END",
                },
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    def _generate_pdf_id(self, filename: str) -> str:
        """Generate unique ID for PDF"""
        return hashlib.md5(filename.encode()).hexdigest()[:16]

    def search_chunks(
        self, query: str, pdf_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if not self.milvus_client:
            return []

        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []

            # Search
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=limit * 2,  # Get more to filter
                output_fields=["id", "text", "metadata"],
            )

            if not results or not results[0]:
                return []

            # Filter by pdf_id and format results
            relevant_chunks = []
            for hit in results[0]:
                metadata = json.loads(hit["entity"].get("metadata", "{}"))
                if metadata.get("pdf_id") == pdf_id:
                    relevant_chunks.append(
                        {
                            "text": hit["entity"].get("text", ""),
                            "pages": metadata.get("pages", []),
                            "chunk_index": metadata.get("chunk_index", 0),
                            "score": hit["distance"],
                        }
                    )

                    if len(relevant_chunks) >= limit:
                        break

            return relevant_chunks

        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []

    def delete_pdf_chunks(self, pdf_id: str) -> int:
        """Delete all chunks for a PDF

        Returns:
            int: Number of chunks deleted
        """
        if not self.milvus_client:
            return 0

        try:
            # Get all chunks for this PDF
            # Use JSON field access
            all_results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter=f'metadata["pdf_id"] == "{pdf_id}"',
                output_fields=["id"],
            )

            if all_results:
                ids_to_delete = [r["id"] for r in all_results]
                # Delete by primary key IDs
                self.milvus_client.delete(
                    collection_name=self.collection_name, ids=ids_to_delete
                )
                logger.info(
                    f"Deleted {len(ids_to_delete)} chunks for PDF: {pdf_id}"
                )
                return len(ids_to_delete)

            return 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return 0

    # Compatibility methods to match original interface
    def chunk_pdf_document(
        self, pdf_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks from PDF document (compatibility method)"""
        filename = pdf_data.get("filename", "Unknown")
        pdf_id = pdf_data.get("pdf_id")
        if not pdf_id:
            raise ValueError("pdf_id must be provided in pdf_data")
        pages = pdf_data.get("pages", [])

        logger.info(f"Creating chunks for {filename} ({len(pages)} pages)")
        chunks = self._create_simple_chunks(pages, pdf_id)

        # Add metadata to match original format
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["filename"] = filename
            chunk["total_chunks"] = len(chunks)
            chunk["type"] = "sliding_window"

        return chunks

    def store_chunks_with_embeddings(
        self, chunks: List[Dict[str, Any]]
    ) -> bool:
        """Store chunks with embeddings (compatibility method)"""
        if not chunks or not self.milvus_client:
            return False

        # Get PDF info from first chunk
        pdf_id = chunks[0].get("pdf_id")
        filename = chunks[0].get("filename", "Unknown")

        logger.info(
            f"📤 Starting Milvus upload for {filename}:"
            f" {len(chunks)} pre-chunked segments"
        )

        # Prepare data for insertion
        data_to_insert = []

        for chunk in chunks:
            # Create embedding
            embedding = self._create_embedding(chunk["text"])
            if not embedding:
                continue

            # Generate unique int64 ID
            # Use hash of pdf_id and chunk_index to ensure uniqueness
            id_str = f"{pdf_id}_{chunk['chunk_index']}"
            chunk_id = abs(hash(id_str)) % (10**15)  # Ensure it fits in int64

            # Prepare data in simplified format
            chunk_data = {
                "id": chunk_id,
                "vector": embedding,
                "text": chunk["text"][:60000],  # Limit text size
                "metadata": json.dumps(
                    {
                        "pdf_id": pdf_id,
                        "filename": filename,
                        "chunk_index": chunk["chunk_index"],
                        "pages": chunk["pages"],
                        "total_chunks": chunk.get("total_chunks", len(chunks)),
                    }
                ),
            }

            data_to_insert.append(chunk_data)

        if data_to_insert:
            try:
                logger.info(
                    f"📊 Uploading {len(data_to_insert)} embeddings to Milvus"
                )

                # Insert all at once
                self.milvus_client.insert(
                    collection_name=self.collection_name, data=data_to_insert
                )

                logger.info(
                    f"✅ Successfully stored {len(data_to_insert)} chunks for"
                    f" {filename} in Milvus collection"
                    f" '{self.collection_name}'"
                )

                # Try to load collection after first insert
                try:
                    self.milvus_client.load_collection(
                        collection_name=self.collection_name
                    )
                    logger.info(
                        f"📚 Milvus collection '{self.collection_name}' loaded"
                        " successfully"
                    )
                except:
                    pass  # Ignore if already loaded

                # Update session state to indicate Milvus upload complete
                import streamlit as st

                if hasattr(st, "session_state"):
                    st.session_state.pdf_milvus_upload_complete = True
                    st.session_state.pdf_milvus_upload_filename = filename
                    st.session_state.pdf_milvus_upload_chunks = len(
                        data_to_insert
                    )

                return True
            except Exception as e:
                logger.error(f"❌ Failed to store chunks in Milvus: {e}")
                return False
        else:
            logger.warning("⚠️ No chunks to store in Milvus")
            return False

    def get_pdf_chunk_info(self, pdf_id: str) -> Dict[str, Any]:
        """Get information about chunks for a PDF"""
        if not self.milvus_client:
            return {"error": "Milvus client not initialized"}

        try:
            # Query all chunks for this PDF
            # Use JSON field access
            all_results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter=f'metadata["pdf_id"] == "{pdf_id}"',
                output_fields=["id", "metadata"],
                limit=1000,
            )

            if all_results:
                total_chunks = len(all_results)
                all_pages = set()

                for result in all_results:
                    metadata = json.loads(result.get("metadata", "{}"))
                    pages = metadata.get("pages", [])
                    all_pages.update(pages)

                # Get first chunk metadata for type
                json.loads(all_results[0].get("metadata", "{}"))

                return {
                    "pdf_id": pdf_id,
                    "total_chunks": total_chunks,
                    "pages_covered": sorted(all_pages),
                    "chunk_type": "sliding_window",
                }
            else:
                return {"pdf_id": pdf_id, "total_chunks": 0}

        except Exception as e:
            logger.error(f"Error getting chunk info: {e}")
            return {"error": str(e)}
