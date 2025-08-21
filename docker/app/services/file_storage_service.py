"""
File Storage Service

This service handles external file storage for images and PDFs,
keeping only references in session state to avoid memory issues.
"""

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from utils.config import config
from utils.exceptions import FileProcessingError, MemoryLimitError

logger = logging.getLogger(__name__)


class FileStorageService:
    """Service for managing external file storage"""

    _instance = None

    def __new__(cls, storage_path: str = "/tmp/chatbot_storage"):
        """Implement singleton pattern to ensure only one instance"""
        if cls._instance is None:
            cls._instance = super(FileStorageService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, storage_path: str = "/tmp/chatbot_storage"):
        """
        Initialize file storage service

        Args:
            storage_path: Base path for file storage
        """
        # Only initialize once
        if self._initialized:
            return

        self.storage_path = Path(storage_path)
        self._ensure_storage_dirs()
        self._initialized = True

    def _ensure_storage_dirs(self):
        """Ensure storage directories exist"""
        try:
            self.images_dir = self.storage_path / "images"
            self.pdfs_dir = self.storage_path / "pdfs"
            self.metadata_dir = self.storage_path / "metadata"

            for dir_path in [
                self.images_dir,
                self.pdfs_dir,
                self.metadata_dir,
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Storage directories initialized at {self.storage_path}"
            )
        except Exception as e:
            logger.error(f"Failed to create storage directories: {e}")
            raise FileProcessingError(f"Storage initialization failed: {e}")

    def store_image(
        self,
        image_data: str,
        enhanced_prompt: str,
        original_prompt: str,
        session_id: str,
    ) -> str:
        """
        Store image externally and return reference ID

        Args:
            image_data: Base64 encoded image data
            enhanced_prompt: Enhanced prompt used
            original_prompt: Original user prompt
            session_id: Session identifier

        Returns:
            Image reference ID
        """
        try:
            # Generate unique ID based on content hash
            image_hash = hashlib.md5(image_data.encode()).hexdigest()[:12]
            image_id = f"{config.session.IMAGE_ID_PREFIX}{image_hash}"

            # Check storage limits
            self._check_storage_limits(session_id, "images")

            # Save image file
            image_path = self.images_dir / f"{image_id}.png"
            if not image_path.exists():
                image_bytes = base64.b64decode(image_data)
                image_path.write_bytes(image_bytes)

            # Save metadata
            metadata = {
                "image_id": image_id,
                "enhanced_prompt": enhanced_prompt,
                "original_prompt": original_prompt,
                "session_id": session_id,
                "file_path": str(image_path),
                "size_bytes": len(image_data),
            }

            metadata_path = self.metadata_dir / f"{image_id}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            logger.info(f"Stored image {image_id} for session {session_id}")
            return image_id

        except Exception as e:
            logger.error(f"Failed to store image: {e}")
            raise FileProcessingError(f"Image storage failed: {e}")

    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image data and metadata

        Args:
            image_id: Image reference ID

        Returns:
            Dict with image data and metadata, or None if not found
        """
        try:
            # Load metadata
            metadata_path = self.metadata_dir / f"{image_id}.json"
            if not metadata_path.exists():
                return None

            metadata = json.loads(metadata_path.read_text())

            # Load image data
            image_path = Path(metadata["file_path"])
            if image_path.exists():
                image_bytes = image_path.read_bytes()
                metadata["image_data"] = base64.b64encode(image_bytes).decode()
            else:
                logger.warning(f"Image file not found: {image_path}")
                return None

            return metadata

        except Exception as e:
            logger.error(f"Failed to retrieve image {image_id}: {e}")
            return None

    def store_uploaded_image(
        self, image_data: str, filename: str, file_type: str, session_id: str
    ) -> str:
        """
        Store uploaded image externally and return reference ID

        Args:
            image_data: Base64 encoded image data
            filename: Original image filename
            file_type: MIME type of the image
            session_id: Session identifier

        Returns:
            Image reference ID
        """
        try:
            logger.debug(
                f"Storing uploaded image: {filename}, type: {file_type}, base64 length: {len(image_data)}"
            )

            # Generate unique ID based on content hash
            image_hash = hashlib.md5(image_data.encode()).hexdigest()[:12]
            image_id = f"uploaded_img_{image_hash}"

            # Check storage limits
            self._check_storage_limits(session_id, "images")

            # Determine file extension from MIME type
            extension = '.png'  # default
            if file_type:
                if 'jpeg' in file_type or 'jpg' in file_type:
                    extension = '.jpg'
                elif 'png' in file_type:
                    extension = '.png'
                elif 'gif' in file_type:
                    extension = '.gif'
                elif 'bmp' in file_type:
                    extension = '.bmp'

            # Save image file
            image_path = self.images_dir / f"{image_id}{extension}"
            if not image_path.exists():
                image_bytes = base64.b64decode(image_data)
                image_path.write_bytes(image_bytes)
                logger.debug(
                    f"Saved image file: {image_path}, size: {len(image_bytes)} bytes"
                )

            # Save metadata
            metadata = {
                "image_id": image_id,
                "filename": filename,
                "file_type": file_type,
                "session_id": session_id,
                "file_path": str(image_path),
                "size_bytes": len(
                    base64.b64decode(image_data)
                ),  # Use actual bytes size, not base64 size
                "upload_type": "user_uploaded",
            }

            metadata_path = self.metadata_dir / f"{image_id}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            logger.info(
                f"Stored uploaded image {image_id} ({filename}), size: {metadata['size_bytes'] / 1024:.1f} KB"
            )
            return image_id

        except Exception as e:
            logger.error(f"Failed to store uploaded image: {e}")
            raise FileProcessingError(f"Uploaded image storage failed: {e}")

    def get_uploaded_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve uploaded image data and metadata

        Args:
            image_id: Image reference ID

        Returns:
            Dict with image data and metadata, or None if not found
        """
        try:
            # Load metadata
            metadata_path = self.metadata_dir / f"{image_id}.json"
            if not metadata_path.exists():
                return None

            metadata = json.loads(metadata_path.read_text())

            # Load image data
            image_path = Path(metadata["file_path"])
            if image_path.exists():
                image_bytes = image_path.read_bytes()
                metadata["image_data"] = base64.b64encode(image_bytes).decode()
            else:
                logger.warning(f"Uploaded image file not found: {image_path}")
                return None

            return metadata

        except Exception as e:
            logger.error(f"Failed to retrieve uploaded image {image_id}: {e}")
            return None

    def store_pdf(
        self, filename: str, pdf_data: Dict[str, Any], session_id: str
    ) -> str:
        """
        Store PDF data externally and return reference ID

        Args:
            filename: Original PDF filename
            pdf_data: Processed PDF data
            session_id: Session identifier

        Returns:
            PDF reference ID
        """
        try:
            # PDF ID should always be provided by the caller
            pdf_id = pdf_data.get('pdf_id')
            if not pdf_id:
                raise ValueError("pdf_id must be provided in pdf_data")

            # Check storage limits
            self._check_storage_limits(session_id, "pdfs")

            # Save PDF data
            pdf_path = self.pdfs_dir / f"{pdf_id}.json"
            logger.debug(f"Storing PDF at: {pdf_path}")
            pdf_path.write_text(json.dumps(pdf_data, indent=2))
            logger.debug(
                f"PDF stored successfully at: {pdf_path}, exists: {pdf_path.exists()}"
            )

            # Save metadata
            metadata = {
                "pdf_id": pdf_id,
                "filename": filename,
                "session_id": session_id,
                "file_path": str(pdf_path),
                "total_pages": len(pdf_data.get("pages", [])),
                "size_bytes": len(json.dumps(pdf_data)),
            }

            metadata_path = self.metadata_dir / f"{pdf_id}_meta.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            logger.info(f"Stored PDF {pdf_id} for session {session_id}")
            return pdf_id

        except Exception as e:
            logger.error(f"Failed to store PDF: {e}")
            raise FileProcessingError(f"PDF storage failed: {e}")

    def get_pdf(self, pdf_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve PDF data

        Args:
            pdf_id: PDF reference ID

        Returns:
            PDF data or None if not found
        """
        try:
            pdf_path = self.pdfs_dir / f"{pdf_id}.json"
            logger.debug(f"Looking for PDF at: {pdf_path}")
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {pdf_path}")
                # List available PDFs for debugging
                available_pdfs = list(self.pdfs_dir.glob("*.json"))
                logger.debug(
                    f"Available PDFs in storage: {[p.name for p in available_pdfs]}"
                )
                return None

            pdf_data = json.loads(pdf_path.read_text())

            # Load metadata and merge filename into pdf_data
            metadata_path = self.metadata_dir / f"{pdf_id}_meta.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                # Add filename from metadata to pdf_data
                filename = metadata.get('filename', '')
                if not filename:
                    # If no filename in metadata, try to derive from PDF data or use a default
                    filename = pdf_data.get('filename', f'{pdf_id}.pdf')
                    logger.warning(
                        f"No filename in metadata for {pdf_id}, using: {filename}"
                    )

                pdf_data['filename'] = filename
                # Add total_pages from metadata to pdf_data
                total_pages = metadata.get('total_pages', 0)
                pdf_data['total_pages'] = total_pages
                # Optionally add other metadata fields if needed
                pdf_data['pdf_id'] = metadata.get('pdf_id', pdf_id)
            else:
                # If no metadata, ensure we still have a filename
                if 'filename' not in pdf_data or not pdf_data['filename']:
                    pdf_data['filename'] = f'{pdf_id}.pdf'
                    logger.warning(
                        f"No metadata found for {pdf_id}, using filename: {pdf_data['filename']}"
                    )
                pdf_data['pdf_id'] = pdf_id
                # Calculate total_pages from pages if metadata not available
                pdf_data['total_pages'] = len(pdf_data.get('pages', []))

            logger.debug(
                f"Retrieved PDF {pdf_id} with filename: {pdf_data.get('filename')}"
            )
            return pdf_data

        except Exception as e:
            logger.error(f"Failed to retrieve PDF {pdf_id}: {e}")
            return None

    def update_pdf(self, pdf_id: str, pdf_data: Dict[str, Any]) -> bool:
        """
        Update existing PDF data

        Args:
            pdf_id: PDF reference ID
            pdf_data: Updated PDF data

        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_path = self.pdfs_dir / f"{pdf_id}.json"
            if not pdf_path.exists():
                logger.error(f"PDF {pdf_id} not found for update")
                return False

            # Remove metadata fields from pdf_data before storing
            # (they should remain in the metadata file only)
            pdf_data_to_store = pdf_data.copy()
            pdf_data_to_store.pop('filename', None)
            pdf_data_to_store.pop('pdf_id', None)

            # Update the PDF data
            pdf_path.write_text(json.dumps(pdf_data_to_store, indent=2))

            # Update metadata with new size
            metadata_path = self.metadata_dir / f"{pdf_id}_meta.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                metadata["size_bytes"] = len(json.dumps(pdf_data_to_store))
                metadata_path.write_text(json.dumps(metadata, indent=2))

            logger.info(f"Updated PDF {pdf_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update PDF {pdf_id}: {e}")
            return False

    def cleanup_session(self, session_id: str):
        """
        Clean up all files for a session

        Args:
            session_id: Session identifier
        """
        try:
            # Find all metadata files for this session
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    metadata = json.loads(metadata_file.read_text())
                    if metadata.get("session_id") == session_id:
                        # Remove associated files
                        if "file_path" in metadata:
                            file_path = Path(metadata["file_path"])
                            if file_path.exists():
                                file_path.unlink()

                        # Remove metadata
                        metadata_file.unlink()

                except Exception as e:
                    logger.warning(f"Error cleaning up file: {e}")

            logger.info(f"Cleaned up files for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup session {session_id}: {e}")

    def _check_storage_limits(self, session_id: str, file_type: str):
        """
        Check storage limits for a session

        Args:
            session_id: Session identifier
            file_type: Type of file ("images" or "pdfs")

        Raises:
            MemoryLimitError: If limits are exceeded
        """
        # Use a more efficient approach with early termination
        count = 0
        total_size = 0

        # Define limits upfront
        max_count = (
            config.session.MAX_IMAGES_IN_SESSION
            if file_type == "images"
            else config.session.MAX_PDFS_IN_SESSION
        )
        max_size = 100 * 1024 * 1024  # 100MB

        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    # Read file only if we haven't exceeded limits
                    if count >= max_count or total_size > max_size:
                        break

                    metadata = json.loads(metadata_file.read_text())

                    # Skip if not for this session
                    if metadata.get("session_id") != session_id:
                        continue

                    # Check file type and update counters
                    if file_type == "images" and "image_id" in metadata:
                        count += 1
                        total_size += metadata.get("size_bytes", 0)
                    elif file_type == "pdfs" and "pdf_id" in metadata:
                        count += 1
                        total_size += metadata.get("size_bytes", 0)

                except (json.JSONDecodeError, IOError) as e:
                    logger.debug(
                        f"Error reading metadata file {metadata_file}: {e}"
                    )
                    continue

            # Check limits
            if count >= max_count:
                raise MemoryLimitError(
                    f"{file_type.title()} limit exceeded: {count}/{max_count}"
                )

            if total_size > max_size:
                raise MemoryLimitError(
                    f"Storage limit exceeded: {total_size / (1024*1024):.1f}MB"
                )

        except MemoryLimitError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error checking storage limits: {e}")
            # Don't suppress the error, let it propagate
            raise
