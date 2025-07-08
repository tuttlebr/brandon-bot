import logging
from typing import Any, Dict

import streamlit as st
from models.chat_config import ChatConfig
from services.file_storage_service import FileStorageService
from utils.config import config
from utils.system_prompt import get_system_prompt


class SessionController:
    """Controller for managing Streamlit session state and cleanup operations"""

    def __init__(self, config_obj: ChatConfig):
        """
        Initialize the session controller

        Args:
            config_obj: Application configuration
        """
        self.config_obj = config_obj
        self.file_storage = FileStorageService()

    def initialize_session_state(self):
        """Initialize Streamlit session state with default values"""
        # Use atomic check-and-set to prevent race conditions in concurrent sessions
        if not getattr(st.session_state, "initialized", False):
            # Set initialization flag first to prevent multiple initializations
            st.session_state.initialized = True

            # Conditionally create a unique session ID if it doesn't exist
            if (
                not hasattr(st.session_state, 'session_id')
                or not st.session_state.session_id
            ):
                import random
                import time

                st.session_state.session_id = (
                    f"session_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
                )

            st.session_state.fast_llm_model_name = self.config_obj.fast_llm_model_name
            st.session_state.llm_model_name = self.config_obj.llm_model_name
            st.session_state.intelligent_llm_model_name = (
                self.config_obj.intelligent_llm_model_name
            )
            st.session_state.messages = [
                {"role": "system", "content": get_system_prompt()}
            ]
            st.session_state.current_page = config.ui.CURRENT_PAGE_DEFAULT
            st.session_state.processing = False

            # Initialize references to stored files (not the files themselves)
            if not hasattr(st.session_state, 'stored_images'):
                st.session_state.stored_images = []  # List of image IDs

            # Initialize references to stored PDFs
            if not hasattr(st.session_state, 'stored_pdfs'):
                st.session_state.stored_pdfs = []  # List of PDF IDs

            logging.info(
                f"Initialized new session state for session: {st.session_state.session_id}"
            )

    def cleanup_session(self):
        """Clean up all session data including external files"""
        if hasattr(st.session_state, 'session_id'):
            self.file_storage.cleanup_session(st.session_state.session_id)
            logging.info(f"Cleaned up session: {st.session_state.session_id}")

    def set_processing_state(self, processing: bool):
        """
        Set the processing state to prevent concurrent operations

        Args:
            processing: Whether the app is currently processing
        """
        # Ensure session state is initialized before setting processing state
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()
        st.session_state.processing = processing

    def is_processing(self) -> bool:
        """
        Check if the app is currently processing

        Returns:
            True if processing, False otherwise
        """
        # Ensure session state is initialized before checking processing state
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()
        return st.session_state.get("processing", False)

    def store_tool_context(self, context: str):
        """
        Store tool context in session state

        Args:
            context: Tool context to store
        """
        st.session_state.last_tool_context = context

    def clear_tool_context(self):
        """Clear previous tool context from session state"""
        if hasattr(st.session_state, 'last_tool_context'):
            st.session_state.last_tool_context = None

    def store_generated_image(
        self, image_data: str, enhanced_prompt: str, original_prompt: str
    ) -> str:
        """
        Store generated image externally and return image ID

        Args:
            image_data: Base64 encoded image data
            enhanced_prompt: Enhanced prompt used for generation
            original_prompt: Original user prompt

        Returns:
            Unique image ID for the stored image
        """
        # Ensure session is initialized
        if not hasattr(st.session_state, 'session_id'):
            self.initialize_session_state()

        # Store image externally
        image_id = self.file_storage.store_image(
            image_data, enhanced_prompt, original_prompt, st.session_state.session_id
        )

        # Keep reference in session state
        if 'stored_images' not in st.session_state:
            st.session_state.stored_images = []
        st.session_state.stored_images.append(image_id)

        # Limit stored references
        if len(st.session_state.stored_images) > config.session.MAX_IMAGES_IN_SESSION:
            st.session_state.stored_images = st.session_state.stored_images[
                -config.session.MAX_IMAGES_IN_SESSION :
            ]

        return image_id

    def get_generated_image(self, image_id: str) -> Dict[str, Any]:
        """
        Retrieve generated image data

        Args:
            image_id: Image ID to retrieve

        Returns:
            Image data dictionary or None
        """
        return self.file_storage.get_image(image_id)

    def get_model_name(self, model_type: str = "fast") -> str:
        """
        Safely get model name from session state with fallback to config

        Args:
            model_type: Type of model ('fast', 'llm', or 'intelligent')

        Returns:
            Model name string
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        # Map model types to session state keys and config fallbacks
        model_mapping = {
            "fast": ("fast_llm_model_name", self.config_obj.fast_llm_model_name),
            "llm": ("llm_model_name", self.config_obj.llm_model_name),
            "intelligent": (
                "intelligent_llm_model_name",
                self.config_obj.intelligent_llm_model_name,
            ),
        }

        if model_type not in model_mapping:
            logging.warning(
                f"Unknown model type '{model_type}', defaulting to fast model"
            )
            model_type = "fast"

        session_key, config_fallback = model_mapping[model_type]
        model_name = st.session_state.get(session_key, config_fallback)

        if not model_name:
            logging.warning(f"No {model_type} model name found, using config fallback")
            model_name = config_fallback

        return model_name

    def get_messages(self):
        """
        Safely get messages from session state with initialization check

        Returns:
            List of messages from session state
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        return getattr(st.session_state, "messages", [])

    def add_message(self, role: str, content: any):
        """
        Safely add a message to session state

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        if not hasattr(st.session_state, "messages"):
            st.session_state.messages = []

        st.session_state.messages.append({"role": role, "content": content})

    def set_messages(self, messages: list):
        """
        Safely set messages in session state

        Args:
            messages: List of messages to set
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        st.session_state.messages = messages

    def store_pdf_document(self, filename: str, pdf_data: dict) -> str:
        """
        Store PDF document externally

        Args:
            filename: Name of the PDF file
            pdf_data: Processed PDF data containing pages and metadata

        Returns:
            Unique PDF ID for the stored document
        """
        # Ensure session is initialized
        if not hasattr(st.session_state, 'session_id'):
            self.initialize_session_state()

        # Store PDF externally
        pdf_id = self.file_storage.store_pdf(
            filename, pdf_data, st.session_state.session_id
        )

        # Keep reference in session state
        if 'stored_pdfs' not in st.session_state:
            st.session_state.stored_pdfs = []
        st.session_state.stored_pdfs.append(pdf_id)

        # Limit stored references
        if len(st.session_state.stored_pdfs) > config.session.MAX_PDFS_IN_SESSION:
            # Remove oldest PDFs
            removed = st.session_state.stored_pdfs[
                : -config.session.MAX_PDFS_IN_SESSION
            ]
            st.session_state.stored_pdfs = st.session_state.stored_pdfs[
                -config.session.MAX_PDFS_IN_SESSION :
            ]

            # Clean up removed PDFs
            for pdf_id in removed:
                logging.info(f"Removing old PDF: {pdf_id}")

        logging.info(f"Stored PDF document '{filename}' with ID '{pdf_id}'")
        return pdf_id

    def get_pdf_documents(self) -> dict:
        """
        Get all stored PDF documents from external storage

        Returns:
            Dictionary of PDF documents keyed by PDF ID
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        pdfs = {}
        for pdf_id in getattr(st.session_state, 'stored_pdfs', []):
            # Check if this is a batch-processed PDF
            batch_info_key = f"{pdf_id}_batch_info"
            if hasattr(st.session_state, batch_info_key):
                batch_info = getattr(st.session_state, batch_info_key)
                if batch_info.get('batch_processed', False):
                    # For batch-processed PDFs, return metadata only
                    pdfs[pdf_id] = {
                        'pdf_id': pdf_id,
                        'filename': batch_info.get('filename', 'Unknown'),
                        'total_pages': batch_info.get('total_pages', 0),
                        'batch_processed': True,
                        'total_batches': batch_info.get('total_batches', 0),
                        'pages': [],  # Empty to avoid loading all batches
                    }
                    continue

            # Regular PDF
            pdf_data = self.file_storage.get_pdf(pdf_id)
            if pdf_data:
                pdfs[pdf_id] = pdf_data

        return pdfs

    def get_latest_pdf_document(self) -> dict:
        """
        Get the most recently uploaded PDF document

        Returns:
            Dictionary containing PDF data or None if no PDFs available
        """
        if (
            not hasattr(st.session_state, 'stored_pdfs')
            or not st.session_state.stored_pdfs
        ):
            return None

        # Get the last PDF ID
        latest_pdf_id = st.session_state.stored_pdfs[-1]

        # Check if this is a batch-processed PDF
        batch_info_key = f"{latest_pdf_id}_batch_info"
        if hasattr(st.session_state, batch_info_key):
            batch_info = getattr(st.session_state, batch_info_key)
            if batch_info.get('batch_processed', False):
                # Return metadata for batch-processed PDF
                return {
                    'pdf_id': latest_pdf_id,
                    'filename': batch_info.get('filename', 'Unknown'),
                    'total_pages': batch_info.get('total_pages', 0),
                    'batch_processed': True,
                    'total_batches': batch_info.get('total_batches', 0),
                    'pages': [],  # Empty to avoid loading all batches
                }

        # Regular PDF
        return self.file_storage.get_pdf(latest_pdf_id)

    def clear_pdf_documents(self):
        """Clear all stored PDF documents from session state"""
        if hasattr(st.session_state, 'stored_pdfs'):
            st.session_state.stored_pdfs = []
            logging.info("Cleared all PDF document references from session state")

    def has_pdf_documents(self) -> bool:
        """
        Check if there are any PDF documents stored in session state

        Returns:
            True if PDFs are available, False otherwise
        """
        return (
            hasattr(st.session_state, 'stored_pdfs')
            and len(st.session_state.stored_pdfs) > 0
        )

    def store_uploaded_image(
        self, image_data: str, filename: str, file_type: str
    ) -> str:
        """
        Store uploaded image externally

        Args:
            image_data: Base64 encoded image data
            filename: Name of the image file
            file_type: MIME type of the image

        Returns:
            Unique image ID for the stored image
        """
        # Ensure session is initialized
        if not hasattr(st.session_state, 'session_id'):
            self.initialize_session_state()

        # Store image externally
        image_id = self.file_storage.store_uploaded_image(
            image_data, filename, file_type, st.session_state.session_id
        )

        # Keep reference in session state
        if 'stored_images' not in st.session_state:
            st.session_state.stored_images = []
        st.session_state.stored_images.append(image_id)

        # Limit stored references
        if len(st.session_state.stored_images) > config.session.MAX_IMAGES_IN_SESSION:
            # Remove oldest images
            removed = st.session_state.stored_images[
                : -config.session.MAX_IMAGES_IN_SESSION
            ]
            st.session_state.stored_images = st.session_state.stored_images[
                -config.session.MAX_IMAGES_IN_SESSION :
            ]

            # Clean up removed images
            for image_id in removed:
                logging.info(f"Removing old image: {image_id}")

        logging.info(f"Stored uploaded image '{filename}' with ID '{image_id}'")
        return image_id

    def get_uploaded_images(self) -> dict:
        """
        Get all stored uploaded images from external storage

        Returns:
            Dictionary of uploaded images keyed by image ID
        """
        # Ensure session state is initialized
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        images = {}
        for image_id in getattr(st.session_state, 'stored_images', []):
            image_data = self.file_storage.get_uploaded_image(image_id)
            if image_data:
                images[image_id] = image_data

        return images

    def get_latest_uploaded_image(self) -> dict:
        """
        Get the most recently uploaded image

        Returns:
            Dictionary containing image data or None if no images available
        """
        if (
            not hasattr(st.session_state, 'stored_images')
            or not st.session_state.stored_images
        ):
            return None

        # Get the last image ID
        latest_image_id = st.session_state.stored_images[-1]
        return self.file_storage.get_uploaded_image(latest_image_id)

    def clear_uploaded_images(self):
        """Clear all stored uploaded images from session state"""
        if hasattr(st.session_state, 'stored_images'):
            st.session_state.stored_images = []
            logging.info("Cleared all uploaded image references from session state")

    def has_uploaded_images(self) -> bool:
        """
        Check if there are any uploaded images stored in session state

        Returns:
            True if images are available, False otherwise
        """
        return (
            hasattr(st.session_state, 'stored_images')
            and len(st.session_state.stored_images) > 0
        )
