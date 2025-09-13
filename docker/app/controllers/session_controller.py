import logging
import random
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from models import (
    ChatConfig,
    FileInfo,
    ProcessingStatus,
    Session,
    SessionStatus,
    User,
    UserRole,
    validation_service,
)
from services.file_storage_service import FileStorageService
from ui.view_helpers import view_factory
from utils.config import config
from utils.system_prompts import get_system_prompt


class SessionController:
    """
    Enhanced controller for managing session state and operations

    Now uses domain models and reduced framework coupling through
    view abstractions while maintaining backward compatibility.
    """

    def __init__(self, config_obj: ChatConfig):
        """
        Initialize the session controller

        Args:
            config_obj: Application configuration
        """
        self.config_obj = config_obj
        self.file_storage = FileStorageService()

        # Initialize view helpers for UI operations
        self.message_helper = view_factory.create_message_helper()
        self.progress_helper = view_factory.create_progress_helper()

        # Cache for current session and user models
        self._current_session: Optional[Session] = None
        self._current_user: Optional[User] = None

    def initialize_session_state(self) -> Session:
        """
        Initialize session state using domain models

        Returns:
            Initialized Session domain model
        """
        # Use atomic check-and-set to prevent race conditions
        if not getattr(st.session_state, "initialized", False):
            st.session_state.initialized = True

            # Create or get session ID
            session_id = self._get_or_create_session_id()
            user_id = self._get_or_create_user_id()

            # Create domain models
            session = Session(
                session_id=session_id,
                user_id=user_id,
                status=SessionStatus.ACTIVE,
                processing_status=ProcessingStatus.IDLE,
                llm_model_name=self.config_obj.llm_model_name,
                fast_llm_model_name=self.config_obj.fast_llm_model_name,
                intelligent_llm_model_name=(
                    self.config_obj.intelligent_llm_model_name
                ),
                vlm_model_name=self.config_obj.vlm_model_name,
            )

            user = User(
                user_id=user_id, session_id=session_id, role=UserRole.USER
            )

            # Validate models
            validation_result = validation_service.validate_user_session_pair(
                user, session
            )
            if not validation_result.is_valid:
                logging.warning(
                    "Session validation warnings: "
                    f"{validation_result.get_error_messages()}"
                )

            # Store models in session state
            self._store_session_state(session, user)

            # Initialize UI state
            if not hasattr(st.session_state, "current_page"):
                st.session_state.current_page = config.ui.CURRENT_PAGE_DEFAULT
                logging.info(
                    "Initialized current_page to "
                    f"{config.ui.CURRENT_PAGE_DEFAULT}"
                )

            # Initialize system prompt with full tools list
            # Ensure tools are initialized first
            from tools.registry import get_all_tool_definitions

            if len(get_all_tool_definitions()) == 0:
                logging.info(
                    "No tools found when initializing session, "
                    "attempting to initialize tools"
                )
                from tools.initialize_tools import initialize_all_tools

                initialize_all_tools()

            st.session_state.system_prompt = get_system_prompt()
            logging.info(
                "Initialized system prompt with available tools "
                f"({len(st.session_state.system_prompt)} chars)"
            )

            # Cache models
            self._current_session = session
            self._current_user = user

            logging.info("Initialized new session: %s", session_id)

            return session

        else:
            # Session already initialized, load existing models
            return self.get_current_session()

    def _get_or_create_session_id(self) -> str:
        """Get existing session ID or create new one"""
        if (
            hasattr(st.session_state, "session_id")
            and st.session_state.session_id
        ):
            return st.session_state.session_id

        return (
            f"session_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        )

    def _get_or_create_user_id(self) -> str:
        """Get existing user ID or create new one"""
        if hasattr(st.session_state, "user_id") and st.session_state.user_id:
            return st.session_state.user_id

        return f"user_{int(time.time() * 1000)}_{random.randint(100, 999)}"

    def _store_session_state(self, session: Session, user: User) -> None:
        """Store domain models in session state"""
        # Convert models to session state format
        session_data = session.to_streamlit_state()
        user_data = user.to_session_state()

        # Store in Streamlit session state
        for key, value in session_data.items():
            setattr(st.session_state, key, value)

        for key, value in user_data.items():
            if not key.startswith("session_"):  # Avoid duplication
                setattr(st.session_state, f"user_{key}", value)

    def get_current_session(self) -> Session:
        """
        Get current session domain model

        Returns:
            Current Session model
        """
        if self._current_session is None:
            # Load from session state
            if hasattr(st.session_state, "session_id"):
                session_data = {
                    "session_id": getattr(st.session_state, "session_id", ""),
                    "user_id": getattr(st.session_state, "user_id", ""),
                    "status": getattr(
                        st.session_state, "status", SessionStatus.ACTIVE
                    ),
                    "processing_status": getattr(
                        st.session_state,
                        "processing_status",
                        ProcessingStatus.IDLE,
                    ),
                    "message_count": getattr(
                        st.session_state, "message_count", 0
                    ),
                    "uploaded_files": getattr(
                        st.session_state, "uploaded_files", []
                    ),
                    "context_data": getattr(
                        st.session_state, "context_data", {}
                    ),
                    "llm_model_name": getattr(
                        st.session_state, "llm_model_name", ""
                    ),
                    "fast_llm_model_name": getattr(
                        st.session_state, "fast_llm_model_name", ""
                    ),
                    "intelligent_llm_model_name": getattr(
                        st.session_state, "intelligent_llm_model_name", ""
                    ),
                    "vlm_model_name": getattr(
                        st.session_state, "vlm_model_name", ""
                    ),
                    "created_at": getattr(
                        st.session_state, "created_at", time.time()
                    ),
                    "updated_at": getattr(
                        st.session_state, "updated_at", time.time()
                    ),
                }

                try:
                    self._current_session = Session.from_streamlit_state(
                        session_data
                    )
                except Exception as e:
                    logging.error("Error loading session from state: %s", e)
                    # Fallback to creating new session
                    return self.initialize_session_state()
            else:
                # No session in state, initialize new one
                return self.initialize_session_state()

        return self._current_session

    def get_current_user(self) -> User:
        """
        Get current user domain model

        Returns:
            Current User model
        """
        if self._current_user is None:
            # Load from session state
            user_data = {
                "user_id": getattr(
                    st.session_state,
                    "user_user_id",
                    getattr(st.session_state, "user_id", "anonymous"),
                ),
                "session_id": getattr(st.session_state, "session_id", ""),
                "role": getattr(st.session_state, "user_role", UserRole.USER),
                "preferences": getattr(
                    st.session_state, "user_preferences", {}
                ),
                "message_count": getattr(
                    st.session_state, "user_message_count", 0
                ),
                "session_count": getattr(
                    st.session_state, "user_session_count", 0
                ),
            }

            try:
                self._current_user = User.from_session_state(user_data)
            except Exception as e:
                logging.error("Error loading user from state: %s", e)
                # Create default user
                self._current_user = User(user_id=user_data["user_id"])

        return self._current_user

    def set_processing_status(self, status: ProcessingStatus) -> None:
        """
        Set processing status using domain model

        Args:
            status: New processing status
        """
        session = self.get_current_session()
        session.set_processing_status(status)

        # Update legacy field for backward compatibility
        st.session_state.processing = status != ProcessingStatus.IDLE

        # Update cache and session state
        self._current_session = session
        self._store_session_state(session, self.get_current_user())

    def set_processing_state(self, processing: bool) -> None:
        """
        Set processing state

        Args:
            processing: Whether app is processing
        """
        status = (
            ProcessingStatus.PROCESSING
            if processing
            else ProcessingStatus.IDLE
        )
        self.set_processing_status(status)

    def is_processing(self) -> bool:
        """
        Check if session is currently processing

        Returns:
            True if processing, False otherwise
        """
        session = self.get_current_session()
        return session.is_processing()

    def add_uploaded_file(
        self, filename: str, file_data: Any, file_type: str
    ) -> FileInfo:
        """
        Add uploaded file using domain model

        Args:
            filename: Name of uploaded file
            file_data: File data object
            file_type: Type of file (pdf, image, etc.)

        Returns:
            FileInfo object for the uploaded file
        """
        session = self.get_current_session()

        # Generate file ID
        file_id = f"file_{int(time.time() * 1000)}_{random.randint(100, 999)}"
        file_size = getattr(file_data, "size", 0) if file_data else 0

        # Add file to session
        file_info = session.add_uploaded_file(
            filename, file_id, file_type, file_size
        )

        # Update cache and session state
        self._current_session = session
        self._store_session_state(session, self.get_current_user())

        logging.info("Added uploaded file: %s (%s)", filename, file_type)
        return file_info

    def mark_file_processed(self, file_id: str) -> bool:
        """
        Mark file as processed using domain model

        Args:
            file_id: File identifier

        Returns:
            True if file was found and marked
        """
        session = self.get_current_session()
        success = session.mark_file_processed(file_id)

        if success:
            # Update cache and session state
            self._current_session = session
            self._store_session_state(session, self.get_current_user())
            logging.info("Marked file as processed: %s", file_id)

        return success

    def get_files_by_type(self, file_type: str) -> List[FileInfo]:
        """
        Get files of specific type

        Args:
            file_type: Type of files to retrieve

        Returns:
            List of FileInfo objects
        """
        session = self.get_current_session()
        return session.get_files_by_type(file_type)

    def set_context(self, key: str, value: Any) -> None:
        """
        Set context data using domain model

        Args:
            key: Context key
            value: Context value
        """
        session = self.get_current_session()
        session.set_context(key, value)

        # Update cache and session state
        self._current_session = session
        self._store_session_state(session, self.get_current_user())

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get context data using domain model

        Args:
            key: Context key
            default: Default value

        Returns:
            Context value or default
        """
        session = self.get_current_session()
        return session.get_context(key, default)

    def clear_context(self) -> None:
        """Clear all context data"""
        session = self.get_current_session()
        session.clear_context()

        # Update cache and session state
        self._current_session = session
        self._store_session_state(session, self.get_current_user())

    def increment_message_count(self) -> None:
        """Increment message count in both session and user models"""
        session = self.get_current_session()
        user = self.get_current_user()

        session.increment_message_count()
        user.increment_message_count()

        # Update cache and session state
        self._current_session = session
        self._current_user = user
        self._store_session_state(session, user)

    def cleanup_session(self) -> None:
        """Clean up session data including external files"""
        session = self.get_current_session()

        if session.session_id:
            self.file_storage.cleanup_session(session.session_id)
            logging.info("Cleaned up session: %s", session.session_id)

        # Clear cached models
        self._current_session = None
        self._current_user = None

    def get_system_prompt(self) -> str:
        """
        Get the system prompt from session state

        Returns:
            The system prompt with full tools list
        """
        # Get from session state if available
        system_prompt = getattr(st.session_state, "system_prompt", None)

        # If not in session state, get it and store it
        if not system_prompt:
            system_prompt = get_system_prompt()
            st.session_state.system_prompt = system_prompt
            logging.info(
                "System prompt not found in session state, "
                "initialized with tools list"
            )

        return system_prompt

    def store_tool_context(self, context: str) -> None:
        """Store tool context in session state"""
        st.session_state.last_tool_context = context

    def clear_tool_context(self) -> None:
        """Clear tool context from session state"""
        if hasattr(st.session_state, "last_tool_context"):
            st.session_state.last_tool_context = None

    def store_generated_image(
        self, image_data: str, enhanced_prompt: str, original_prompt: str
    ) -> str:
        """Store AI-generated image

        Args:
            image_data: Base64 encoded image data
            enhanced_prompt: Enhanced prompt used for generation
            original_prompt: Original user prompt

        Returns:
            Image ID for retrieval
        """
        session = self.get_current_session()

        image_id = self.file_storage.store_image(
            image_data, enhanced_prompt, original_prompt, session.session_id
        )

        if "stored_images" not in st.session_state:
            st.session_state.stored_images = []
        st.session_state.stored_images.append(image_id)

        if (
            len(st.session_state.stored_images)
            > config.session.MAX_IMAGES_IN_SESSION
        ):
            st.session_state.stored_images = st.session_state.stored_images[
                -config.session.MAX_IMAGES_IN_SESSION :
            ]

        return image_id

    def get_model_name(self, model_type: str = "fast") -> str:
        """
        Get model name using domain model

        Args:
            model_type: Type of model to get

        Returns:
            Model name string
        """
        session = self.get_current_session()

        model_mapping = {
            "fast": session.fast_llm_model_name,
            "llm": session.llm_model_name,
            "intelligent": session.intelligent_llm_model_name,
            "vlm": session.vlm_model_name,
        }

        model_name = model_mapping.get(model_type)
        if not model_name:
            logging.error("No %s model name found", model_type)
            # Fallback to config
            fallback_mapping = {
                "fast": self.config_obj.fast_llm_model_name,
                "llm": self.config_obj.llm_model_name,
                "intelligent": self.config_obj.intelligent_llm_model_name,
                "vlm": self.config_obj.vlm_model_name,
            }
            model_name = fallback_mapping.get(
                model_type, self.config_obj.llm_model_name
            )

        return model_name

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get messages from session state

        Returns:
            List of message dictionaries
        """
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        # Initialize messages if not present
        if not hasattr(st.session_state, "messages"):
            # Use system prompt from session state if available,
            # otherwise get it
            system_prompt = getattr(st.session_state, "system_prompt", None)
            if not system_prompt:
                system_prompt = get_system_prompt()
                st.session_state.system_prompt = system_prompt

            st.session_state.messages = [
                {"role": "system", "content": system_prompt}
            ]

        return getattr(st.session_state, "messages", [])

    def add_message(self, role: str, content: Any) -> None:
        """Add message to session state

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        if not getattr(st.session_state, "initialized", False):
            self.initialize_session_state()

        if not hasattr(st.session_state, "messages"):
            st.session_state.messages = [
                {"role": "system", "content": get_system_prompt()}
            ]

        st.session_state.messages.append({"role": role, "content": content})
        self.increment_message_count()

    def get_latest_pdf_document(self) -> Optional[Dict[str, Any]]:
        """Get the most recently uploaded PDF document

        Returns:
            PDF data dictionary or None
        """
        if (
            not hasattr(st.session_state, "stored_pdfs")
            or not st.session_state.stored_pdfs
        ):
            return None

        latest_pdf_id = st.session_state.stored_pdfs[-1]
        batch_info_key = f"{latest_pdf_id}_batch_info"

        if hasattr(st.session_state, batch_info_key):
            batch_info = getattr(st.session_state, batch_info_key)
            if batch_info.get("batch_processed", False):
                return {
                    "pdf_id": latest_pdf_id,
                    "filename": batch_info.get("filename", "Unknown"),
                    "total_pages": batch_info.get("total_pages", 0),
                    "batch_processed": True,
                    "total_batches": batch_info.get("total_batches", 0),
                    "pages": [],
                }

        return self.file_storage.get_pdf(latest_pdf_id)

    def clear_pdf_documents(self) -> None:
        """Clear all PDF documents from session"""
        if hasattr(st.session_state, "stored_pdfs"):
            st.session_state.stored_pdfs = []
            logging.info(
                "Cleared all PDF document references from session state"
            )

    def has_pdf_documents(self) -> bool:
        """Check if there are any PDF documents in session

        Returns:
            True if PDFs are available, False otherwise
        """
        return (
            hasattr(st.session_state, "stored_pdfs")
            and len(st.session_state.stored_pdfs) > 0
        )

    def store_uploaded_image(
        self, image_data: str, filename: str, file_type: str
    ) -> str:
        """Store user-uploaded image

        Args:
            image_data: Base64 encoded image data
            filename: Original filename
            file_type: Image file type

        Returns:
            Image ID for retrieval
        """
        session = self.get_current_session()

        image_id = self.file_storage.store_uploaded_image(
            image_data, filename, file_type, session.session_id
        )

        if "stored_images" not in st.session_state:
            st.session_state.stored_images = []
        st.session_state.stored_images.append(image_id)

        if (
            len(st.session_state.stored_images)
            > config.session.MAX_IMAGES_IN_SESSION
        ):
            removed = st.session_state.stored_images[
                : -config.session.MAX_IMAGES_IN_SESSION
            ]
            st.session_state.stored_images = st.session_state.stored_images[
                -config.session.MAX_IMAGES_IN_SESSION :
            ]

            for image_id in removed:
                logging.info("Removing old image: %s", image_id)

        logging.info(
            "Stored uploaded image '%s' with ID '%s'", filename, image_id
        )
        return image_id

    def get_latest_uploaded_image(self) -> Optional[Dict[str, Any]]:
        """Get the most recently uploaded image

        Returns:
            Image data dictionary or None
        """
        if (
            not hasattr(st.session_state, "stored_images")
            or not st.session_state.stored_images
        ):
            return None

        latest_image_id = st.session_state.stored_images[-1]
        return self.file_storage.get_uploaded_image(latest_image_id)

    def clear_uploaded_images(self) -> None:
        """Clear all uploaded images from session"""
        if hasattr(st.session_state, "stored_images"):
            st.session_state.stored_images = []
            logging.info(
                "Cleared all uploaded image references from session state"
            )

    def has_uploaded_images(self) -> bool:
        """
        Check if there are any uploaded images in session

        Returns:
            True if images are available, False otherwise
        """
        return (
            hasattr(st.session_state, "stored_images")
            and len(st.session_state.stored_images) > 0
        )
