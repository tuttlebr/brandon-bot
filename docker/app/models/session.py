"""
Session Domain Model

Represents a chat session entity with proper validation,
state management, and business logic encapsulation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Session status enumeration"""

    ACTIVE = "active"


class ProcessingStatus(str, Enum):
    """Processing status for various operations"""

    IDLE = "idle"
    PROCESSING = "processing"


class FileInfo(BaseModel):
    """Information about uploaded files"""

    filename: str = Field(..., description="Original filename")
    file_id: str = Field(..., description="Unique file identifier")
    file_type: str = Field(..., description="File type (pdf, image, etc.)")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    uploaded_at: datetime = Field(
        default_factory=datetime.now, description="Upload timestamp"
    )
    processed: bool = Field(
        default=False, description="Whether file has been processed"
    )


class Session(BaseModel):
    """
    Session domain model representing a chat session

    This model encapsulates session data, state management,
    and business rules for chat interactions.
    """

    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="Associated user identifier")
    status: SessionStatus = Field(
        default=SessionStatus.ACTIVE, description="Current session status"
    )
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.IDLE, description="Current processing status"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Session creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last update timestamp"
    )
    message_count: int = Field(
        default=0, ge=0, description="Number of messages in session"
    )
    uploaded_files: List[FileInfo] = Field(
        default_factory=list, description="Files uploaded in this session"
    )
    context_data: Dict[str, Any] = Field(
        default_factory=dict, description="Session context and metadata"
    )
    llm_model_name: str = Field(default="", description="LLM model being used")
    fast_llm_model_name: str = Field(
        default="", description="Fast LLM model being used"
    )
    intelligent_llm_model_name: str = Field(
        default="", description="Intelligent LLM model being used"
    )
    vlm_model_name: str = Field(
        default="", description="Vision LLM model being used"
    )

    def update_timestamp(self) -> None:
        """Update the session's last update timestamp"""
        self.updated_at = datetime.now()

    def increment_message_count(self) -> None:
        """Increment message count and update timestamp"""
        self.message_count += 1
        self.update_timestamp()

    def add_uploaded_file(
        self, filename: str, file_id: str, file_type: str, size_bytes: int
    ) -> FileInfo:
        """
        Add an uploaded file to the session

        Args:
            filename: Original filename
            file_id: Unique file identifier
            file_type: File type
            size_bytes: File size in bytes

        Returns:
            FileInfo object for the uploaded file
        """
        file_info = FileInfo(
            filename=filename,
            file_id=file_id,
            file_type=file_type,
            size_bytes=size_bytes,
        )
        self.uploaded_files.append(file_info)
        self.update_timestamp()
        return file_info

    def mark_file_processed(self, file_id: str) -> bool:
        """
        Mark a file as processed

        Args:
            file_id: File identifier to mark as processed

        Returns:
            True if file was found and marked, False otherwise
        """
        for file_info in self.uploaded_files:
            if file_info.file_id == file_id:
                file_info.processed = True
                self.update_timestamp()
                return True
        return False

    def get_files_by_type(self, file_type: str) -> List[FileInfo]:
        """
        Get files of a specific type

        Args:
            file_type: Type of files to retrieve

        Returns:
            List of FileInfo objects of the specified type
        """
        return [
            f for f in self.uploaded_files if f.file_type == file_type.lower()
        ]

    def set_processing_status(self, status: ProcessingStatus) -> None:
        """
        Set the processing status

        Args:
            status: New processing status
        """
        self.processing_status = status
        self.update_timestamp()

    def is_processing(self) -> bool:
        """
        Check if session is currently processing

        Returns:
            True if session is in any processing state
        """
        return self.processing_status != ProcessingStatus.IDLE

    def set_context(self, key: str, value: Any) -> None:
        """
        Set context data

        Args:
            key: Context key
            value: Context value
        """
        self.context_data[key] = value
        self.update_timestamp()

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get context data

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.context_data.get(key, default)

    def clear_context(self) -> None:
        """Clear all context data"""
        self.context_data.clear()
        self.update_timestamp()

    def to_streamlit_state(self) -> Dict[str, Any]:
        """
        Convert session to Streamlit session state format

        Returns:
            Dictionary suitable for Streamlit session state
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status,
            "processing_status": self.processing_status,
            "message_count": self.message_count,
            "uploaded_files": [f.dict() for f in self.uploaded_files],
            "context_data": self.context_data,
            "llm_model_name": self.llm_model_name,
            "fast_llm_model_name": self.fast_llm_model_name,
            "intelligent_llm_model_name": self.intelligent_llm_model_name,
            "vlm_model_name": self.vlm_model_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_streamlit_state(cls, state_data: Dict[str, Any]) -> "Session":
        """
        Create session from Streamlit session state

        Args:
            state_data: Session state dictionary

        Returns:
            Session model instance
        """
        # Parse uploaded files
        uploaded_files = []
        for file_data in state_data.get("uploaded_files", []):
            uploaded_files.append(FileInfo(**file_data))

        # Parse timestamps
        created_at = datetime.fromisoformat(
            state_data.get("created_at", datetime.now().isoformat())
        )
        updated_at = datetime.fromisoformat(
            state_data.get("updated_at", datetime.now().isoformat())
        )

        return cls(
            session_id=state_data.get("session_id", ""),
            user_id=state_data.get("user_id", "anonymous"),
            status=state_data.get("status", SessionStatus.ACTIVE),
            processing_status=state_data.get(
                "processing_status", ProcessingStatus.IDLE
            ),
            created_at=created_at,
            updated_at=updated_at,
            message_count=state_data.get("message_count", 0),
            uploaded_files=uploaded_files,
            context_data=state_data.get("context_data", {}),
            llm_model_name=state_data.get("llm_model_name", ""),
            fast_llm_model_name=state_data.get("fast_llm_model_name", ""),
            intelligent_llm_model_name=state_data.get(
                "intelligent_llm_model_name", ""
            ),
            vlm_model_name=state_data.get("vlm_model_name", ""),
        )
