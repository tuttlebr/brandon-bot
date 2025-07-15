"""
User Domain Model

Represents a user entity in the chatbot system with proper validation
and business logic encapsulation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class UserRole(str, Enum):
    """User role enumeration"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class UserPreferences(BaseModel):
    """User preferences and settings"""

    language: str = Field(default="en", description="Preferred language code")
    theme: str = Field(default="light", description="UI theme preference")
    message_limit: int = Field(
        default=100, ge=1, le=1000, description="Messages per session limit"
    )
    auto_save: bool = Field(default=True, description="Auto-save conversations")

    @validator('language')
    def validate_language(cls, v):
        """Validate language code format"""
        if len(v) not in [2, 5]:  # ISO 639-1 or locale format
            raise ValueError(
                "Language must be ISO 639-1 format (e.g., 'en') or locale format (e.g., 'en-US')"
            )
        return v.lower()

    @validator('theme')
    def validate_theme(cls, v):
        """Validate theme options"""
        allowed_themes = {"light", "dark", "auto"}
        if v not in allowed_themes:
            raise ValueError(f"Theme must be one of {allowed_themes}")
        return v


class User(BaseModel):
    """
    User domain model representing a chatbot user

    This model encapsulates user data and business rules,
    providing validation and behavior methods.
    """

    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Current session identifier")
    role: UserRole = Field(default=UserRole.USER, description="User role in the system")
    preferences: UserPreferences = Field(
        default_factory=UserPreferences, description="User preferences"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="User creation timestamp"
    )
    last_active: datetime = Field(
        default_factory=datetime.now, description="Last activity timestamp"
    )
    message_count: int = Field(default=0, ge=0, description="Total messages sent")
    session_count: int = Field(default=0, ge=0, description="Total sessions created")

    class Config:
        """Pydantic configuration"""

        use_enum_values = True
        validate_assignment = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate user ID format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("User ID cannot be empty")
        if len(v) > 100:
            raise ValueError("User ID cannot exceed 100 characters")
        return v.strip()

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_active = datetime.now()

    def increment_message_count(self) -> None:
        """Increment user message count"""
        self.message_count += 1
        self.update_activity()

    def increment_session_count(self) -> None:
        """Increment user session count"""
        self.session_count += 1
        self.update_activity()

    def can_send_message(self) -> bool:
        """
        Check if user can send another message based on preferences

        Returns:
            True if user can send message, False otherwise
        """
        return self.message_count < self.preferences.message_limit

    def get_display_name(self) -> str:
        """
        Get user display name for UI

        Returns:
            Formatted display name
        """
        if self.role == UserRole.ASSISTANT:
            return "Assistant"
        elif self.role == UserRole.SYSTEM:
            return "System"
        else:
            return f"User {self.user_id[:8]}"

    def to_session_state(self) -> Dict[str, Any]:
        """
        Convert user model to session state format

        Returns:
            Dictionary suitable for Streamlit session state
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "preferences": self.preferences.dict(),
            "message_count": self.message_count,
            "session_count": self.session_count,
        }

    @classmethod
    def from_session_state(cls, session_data: Dict[str, Any]) -> "User":
        """
        Create user model from session state data

        Args:
            session_data: Session state dictionary

        Returns:
            User model instance
        """
        preferences_data = session_data.get("preferences", {})
        preferences = UserPreferences(**preferences_data)

        return cls(
            user_id=session_data.get("user_id", "anonymous"),
            session_id=session_data.get("session_id"),
            role=session_data.get("role", UserRole.USER),
            preferences=preferences,
            message_count=session_data.get("message_count", 0),
            session_count=session_data.get("session_count", 0),
        )
