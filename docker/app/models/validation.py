"""
Data Validation Layer

Provides centralized validation services for domain models,
including business rule validation and cross-model consistency checks.
"""

import logging
from typing import Any, List, Optional

from .session import FileInfo, Session
from .user import User, UserPreferences

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        code: Optional[str] = None,
    ):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)


class ValidationResult:
    """Validation result container"""

    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []

    def add_error(
        self,
        message: str,
        field: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        """Add a validation error"""
        self.errors.append(ValidationError(message, field, code))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning"""
        self.warnings.append(message)

    def get_error_messages(self) -> List[str]:
        """Get list of error messages"""
        return [error.message for error in self.errors]


class ModelValidator:
    """Base model validator with common validation methods"""

    @staticmethod
    def validate_file_size(size_bytes: int, max_size_mb: int = 100) -> bool:
        """Validate file size"""
        max_size_bytes = max_size_mb * 1024 * 1024
        return 0 <= size_bytes <= max_size_bytes

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename format"""
        if not filename or len(filename) > 255:
            return False
        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        return not any(char in filename for char in invalid_chars)

    @staticmethod
    def validate_json_serializable(data: Any) -> bool:
        """Check if data is JSON serializable"""
        try:
            import json

            json.dumps(data, default=str)
            return True
        except (TypeError, ValueError):
            return False


class UserValidator(ModelValidator):
    """Validator for User domain model"""

    def validate_user(self, user: User) -> ValidationResult:
        """
        Validate a User model

        Args:
            user: User model to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()

        # Validate user ID
        if not user.user_id:
            result.add_error("User ID is required", "user_id", "REQUIRED")
        elif len(user.user_id) < 3:
            result.add_error(
                "User ID must be at least 3 characters",
                "user_id",
                "MIN_LENGTH",
            )

        # Validate preferences
        self._validate_preferences(user.preferences, result)

        # Business rule validations
        if user.message_count > user.preferences.message_limit:
            result.add_error(
                f"Message count ({user.message_count}) exceeds limit ({user.preferences.message_limit})",
                "message_count",
                "BUSINESS_RULE",
            )

        # Validate timestamps
        if user.last_active < user.created_at:
            result.add_error(
                "Last active time cannot be before creation time",
                "last_active",
                "LOGIC_ERROR",
            )

        return result

    def _validate_preferences(
        self, preferences: UserPreferences, result: ValidationResult
    ) -> None:
        """Validate user preferences"""
        if preferences.message_limit <= 0:
            result.add_error(
                "Message limit must be positive",
                "preferences.message_limit",
                "INVALID_VALUE",
            )

        if preferences.message_limit > 10000:
            result.add_warning(
                "Message limit is very high, consider performance implications"
            )


class SessionValidator(ModelValidator):
    """Validator for Session domain model"""

    def validate_session(self, session: Session) -> ValidationResult:
        """
        Validate a Session model

        Args:
            session: Session model to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()

        # Validate session ID
        if not session.session_id:
            result.add_error(
                "Session ID is required", "session_id", "REQUIRED"
            )

        # Validate user ID
        if not session.user_id:
            result.add_error("User ID is required", "user_id", "REQUIRED")

        # Validate file uploads
        for i, file_info in enumerate(session.uploaded_files):
            file_result = self._validate_file_info(file_info)
            for error in file_result.errors:
                result.add_error(
                    f"File {i}: {error.message}",
                    f"uploaded_files[{i}]",
                    error.code,
                )

        # Business rule validations
        if session.message_count < 0:
            result.add_error(
                "Message count cannot be negative",
                "message_count",
                "INVALID_VALUE",
            )

        # Validate timestamps
        if session.updated_at < session.created_at:
            result.add_error(
                "Updated time cannot be before creation time",
                "updated_at",
                "LOGIC_ERROR",
            )

        # Validate context data is JSON serializable
        if not self.validate_json_serializable(session.context_data):
            result.add_error(
                "Context data must be JSON serializable",
                "context_data",
                "SERIALIZATION_ERROR",
            )

        return result

    def _validate_file_info(self, file_info: FileInfo) -> ValidationResult:
        """Validate FileInfo object"""
        result = ValidationResult()

        if not self.validate_filename(file_info.filename):
            result.add_error(
                "Invalid filename format", "filename", "INVALID_FORMAT"
            )

        if not self.validate_file_size(file_info.size_bytes):
            result.add_error(
                "File size exceeds maximum allowed", "size_bytes", "SIZE_LIMIT"
            )

        return result


class CrossModelValidator:
    """Validator for cross-model consistency"""

    def validate_user_session_consistency(
        self, user: User, session: Session
    ) -> ValidationResult:
        """
        Validate consistency between User and Session models

        Args:
            user: User model
            session: Session model

        Returns:
            ValidationResult with consistency checks
        """
        result = ValidationResult()

        # Validate user_id consistency
        if session.user_id != user.user_id:
            result.add_error(
                "Session user_id does not match User user_id",
                "user_id",
                "CONSISTENCY",
            )

        # Validate session_id consistency
        if user.session_id and user.session_id != session.session_id:
            result.add_error(
                "User session_id does not match Session session_id",
                "session_id",
                "CONSISTENCY",
            )

        # Validate message count consistency
        if user.message_count > session.message_count:
            result.add_warning(
                "User message count exceeds session message count - possible data sync issue"
            )

        return result


class ValidationService:
    """
    Centralized validation service for all domain models

    This service coordinates validation across different model types
    and provides a unified validation interface.
    """

    def __init__(self):
        self.user_validator = UserValidator()
        self.session_validator = SessionValidator()
        self.cross_validator = CrossModelValidator()

    def validate_user(self, user: User) -> ValidationResult:
        """Validate User model"""
        return self.user_validator.validate_user(user)

    def validate_session(self, session: Session) -> ValidationResult:
        """Validate Session model"""
        return self.session_validator.validate_session(session)

    def validate_user_session_pair(
        self, user: User, session: Session
    ) -> ValidationResult:
        """Validate User and Session consistency"""
        # First validate individual models
        user_result = self.validate_user(user)
        session_result = self.validate_session(session)

        # Then validate cross-model consistency
        consistency_result = (
            self.cross_validator.validate_user_session_consistency(
                user, session
            )
        )

        # Combine results
        combined_result = ValidationResult()
        combined_result.errors.extend(user_result.errors)
        combined_result.errors.extend(session_result.errors)
        combined_result.errors.extend(consistency_result.errors)
        combined_result.warnings.extend(user_result.warnings)
        combined_result.warnings.extend(session_result.warnings)
        combined_result.warnings.extend(consistency_result.warnings)

        combined_result.is_valid = not combined_result.errors

        return combined_result


# Singleton instance for global use
validation_service = ValidationService()
