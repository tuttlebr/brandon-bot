"""
Data Validation Layer

Provides centralized validation services for domain models,
including business rule validation and cross-model consistency checks.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .chat_message import ChatMessage
from .session import FileInfo, ProcessingStatus, Session, SessionStatus
from .tool_entity import ToolEntity, ToolParameter, ToolStatus, ToolType
from .user import User, UserPreferences, UserRole

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""

    def __init__(
        self, message: str, field: Optional[str] = None, code: Optional[str] = None
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
        self, message: str, field: Optional[str] = None, code: Optional[str] = None
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

    def get_errors_by_field(self, field: str) -> List[ValidationError]:
        """Get errors for a specific field"""
        return [error for error in self.errors if error.field == field]


class ModelValidator:
    """Base model validator with common validation methods"""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
        return re.match(pattern, url) is not None

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
                "User ID must be at least 3 characters", "user_id", "MIN_LENGTH"
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
            result.add_error("Session ID is required", "session_id", "REQUIRED")

        # Validate user ID
        if not session.user_id:
            result.add_error("User ID is required", "user_id", "REQUIRED")

        # Validate file uploads
        for i, file_info in enumerate(session.uploaded_files):
            file_result = self._validate_file_info(file_info)
            for error in file_result.errors:
                result.add_error(
                    f"File {i}: {error.message}", f"uploaded_files[{i}]", error.code
                )

        # Business rule validations
        if session.message_count < 0:
            result.add_error(
                "Message count cannot be negative", "message_count", "INVALID_VALUE"
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
            result.add_error("Invalid filename format", "filename", "INVALID_FORMAT")

        if not self.validate_file_size(file_info.size_bytes):
            result.add_error(
                "File size exceeds maximum allowed", "size_bytes", "SIZE_LIMIT"
            )

        return result


class ToolValidator(ModelValidator):
    """Validator for Tool domain model"""

    def validate_tool(self, tool: ToolEntity) -> ValidationResult:
        """
        Validate a ToolEntity model

        Args:
            tool: ToolEntity model to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()

        # Validate tool name uniqueness (would need registry access in real implementation)
        if not tool.name:
            result.add_error("Tool name is required", "name", "REQUIRED")

        # Validate parameters
        parameter_names = set()
        for i, param in enumerate(tool.parameters):
            param_result = self._validate_parameter(param)
            for error in param_result.errors:
                result.add_error(
                    f"Parameter {i}: {error.message}", f"parameters[{i}]", error.code
                )

            # Check for duplicate parameter names
            if param.name in parameter_names:
                result.add_error(
                    f"Duplicate parameter name: {param.name}",
                    f"parameters[{i}].name",
                    "DUPLICATE",
                )
            parameter_names.add(param.name)

        # Validate configuration is JSON serializable
        if not self.validate_json_serializable(tool.configuration):
            result.add_error(
                "Configuration must be JSON serializable",
                "configuration",
                "SERIALIZATION_ERROR",
            )

        return result

    def _validate_parameter(self, parameter: ToolParameter) -> ValidationResult:
        """Validate ToolParameter object"""
        result = ValidationResult()

        # Validate parameter type
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        if parameter.type not in valid_types:
            result.add_error(
                f"Invalid parameter type: {parameter.type}", "type", "INVALID_TYPE"
            )

        # Validate enum values if provided
        if parameter.enum_values and parameter.type != "string":
            result.add_warning("Enum values typically used with string type parameters")

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
            ValidationResult with validation details
        """
        result = ValidationResult()

        # Check user ID consistency
        if user.user_id != session.user_id:
            result.add_error(
                "User ID mismatch between User and Session",
                "user_id",
                "CONSISTENCY_ERROR",
            )

        # Check session ID consistency
        if user.session_id and user.session_id != session.session_id:
            result.add_error("Session ID mismatch", "session_id", "CONSISTENCY_ERROR")

        # Check message count consistency
        if user.message_count != session.message_count:
            result.add_warning("Message count mismatch between User and Session")

        return result

    def validate_file_session_consistency(
        self, session: Session, uploaded_files: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate consistency between Session and actual uploaded files

        Args:
            session: Session model
            uploaded_files: List of actual uploaded file metadata

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult()

        session_file_ids = {f.file_id for f in session.uploaded_files}
        actual_file_ids = {f.get("file_id") for f in uploaded_files}

        # Check for missing files
        missing_files = session_file_ids - actual_file_ids
        if missing_files:
            result.add_error(
                f"Session references missing files: {missing_files}",
                "uploaded_files",
                "MISSING_FILES",
            )

        # Check for extra files
        extra_files = actual_file_ids - session_file_ids
        if extra_files:
            result.add_warning(f"Files exist but not tracked in session: {extra_files}")

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
        self.tool_validator = ToolValidator()
        self.cross_validator = CrossModelValidator()

    def validate_user(self, user: User) -> ValidationResult:
        """Validate User model"""
        return self.user_validator.validate_user(user)

    def validate_session(self, session: Session) -> ValidationResult:
        """Validate Session model"""
        return self.session_validator.validate_session(session)

    def validate_tool(self, tool: ToolEntity) -> ValidationResult:
        """Validate ToolEntity model"""
        return self.tool_validator.validate_tool(tool)

    def validate_user_session_pair(
        self, user: User, session: Session
    ) -> ValidationResult:
        """Validate User and Session consistency"""
        # First validate individual models
        user_result = self.validate_user(user)
        session_result = self.validate_session(session)

        # Then validate cross-model consistency
        consistency_result = self.cross_validator.validate_user_session_consistency(
            user, session
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

    def validate_batch(
        self, models: List[Union[User, Session, ToolEntity]]
    ) -> Dict[str, ValidationResult]:
        """
        Validate multiple models at once

        Args:
            models: List of model instances to validate

        Returns:
            Dictionary mapping model ID to ValidationResult
        """
        results = {}

        for i, model in enumerate(models):
            try:
                if isinstance(model, User):
                    results[f"user_{i}_{model.user_id}"] = self.validate_user(model)
                elif isinstance(model, Session):
                    results[f"session_{i}_{model.session_id}"] = self.validate_session(
                        model
                    )
                elif isinstance(model, ToolEntity):
                    results[f"tool_{i}_{model.tool_id}"] = self.validate_tool(model)
                else:
                    results[f"unknown_{i}"] = ValidationResult()
                    results[f"unknown_{i}"].add_error(
                        f"Unknown model type: {type(model)}"
                    )
            except Exception as e:
                logger.error(f"Validation error for model {i}: {e}")
                results[f"error_{i}"] = ValidationResult()
                results[f"error_{i}"].add_error(f"Validation exception: {str(e)}")

        return results


# Singleton instance for global use
validation_service = ValidationService()
