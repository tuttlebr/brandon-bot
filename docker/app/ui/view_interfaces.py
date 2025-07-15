"""
View Interfaces

Defines contracts for UI operations to reduce framework coupling
and enable easier testing and framework switching.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from models import FileInfo, Session, User

from .view_helpers import FileUploadResult, UIMessage, UserInputResult


class DisplayMode(str, Enum):
    """Display mode enumeration"""

    NORMAL = "normal"
    COMPACT = "compact"
    DETAILED = "detailed"


class ChatRole(str, Enum):
    """Chat role enumeration"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class IChatDisplayInterface(ABC):
    """
    Interface for chat display operations

    Defines contract for displaying chat messages, history,
    and managing chat-specific UI interactions.
    """

    @abstractmethod
    def display_message(
        self,
        role: ChatRole,
        content: str,
        avatar: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Display a chat message"""
        pass

    @abstractmethod
    def display_message_history(
        self, messages: List[Dict[str, Any]], page_size: int = 25, current_page: int = 0
    ) -> None:
        """Display chat message history with pagination"""
        pass

    @abstractmethod
    def get_user_message(
        self, placeholder: str = "Type your message..."
    ) -> Optional[str]:
        """Get user message input"""
        pass

    @abstractmethod
    def show_typing_indicator(self, message: str = "Assistant is typing...") -> Any:
        """Show typing indicator"""
        pass

    @abstractmethod
    def clear_chat_display(self) -> None:
        """Clear chat display"""
        pass


class IFileManagementInterface(ABC):
    """
    Interface for file management operations

    Defines contract for file upload, display, and management
    within the UI.
    """

    @abstractmethod
    def show_file_uploader(
        self,
        accepted_types: List[str],
        max_size_mb: int = 100,
        multiple_files: bool = False,
        help_text: str = "",
    ) -> FileUploadResult:
        """Show file uploader widget"""
        pass

    @abstractmethod
    def display_file_info(self, file_info: FileInfo, show_actions: bool = True) -> None:
        """Display file information"""
        pass

    @abstractmethod
    def display_file_list(
        self, files: List[FileInfo], display_mode: DisplayMode = DisplayMode.NORMAL
    ) -> None:
        """Display list of files"""
        pass

    @abstractmethod
    def show_file_processing_status(
        self, filename: str, status: str, progress: Optional[float] = None
    ) -> None:
        """Show file processing status"""
        pass

    @abstractmethod
    def show_file_actions(self, file_info: FileInfo) -> Dict[str, bool]:
        """Show file action buttons and return which were clicked"""
        pass


class ISessionManagementInterface(ABC):
    """
    Interface for session management operations

    Defines contract for displaying session information
    and managing session state in the UI.
    """

    @abstractmethod
    def display_session_info(
        self, session: Session, display_mode: DisplayMode = DisplayMode.COMPACT
    ) -> None:
        """Display session information"""
        pass

    @abstractmethod
    def show_session_status(self, session: Session) -> None:
        """Show current session status"""
        pass

    @abstractmethod
    def display_session_metrics(self, session: Session) -> None:
        """Display session metrics and statistics"""
        pass

    @abstractmethod
    def show_session_actions(self) -> Dict[str, bool]:
        """Show session action buttons"""
        pass


class INotificationInterface(ABC):
    """
    Interface for notification operations

    Defines contract for displaying various types of
    notifications and alerts to users.
    """

    @abstractmethod
    def show_success_notification(
        self, message: str, duration: Optional[int] = None
    ) -> None:
        """Show success notification"""
        pass

    @abstractmethod
    def show_error_notification(
        self, message: str, error_details: Optional[str] = None
    ) -> None:
        """Show error notification"""
        pass

    @abstractmethod
    def show_warning_notification(self, message: str) -> None:
        """Show warning notification"""
        pass

    @abstractmethod
    def show_info_notification(self, message: str) -> None:
        """Show info notification"""
        pass

    @abstractmethod
    def show_progress_notification(self, message: str, progress: float) -> None:
        """Show progress notification"""
        pass


class ILayoutInterface(ABC):
    """
    Interface for layout operations

    Defines contract for creating and managing
    UI layout structures.
    """

    @abstractmethod
    def create_sidebar(self) -> Any:
        """Create sidebar container"""
        pass

    @abstractmethod
    def create_main_content_area(self) -> Any:
        """Create main content area"""
        pass

    @abstractmethod
    def create_columns(self, column_specs: List[Union[int, float]]) -> List[Any]:
        """Create columns with specified widths"""
        pass

    @abstractmethod
    def create_tabs(self, tab_names: List[str]) -> List[Any]:
        """Create tabbed interface"""
        pass

    @abstractmethod
    def create_expander(self, title: str, expanded: bool = False) -> Any:
        """Create expandable section"""
        pass


class IValidationDisplayInterface(ABC):
    """
    Interface for validation display operations

    Defines contract for displaying validation results
    and errors in a user-friendly manner.
    """

    @abstractmethod
    def display_validation_errors(
        self, errors: List[str], field_errors: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """Display validation errors"""
        pass

    @abstractmethod
    def display_validation_warnings(self, warnings: List[str]) -> None:
        """Display validation warnings"""
        pass

    @abstractmethod
    def show_field_error(self, field_name: str, error_message: str) -> None:
        """Show error for specific field"""
        pass

    @abstractmethod
    def clear_validation_messages(self) -> None:
        """Clear all validation messages"""
        pass


class IApplicationInterface(ABC):
    """
    Main application interface that combines all sub-interfaces

    This interface provides a unified contract for all UI operations
    in the application, serving as the primary abstraction layer.
    """

    # Chat operations
    chat: IChatDisplayInterface

    # File operations
    files: IFileManagementInterface

    # Session operations
    session: ISessionManagementInterface

    # Notifications
    notifications: INotificationInterface

    # Layout operations
    layout: ILayoutInterface

    # Validation display
    validation: IValidationDisplayInterface

    @abstractmethod
    def initialize_application_ui(self, config: Dict[str, Any]) -> None:
        """Initialize the application UI"""
        pass

    @abstractmethod
    def set_page_config(
        self, title: str, icon: str = "🤖", layout: str = "wide"
    ) -> None:
        """Set page configuration"""
        pass

    @abstractmethod
    def render_header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Render application header"""
        pass

    @abstractmethod
    def render_footer(self, content: Optional[str] = None) -> None:
        """Render application footer"""
        pass

    @abstractmethod
    def handle_errors(self, error: Exception, context: str = "") -> None:
        """Handle and display application errors"""
        pass


class IFormInterface(ABC):
    """
    Interface for form operations

    Defines contract for creating and managing forms
    with validation and user interaction.
    """

    @abstractmethod
    def create_text_input(
        self,
        label: str,
        key: str,
        default_value: str = "",
        placeholder: str = "",
        help_text: str = "",
        validation_func: Optional[Callable] = None,
    ) -> UserInputResult:
        """Create text input field"""
        pass

    @abstractmethod
    def create_number_input(
        self,
        label: str,
        key: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        default_value: Optional[float] = None,
        help_text: str = "",
    ) -> UserInputResult:
        """Create number input field"""
        pass

    @abstractmethod
    def create_select_box(
        self,
        label: str,
        key: str,
        options: List[Any],
        default_index: int = 0,
        help_text: str = "",
    ) -> UserInputResult:
        """Create select box"""
        pass

    @abstractmethod
    def create_checkbox(
        self, label: str, key: str, default_value: bool = False, help_text: str = ""
    ) -> UserInputResult:
        """Create checkbox"""
        pass

    @abstractmethod
    def create_button(
        self,
        label: str,
        key: str,
        button_type: str = "primary",
        disabled: bool = False,
        help_text: str = "",
    ) -> UserInputResult:
        """Create button"""
        pass

    @abstractmethod
    def create_form_container(self, form_key: str) -> Any:
        """Create form container"""
        pass

    @abstractmethod
    def submit_form(self, form_container: Any) -> bool:
        """Submit form and return success status"""
        pass


class IMetricsDisplayInterface(ABC):
    """
    Interface for metrics and analytics display

    Defines contract for displaying application
    metrics, statistics, and analytics.
    """

    @abstractmethod
    def display_metric(
        self,
        label: str,
        value: Union[int, float, str],
        delta: Optional[Union[int, float, str]] = None,
        delta_color: str = "normal",
    ) -> None:
        """Display a single metric"""
        pass

    @abstractmethod
    def display_metrics_grid(self, metrics: List[Dict[str, Any]]) -> None:
        """Display grid of metrics"""
        pass

    @abstractmethod
    def display_chart(
        self, chart_type: str, data: Dict[str, Any], title: Optional[str] = None
    ) -> None:
        """Display chart visualization"""
        pass

    @abstractmethod
    def display_data_table(
        self,
        data: List[Dict[str, Any]],
        columns: Optional[List[str]] = None,
        sortable: bool = True,
        searchable: bool = True,
    ) -> None:
        """Display data table"""
        pass


class ViewInterfaceFactory:
    """
    Factory for creating view interface implementations

    This factory allows switching between different UI framework
    implementations while maintaining the same interface contracts.
    """

    @staticmethod
    def create_streamlit_interface() -> IApplicationInterface:
        """Create Streamlit implementation of application interface"""
        from .streamlit_implementation import StreamlitApplicationInterface

        return StreamlitApplicationInterface()

    @staticmethod
    def create_test_interface() -> IApplicationInterface:
        """Create test mock implementation"""
        from .mock_implementation import MockApplicationInterface

        return MockApplicationInterface()

    @staticmethod
    def get_default_interface() -> IApplicationInterface:
        """Get default interface implementation"""
        return ViewInterfaceFactory.create_streamlit_interface()


# Interface registry for dependency injection
class ViewInterfaceRegistry:
    """Registry for view interface implementations"""

    def __init__(self):
        self._interfaces: Dict[str, IApplicationInterface] = {}
        self._default_interface: Optional[IApplicationInterface] = None

    def register_interface(self, name: str, interface: IApplicationInterface) -> None:
        """Register an interface implementation"""
        self._interfaces[name] = interface

    def get_interface(self, name: str) -> Optional[IApplicationInterface]:
        """Get interface by name"""
        return self._interfaces.get(name)

    def set_default_interface(self, interface: IApplicationInterface) -> None:
        """Set default interface"""
        self._default_interface = interface

    def get_default_interface(self) -> IApplicationInterface:
        """Get default interface"""
        if self._default_interface is None:
            self._default_interface = ViewInterfaceFactory.get_default_interface()
        return self._default_interface

    def list_interfaces(self) -> List[str]:
        """List registered interface names"""
        return list(self._interfaces.keys())


# Global registry instance
view_registry = ViewInterfaceRegistry()
