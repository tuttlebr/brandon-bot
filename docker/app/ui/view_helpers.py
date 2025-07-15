"""
View Helpers

Provides abstraction layer for UI framework-specific operations,
reducing coupling between controllers and Streamlit framework.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import streamlit as st

logger = logging.getLogger(__name__)


class UIMessage:
    """Represents a UI message with metadata"""

    def __init__(
        self, content: str, message_type: str = "info", title: Optional[str] = None
    ):
        self.content = content
        self.message_type = message_type  # info, success, warning, error
        self.title = title
        self.timestamp = datetime.now()


class UIComponent:
    """Base class for UI components"""

    def __init__(self, component_id: str):
        self.component_id = component_id
        self.is_visible = True
        self.css_classes = []

    def set_visibility(self, visible: bool) -> None:
        """Set component visibility"""
        self.is_visible = visible

    def add_css_class(self, css_class: str) -> None:
        """Add CSS class to component"""
        if css_class not in self.css_classes:
            self.css_classes.append(css_class)


class FileUploadResult:
    """Result of file upload operation"""

    def __init__(
        self,
        success: bool,
        file_data: Optional[Any] = None,
        filename: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self.success = success
        self.file_data = file_data
        self.filename = filename
        self.error_message = error_message
        self.file_size = getattr(file_data, 'size', 0) if file_data else 0


class UserInputResult:
    """Result of user input operation"""

    def __init__(self, value: Any, input_type: str, submitted: bool = False):
        self.value = value
        self.input_type = input_type
        self.submitted = submitted
        self.timestamp = datetime.now()


class IViewInterface(ABC):
    """
    Abstract interface for view operations

    This interface abstracts UI framework operations to reduce
    coupling with specific frameworks like Streamlit.
    """

    @abstractmethod
    def show_message(self, message: UIMessage) -> None:
        """Display a message to the user"""
        pass

    @abstractmethod
    def show_loading(self, message: str = "Loading...") -> Any:
        """Show loading indicator"""
        pass

    @abstractmethod
    def get_user_input(
        self, prompt: str, input_type: str = "text", **kwargs
    ) -> UserInputResult:
        """Get user input"""
        pass

    @abstractmethod
    def show_file_uploader(
        self, accepted_types: List[str], **kwargs
    ) -> FileUploadResult:
        """Show file uploader"""
        pass

    @abstractmethod
    def show_progress_bar(self, progress: float, message: str = "") -> None:
        """Show progress bar"""
        pass

    @abstractmethod
    def render_markdown(self, content: str) -> None:
        """Render markdown content"""
        pass

    @abstractmethod
    def render_json(self, data: Dict[str, Any]) -> None:
        """Render JSON data"""
        pass

    @abstractmethod
    def create_columns(self, num_columns: int) -> List[Any]:
        """Create layout columns"""
        pass

    @abstractmethod
    def create_sidebar(self) -> Any:
        """Create sidebar container"""
        pass


class StreamlitViewInterface(IViewInterface):
    """
    Streamlit implementation of view interface

    This class provides Streamlit-specific implementations while
    maintaining the abstract interface for controllers.
    """

    def __init__(self):
        self.current_loading_context = None

    def show_message(self, message: UIMessage) -> None:
        """Display a message using Streamlit components"""
        if not message.content:
            return

        if message.message_type == "success":
            st.success(message.content)
        elif message.message_type == "warning":
            st.warning(message.content)
        elif message.message_type == "error":
            st.error(message.content)
        else:  # info or default
            st.info(message.content)

    def show_loading(self, message: str = "Loading...") -> Any:
        """Show loading indicator using Streamlit spinner"""
        return st.spinner(message)

    def get_user_input(
        self, prompt: str, input_type: str = "text", **kwargs
    ) -> UserInputResult:
        """Get user input using appropriate Streamlit widget"""
        value = None
        submitted = False

        if input_type == "text":
            value = st.text_input(prompt, **kwargs)
        elif input_type == "textarea":
            value = st.text_area(prompt, **kwargs)
        elif input_type == "number":
            value = st.number_input(prompt, **kwargs)
        elif input_type == "select":
            options = kwargs.get("options", [])
            value = st.selectbox(
                prompt, options, **{k: v for k, v in kwargs.items() if k != "options"}
            )
        elif input_type == "multiselect":
            options = kwargs.get("options", [])
            value = st.multiselect(
                prompt, options, **{k: v for k, v in kwargs.items() if k != "options"}
            )
        elif input_type == "checkbox":
            value = st.checkbox(prompt, **kwargs)
        elif input_type == "button":
            value = st.button(prompt, **kwargs)
            submitted = value  # Button click indicates submission
        elif input_type == "chat_input":
            value = st.chat_input(prompt)
            submitted = value is not None and value.strip() != ""
        else:
            value = st.text_input(prompt, **kwargs)

        return UserInputResult(value, input_type, submitted)

    def show_file_uploader(
        self, accepted_types: List[str], **kwargs
    ) -> FileUploadResult:
        """Show file uploader using Streamlit component"""
        try:
            uploaded_file = st.file_uploader(
                "Choose file", type=accepted_types, **kwargs
            )

            if uploaded_file is not None:
                return FileUploadResult(
                    success=True, file_data=uploaded_file, filename=uploaded_file.name
                )
            else:
                return FileUploadResult(success=False)

        except Exception as e:
            logger.error(f"File upload error: {e}")
            return FileUploadResult(success=False, error_message=str(e))

    def show_progress_bar(self, progress: float, message: str = "") -> None:
        """Show progress bar using Streamlit component"""
        if message:
            st.text(message)
        st.progress(progress)

    def render_markdown(self, content: str) -> None:
        """Render markdown content using Streamlit"""
        st.markdown(content)

    def render_json(self, data: Dict[str, Any]) -> None:
        """Render JSON data using Streamlit"""
        st.json(data)

    def create_columns(self, num_columns: int) -> List[Any]:
        """Create layout columns using Streamlit"""
        return st.columns(num_columns)

    def create_sidebar(self) -> Any:
        """Create sidebar container using Streamlit"""
        return st.sidebar


class MessageHelper:
    """Helper for displaying different types of messages"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface

    def show_success(self, message: str, title: Optional[str] = None) -> None:
        """Show success message"""
        ui_message = UIMessage(message, "success", title)
        self.view.show_message(ui_message)

    def show_error(self, message: str, title: Optional[str] = None) -> None:
        """Show error message"""
        ui_message = UIMessage(message, "error", title)
        self.view.show_message(ui_message)

    def show_warning(self, message: str, title: Optional[str] = None) -> None:
        """Show warning message"""
        ui_message = UIMessage(message, "warning", title)
        self.view.show_message(ui_message)

    def show_info(self, message: str, title: Optional[str] = None) -> None:
        """Show info message"""
        ui_message = UIMessage(message, "info", title)
        self.view.show_message(ui_message)


class LayoutHelper:
    """Helper for creating UI layouts"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface

    def create_two_column_layout(self) -> tuple[Any, Any]:
        """Create a two-column layout"""
        columns = self.view.create_columns(2)
        return columns[0], columns[1]

    def create_three_column_layout(self) -> tuple[Any, Any, Any]:
        """Create a three-column layout"""
        columns = self.view.create_columns(3)
        return columns[0], columns[1], columns[2]

    def create_sidebar_layout(self) -> Any:
        """Create sidebar layout"""
        return self.view.create_sidebar()


class FormHelper:
    """Helper for creating forms and input widgets"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface

    def create_text_input(
        self,
        label: str,
        default_value: str = "",
        placeholder: str = "",
        help_text: str = "",
    ) -> UserInputResult:
        """Create text input widget"""
        kwargs = {}
        if default_value:
            kwargs["value"] = default_value
        if placeholder:
            kwargs["placeholder"] = placeholder
        if help_text:
            kwargs["help"] = help_text

        return self.view.get_user_input(label, "text", **kwargs)

    def create_select_box(
        self,
        label: str,
        options: List[str],
        default_index: int = 0,
        help_text: str = "",
    ) -> UserInputResult:
        """Create select box widget"""
        kwargs = {"options": options, "index": default_index}
        if help_text:
            kwargs["help"] = help_text

        return self.view.get_user_input(label, "select", **kwargs)

    def create_file_uploader(
        self,
        accepted_types: List[str],
        help_text: str = "",
        multiple_files: bool = False,
    ) -> FileUploadResult:
        """Create file uploader widget"""
        kwargs = {"accept_multiple_files": multiple_files}
        if help_text:
            kwargs["help"] = help_text

        return self.view.show_file_uploader(accepted_types, **kwargs)

    def create_button(
        self,
        label: str,
        button_type: str = "primary",
        help_text: str = "",
        disabled: bool = False,
    ) -> UserInputResult:
        """Create button widget"""
        kwargs = {"type": button_type, "disabled": disabled}
        if help_text:
            kwargs["help"] = help_text

        return self.view.get_user_input(label, "button", **kwargs)


class ChatHelper:
    """Helper for chat-specific UI operations"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface
        self.message_helper = MessageHelper(view_interface)

    def display_chat_message(
        self, role: str, content: str, avatar: Optional[str] = None
    ) -> None:
        """Display a chat message"""
        # This would need Streamlit-specific implementation
        # For now, we'll use the generic interface
        if role == "user":
            self.view.render_markdown(f"**👤 User:** {content}")
        elif role == "assistant":
            self.view.render_markdown(f"**🤖 Assistant:** {content}")
        else:
            self.view.render_markdown(f"**{role}:** {content}")

    def get_chat_input(
        self, placeholder: str = "Type your message..."
    ) -> UserInputResult:
        """Get chat input from user"""
        return self.view.get_user_input(placeholder, "chat_input")

    def show_typing_indicator(self, message: str = "Assistant is typing...") -> Any:
        """Show typing indicator"""
        return self.view.show_loading(message)


class FileHelper:
    """Helper for file-related UI operations"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface
        self.message_helper = MessageHelper(view_interface)

    def upload_pdf(
        self, help_text: str = "Upload a PDF file for analysis"
    ) -> FileUploadResult:
        """Create PDF uploader"""
        return self.view.show_file_uploader(["pdf"], help=help_text)

    def upload_image(self, help_text: str = "Upload an image file") -> FileUploadResult:
        """Create image uploader"""
        image_types = ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"]
        return self.view.show_file_uploader(image_types, help=help_text)

    def show_file_processing_status(self, filename: str, status: str) -> None:
        """Show file processing status"""
        if status == "processing":
            self.message_helper.show_info(f"🔄 Processing {filename}...")
        elif status == "completed":
            self.message_helper.show_success(f"✅ Successfully processed {filename}")
        elif status == "error":
            self.message_helper.show_error(f"❌ Failed to process {filename}")


class ProgressHelper:
    """Helper for progress indicators"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface

    def show_determinate_progress(self, progress: float, message: str = "") -> None:
        """Show determinate progress bar"""
        self.view.show_progress_bar(progress, message)

    def show_indeterminate_progress(self, message: str = "Processing...") -> Any:
        """Show indeterminate progress (spinner)"""
        return self.view.show_loading(message)


class ViewHelperFactory:
    """
    Factory for creating view helpers with consistent interface

    This factory provides a central point for creating view helpers
    with the appropriate view interface implementation.
    """

    def __init__(self, view_interface: Optional[IViewInterface] = None):
        self.view_interface = view_interface or StreamlitViewInterface()

    def create_message_helper(self) -> MessageHelper:
        """Create message helper"""
        return MessageHelper(self.view_interface)

    def create_layout_helper(self) -> LayoutHelper:
        """Create layout helper"""
        return LayoutHelper(self.view_interface)

    def create_form_helper(self) -> FormHelper:
        """Create form helper"""
        return FormHelper(self.view_interface)

    def create_chat_helper(self) -> ChatHelper:
        """Create chat helper"""
        return ChatHelper(self.view_interface)

    def create_file_helper(self) -> FileHelper:
        """Create file helper"""
        return FileHelper(self.view_interface)

    def create_progress_helper(self) -> ProgressHelper:
        """Create progress helper"""
        return ProgressHelper(self.view_interface)

    def get_view_interface(self) -> IViewInterface:
        """Get the view interface"""
        return self.view_interface


# Global factory instance for easy access
view_factory = ViewHelperFactory()
