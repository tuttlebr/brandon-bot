"""
View Helpers

Provides abstraction layer for UI framework-specific operations,
reducing coupling between controllers and Streamlit framework.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)


class UIMessage:
    """Represents a UI message with metadata"""

    def __init__(
        self,
        content: str,
        message_type: str = "info",
        title: Optional[str] = None,
    ):
        self.content = content
        self.message_type = message_type  # info, success, warning, error
        self.title = title
        self.timestamp = datetime.now()


class IViewInterface(ABC):
    """
    Abstract interface for view operations

    This interface abstracts UI framework operations to reduce
    coupling with specific frameworks like Streamlit.
    """

    @abstractmethod
    def show_message(self, message: UIMessage) -> None:  # dead: disable
        """Display a message to the user"""

    @abstractmethod
    def show_loading(
        self, message: str = "Loading..."
    ) -> Any:  # dead: disable
        """Show loading indicator"""


class StreamlitViewInterface(IViewInterface):
    """
    Streamlit implementation of view interface

    This class provides Streamlit-specific implementations while
    maintaining the abstract interface for controllers.
    """

    def __init__(self):
        self.current_loading_context = None

    def show_message(self, message: UIMessage) -> None:  # dead: disable
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

    def show_loading(
        self, message: str = "Loading..."
    ) -> Any:  # dead: disable
        """Show loading indicator using Streamlit spinner"""
        return st.spinner(message)


class MessageHelper:
    """Helper for displaying different types of messages"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface


class ProgressHelper:
    """Helper for progress indicators"""

    def __init__(self, view_interface: IViewInterface):
        self.view = view_interface


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

    def create_progress_helper(self) -> ProgressHelper:
        """Create progress helper"""
        return ProgressHelper(self.view_interface)


# Global factory instance for easy access
view_factory = ViewHelperFactory()
