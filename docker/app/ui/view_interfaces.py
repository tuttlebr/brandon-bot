"""
View Interfaces

Defines contracts for UI operations to reduce framework coupling
and enable easier testing and framework switching.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class IChatDisplayInterface(ABC):
    """
    Interface for chat display operations

    Defines contract for displaying chat messages, history,
    and managing chat-specific UI interactions.
    """


class IFileManagementInterface(ABC):
    """
    Interface for file management operations

    Defines contract for file upload, display, and management
    within the UI.
    """


class ISessionManagementInterface(ABC):
    """
    Interface for session management operations

    Defines contract for displaying session information
    and managing session state in the UI.
    """


class ILayoutInterface(ABC):
    """
    Interface for layout operations

    Defines contract for creating and managing
    UI layout structures.
    """


class IApplicationInterface(ABC):
    """
    Main application interface that combines all sub-interfaces

    This interface provides a unified contract for all UI operations
    in the application, serving as the primary abstraction layer.
    """

    @abstractmethod
    def set_page_config(
        self, title: str, icon: str = "🤖", layout: str = "wide"
    ) -> None:
        """Set page configuration"""


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
    def get_default_interface() -> IApplicationInterface:
        """Get default interface implementation"""
        return ViewInterfaceFactory.create_streamlit_interface()


# Interface registry for dependency injection
class ViewInterfaceRegistry:
    """Registry for view interface implementations"""

    def __init__(self):
        self._interfaces: Dict[str, IApplicationInterface] = {}
        self._default_interface: Optional[IApplicationInterface] = None

    def get_default_interface(self) -> IApplicationInterface:
        """Get default interface"""
        if self._default_interface is None:
            self._default_interface = (
                ViewInterfaceFactory.get_default_interface()
            )
        return self._default_interface


# Global registry instance
view_registry = ViewInterfaceRegistry()
