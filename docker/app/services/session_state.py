"""Session state helpers with Streamlit integration

This file provides session state management that works with Streamlit's
session state to persist data across requests. Falls back to contextvars
when Streamlit is not available (e.g., in tests).
"""

from __future__ import annotations

import contextvars
from typing import Optional

# Context variables as fallback when Streamlit isn't available
_active_pdf_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_pdf_id", default=None
)
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)

# ---------------------------------------------------------------------
# PDF id helpers
# ---------------------------------------------------------------------


def set_active_pdf_id(pdf_id: str | None):
    """Set the active PDF ID in session state"""
    try:
        import streamlit as st

        if hasattr(st, "session_state"):
            if pdf_id:
                st.session_state["active_pdf_id"] = pdf_id
            elif "active_pdf_id" in st.session_state:
                del st.session_state["active_pdf_id"]
    except ImportError:
        # Fallback to contextvars if Streamlit not available
        pass

    # Always set in contextvars too for immediate access
    _active_pdf_id.set(pdf_id)


def get_active_pdf_id() -> Optional[str]:
    """Get the active PDF ID from session state"""
    try:
        import streamlit as st

        if (
            hasattr(st, "session_state")
            and "active_pdf_id" in st.session_state
        ):
            pdf_id = st.session_state["active_pdf_id"]
            import logging

            logging.debug(
                "Retrieved active PDF ID from session state: %s", pdf_id
            )
            return pdf_id
    except ImportError:
        pass

    # Fallback to contextvars
    pdf_id = _active_pdf_id.get()
    if pdf_id:
        import logging

        logging.debug("Retrieved active PDF ID from contextvars: %s", pdf_id)
    return pdf_id


# ---------------------------------------------------------------------
# Session id helpers
# ---------------------------------------------------------------------


def set_session_id(session_id: str | None):
    """Set the session ID"""
    try:
        import streamlit as st

        if hasattr(st, "session_state"):
            if session_id:
                st.session_state["session_id"] = session_id
            elif "session_id" in st.session_state:
                del st.session_state["session_id"]
    except ImportError:
        pass

    # Always set in contextvars too
    _session_id.set(session_id)


def get_session_id() -> Optional[str]:
    """Get the session ID"""
    try:
        import streamlit as st

        if hasattr(st, "session_state") and "session_id" in st.session_state:
            return st.session_state["session_id"]
    except ImportError:
        pass

    # Fallback to contextvars
    return _session_id.get()
