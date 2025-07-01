"""
Streamlit Context Utilities

This module provides utilities for preserving Streamlit's script run context
when executing code in threads or async contexts.
"""

import functools
import logging
from typing import Any, Callable, Optional

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

logger = logging.getLogger(__name__)


def get_streamlit_context():
    """
    Get the current Streamlit script run context

    Returns:
        The current script run context or None if not in a Streamlit app
    """
    try:
        return get_script_run_ctx()
    except Exception as e:
        logger.debug(f"Could not get Streamlit context: {e}")
        return None


def with_streamlit_context(func: Callable) -> Callable:
    """
    Decorator that preserves Streamlit context when running in threads

    Args:
        func: Function to wrap with Streamlit context

    Returns:
        Wrapped function that preserves context
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Try to add script run context if available
        try:
            ctx = get_script_run_ctx()
            if ctx is not None:
                # Add context to the current thread
                import threading

                add_script_run_ctx(threading.current_thread(), ctx)
        except Exception:
            # Context not available, continue without it
            pass

        return func(*args, **kwargs)

    return wrapper


def run_with_streamlit_context(func: Callable, *args, **kwargs) -> Any:
    """
    Run a function with Streamlit context preserved

    This is useful when executing functions in thread pools or async contexts

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function execution
    """
    try:
        # Get the current context before switching threads
        ctx = get_script_run_ctx()

        if ctx is not None:
            # Wrap the function to add context
            def wrapped_func():
                # Import here to avoid circular imports
                import threading

                add_script_run_ctx(threading.current_thread(), ctx)
                return func(*args, **kwargs)

            return wrapped_func()
        else:
            # No context available, run normally
            return func(*args, **kwargs)

    except Exception as e:
        logger.debug(f"Could not preserve Streamlit context: {e}")
        # Fallback to running without context
        return func(*args, **kwargs)


class StreamlitContextManager:
    """
    Context manager for preserving Streamlit context in threads

    Usage:
        with StreamlitContextManager():
            # Your threaded code here
            pass
    """

    def __init__(self):
        self.ctx = None

    def __enter__(self):
        try:
            self.ctx = get_script_run_ctx()
            if self.ctx is not None:
                import threading

                add_script_run_ctx(threading.current_thread(), self.ctx)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context cleanup is automatic when thread ends
        pass


def suppress_streamlit_warnings():
    """
    Suppress Streamlit ScriptRunContext warnings in logs

    This should be used sparingly and only when you're sure the warnings
    are benign (e.g., in background tasks that don't interact with UI)
    """
    import sys

    # Create a custom filter to suppress specific warnings
    class StreamlitContextFilter(logging.Filter):
        def filter(self, record):
            # Filter out the specific ScriptRunContext warning
            if "missing ScriptRunContext" in record.getMessage():
                return False
            return True

    # Add filter to Streamlit logger
    streamlit_logger = logging.getLogger("streamlit")
    streamlit_logger.addFilter(StreamlitContextFilter())
