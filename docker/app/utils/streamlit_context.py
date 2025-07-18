"""
Streamlit Context Utilities

This module provides utilities for preserving Streamlit's script run context
when executing code in threads or async contexts.
"""

import logging
from typing import Any, Callable

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

logger = logging.getLogger(__name__)


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

                # Add context to current thread
                try:
                    add_script_run_ctx(threading.current_thread(), ctx)
                except Exception:
                    # If adding context fails, continue without it
                    pass

                return func(*args, **kwargs)

            return wrapped_func()
        else:
            # No context available, run normally
            return func(*args, **kwargs)

    except Exception as e:
        logger.debug(f"Could not preserve Streamlit context: {e}")
        # Fallback to running without context
        return func(*args, **kwargs)


def suppress_streamlit_warnings():
    """
    Suppress Streamlit ScriptRunContext warnings in logs

    This should be used sparingly and only when you're sure the warnings
    are benign (e.g., in background tasks that don't interact with UI)
    """

    # Also try to suppress at the thread level by capturing the warning source
    try:
        import streamlit.runtime.scriptrunner as scriptrunner

        # Monkey patch to suppress the warning at source
        original_get_script_run_ctx = scriptrunner.get_script_run_ctx

        def patched_get_script_run_ctx(*args, **kwargs):
            try:
                return original_get_script_run_ctx(*args, **kwargs)
            except Exception:
                # Silently return None instead of logging warning
                return None

        scriptrunner.get_script_run_ctx = patched_get_script_run_ctx
    except Exception:
        # If patching fails, fall back to just the filters
        pass
