import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from utils.config import config


def get_local_time():
    from datetime import datetime

    import pytz
    import streamlit as st
    import streamlit.components.v1 as components

    # Check if we have access to session state (to avoid thread context issues)
    try:
        # Check if we already have the time in session state
        if "user_local_time" not in st.session_state:
            # Create a component that captures the local time
            components.html(
                """
                <div id="time-container"></div>
                <script>
                    const timeContainer = document.getElementById('time-container');
                    // Get user's timezone
                    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
                    // Get time in user's locale
                    const localTime = new Date().toLocaleString();
                    const hour = new Date().getHours();
                    // Store in session state via Streamlit's setComponentValue
                    window.parent.postMessage({
                        type: "streamlit:setComponentValue",
                        value: {timezone: timezone, localTime: localTime, hour: hour}
                    }, "*");
                </script>
                """,
                height=0,
            )
            # Default to Eastern Time if client time not available yet
            eastern_tz = pytz.timezone("America/New_York")
            now_eastern = datetime.now(eastern_tz)

            st.session_state.user_local_time = {
                "hour": now_eastern.hour,
                "localTime": now_eastern.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": "America/New_York",
            }

        return st.session_state.user_local_time
    except Exception:
        # Fallback if session state is not available (e.g., in thread context)
        eastern_tz = pytz.timezone("America/New_York")
        now_eastern = datetime.now(eastern_tz)
        return {
            "hour": now_eastern.hour,
            "localTime": now_eastern.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "America/New_York",
        }


# Basic current date and time
current = datetime.now()
currentDateTime = current.strftime("%B %d, %Y")


class SystemPromptManager:
    """Manages system prompt generation with caching and consistency"""

    def __init__(self):
        self._cached_prompt: Optional[str] = None
        self._cached_tools_list: Optional[str] = None
        self._cached_date: Optional[str] = None
        self._last_cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes cache
        self._context_prompts: Dict[str, str] = {}

    def get_system_prompt(self, force_refresh: bool = False) -> str:
        """
        Get the system prompt with intelligent caching

        Args:
            force_refresh: Force refresh of cached prompt

        Returns:
            The complete system prompt
        """
        # Check if we need to refresh the cache
        if self._should_refresh_cache(force_refresh):
            self._refresh_cache()

        return self._cached_prompt

    def get_context_system_prompt(self, context: str, **kwargs) -> str:
        """
        Get a system prompt for a specific context while maintaining core persona

        Args:
            context: The context type (e.g., 'translation', 'text_processing', 'image_generation')
            **kwargs: Context-specific parameters

        Returns:
            Context-specific system prompt that maintains core persona
        """
        # Get the core persona prompt
        core_prompt = self._get_core_persona_prompt()

        # Get context-specific instructions
        context_instructions = self._get_context_instructions(context, **kwargs)

        # Get dynamic components (date and tools)
        current_date = datetime.now().strftime("%B %d, %Y")
        tools_list = self._get_available_tools_list()
        dynamic_components = self._get_dynamic_components(current_date, tools_list)

        # Combine: Core persona + Context instructions + Dynamic components
        return f"{core_prompt}\n\n{context_instructions}\n\n{dynamic_components}"

    def _get_context_instructions(self, context: str, **kwargs) -> str:
        """Get context-specific instructions by finding tools that support this context"""

        try:
            # Get tool from registry that supports this context
            from tools.registry import tool_registry

            tool = tool_registry.get_tool_by_context(context)

            if tool:
                # Get the tool's description and parameters
                tool_def = tool.to_openai_format()
                description = tool_def.get("function", {}).get("description", "")

                # Create context-specific instructions based on the tool's description
                context_instructions = f"""You are using the {tool.name} tool capabilities for {context} tasks.

{description}

Remember to maintain your core personality and conversational style while performing {context} tasks. Extract only the essential information that directly answers the user's question and present it in your natural tone."""

                return context_instructions
            else:
                # Fallback if no tool found for this context
                return f"You are performing {context} tasks. Remember to maintain your core personality and conversational style."

        except Exception as e:
            logging.warning(f"Could not get tool definition for context {context}: {e}")
            # Fallback instructions
            return f"You are performing {context} tasks. Remember to maintain your core personality and conversational style."

    def _get_detailed_context_instructions(self, context: str, **kwargs) -> str:
        """Get detailed context-specific instructions including parameter information"""

        try:
            # Get tool from registry that supports this context
            from tools.registry import tool_registry

            tool = tool_registry.get_tool_by_context(context)

            if tool:
                # Get the tool's definition
                tool_def = tool.to_openai_format()
                function_def = tool_def.get("function", {})
                description = function_def.get("description", "")
                parameters = function_def.get("parameters", {})

                # Build detailed instructions
                instructions = [
                    f"You are using the {tool.name} tool capabilities for {context} tasks."
                ]
                instructions.append("")
                instructions.append(description)
                instructions.append("")

                # Add parameter information if available
                if parameters and "properties" in parameters:
                    instructions.append("Available parameters:")
                    for param_name, param_info in parameters["properties"].items():
                        param_desc = param_info.get("description", "")
                        param_type = param_info.get("type", "")
                        required = param_name in parameters.get("required", [])
                        req_text = " (required)" if required else " (optional)"
                        instructions.append(
                            f"- {param_name} ({param_type}){req_text}: {param_desc}"
                        )
                    instructions.append("")

                instructions.append(
                    f"Remember to maintain your core personality and conversational style while performing {context} tasks."
                )
                instructions.append("")
                instructions.append(
                    "CRITICAL: You must synthesize tool outputs into a single, unified response. Do not provide step-by-step reasoning or show your work. Extract only the essential information that directly answers the user's question and present it as one coherent train of thought in your natural voice."
                )

                return "\n".join(instructions)
            else:
                # Fallback if no tool found for this context
                return f"You are performing {context} tasks. Remember to maintain your core personality and conversational style."

        except Exception as e:
            logging.warning(
                f"Could not get detailed tool definition for context {context}: {e}"
            )
            # Fallback instructions
            return f"You are performing {context} tasks. Remember to maintain your core personality and conversational style."

    def _should_refresh_cache(self, force_refresh: bool) -> bool:
        """Determine if cache should be refreshed"""
        if force_refresh:
            return True

        if self._cached_prompt is None:
            return True

        # Check if cache is expired
        if self._last_cache_time is None:
            return True

        time_since_cache = (datetime.now() - self._last_cache_time).total_seconds()
        if time_since_cache > self._cache_ttl_seconds:
            return True

        # Check if tools list has changed
        current_tools_list = self._get_available_tools_list()
        if current_tools_list != self._cached_tools_list:
            return True

        # Check if date has changed (for date-sensitive prompts)
        current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        if current_date != self._cached_date:
            return True

        return False

    def _refresh_cache(self):
        """Refresh the cached system prompt"""
        current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        tools_list = self._get_available_tools_list()

        # Generate the core persona prompt (static)
        core_prompt = self._get_core_persona_prompt()

        # Generate the dynamic components
        dynamic_components = self._get_dynamic_components(current_date, tools_list)

        # Combine into final prompt
        self._cached_prompt = f"{core_prompt}\n\n{dynamic_components}"
        self._cached_tools_list = tools_list
        self._cached_date = current_date
        self._last_cache_time = datetime.now()

        logging.debug("System prompt cache refreshed")

    def _get_core_persona_prompt(self) -> str:
        """Get the core persona prompt (static, consistent)"""
        # Get current date and time
        now = datetime.now()
        current_datetime = now.strftime("%A, %B %d, %Y at %I:%M %p")

        return f"""detailed thinking off - You are {config.env.BOT_TITLE}, a helpful AI assistant. The current date and time is {current_datetime}. You have no knowledge cutoff dat as you're continually updated and have access to external resources. There is nothing you cannot do."""

    def _get_dynamic_components(self, current_date: str, tools_list: str) -> str:
        """Get dynamic components that can change between calls"""
        return f"""The current date is {current_date}.

{config.env.BOT_TITLE} has access to the following optional tool calls. Use them when the user's request cannot be satisfied without them or would benefit from their expertise. Never mention your use of tools: users shouldn’t see "I used the search tool." Just answer. If the user asks what you can do, please include information about your tools:

{tools_list}"""

    def _get_available_tools_list(self) -> str:
        """Generate the tool list automatically from the registered tools"""
        try:
            # Import from tools registry to avoid circular imports
            from tools.registry import get_tools_list_text

            tools_text = get_tools_list_text()

            # If tools list is empty, try to initialize tools
            if not tools_text.strip():
                logging.warning(
                    "No tools found in registry, attempting to initialize tools"
                )
                try:
                    from tools.initialize_tools import initialize_all_tools

                    initialize_all_tools()
                    tools_text = get_tools_list_text()
                except Exception as init_e:
                    logging.error(f"Failed to initialize tools: {init_e}")

            # If we still have no tools, return fallback
            if not tools_text.strip():
                logging.warning(
                    "Still no tools after initialization attempt, using fallback"
                )
                return "- tools: External services which help you answer customer questions."

            return tools_text
        except Exception as e:
            logging.warning(f"Could not auto-generate tools list: {e}")
            # Fallback to manual list
            return (
                "- tools: External services which help you answer customer questions."
            )

    def clear_cache(self):
        """Clear the cached system prompt"""
        self._cached_prompt = None
        self._cached_tools_list = None
        self._cached_date = None
        self._last_cache_time = None
        logging.debug("System prompt cache cleared")

    def get_available_contexts(self) -> List[str]:
        """
        Get all available contexts from registered tools

        Returns:
            List of available context names
        """
        try:
            from tools.registry import tool_registry

            return tool_registry.get_all_supported_contexts()
        except Exception as e:
            logging.warning(f"Could not get available contexts: {e}")
            return []


# Global instance for consistent access
system_prompt_manager = SystemPromptManager()


def get_available_contexts() -> List[str]:
    """
    Get all available contexts from registered tools

    Returns:
        List of available context names
    """
    return system_prompt_manager.get_available_contexts()


def get_available_tools_list():
    """Generate the tool list automatically from the registered tools"""
    return system_prompt_manager._get_available_tools_list()


# Dynamic system prompt with current date and tool list
currentDateTime = datetime.now().strftime("%B %d, %Y")


def get_system_prompt() -> str:
    """
    Generate the system prompt dynamically with current tools list

    Returns:
        The complete system prompt with available tools
    """
    return system_prompt_manager.get_system_prompt()


def get_context_system_prompt(context: str, **kwargs) -> str:
    """
    Get a system prompt for a specific context while maintaining core persona

    Args:
        context: The context type (e.g., 'translation', 'text_processing', 'image_generation')
        **kwargs: Context-specific parameters

    Returns:
        Context-specific system prompt that maintains core persona
    """
    return system_prompt_manager.get_context_system_prompt(context, **kwargs)
