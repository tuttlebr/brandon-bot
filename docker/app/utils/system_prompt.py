import logging
from datetime import datetime
from typing import List, Optional

from utils.system_prompts import prompt_manager


class SystemPromptManager:
    """
    Manages system prompt generation with caching and consistency.

    This class now delegates to the centralized system_prompts module
    while maintaining backward compatibility.
    """

    def __init__(self):
        self._cached_prompt: Optional[str] = None
        self._cached_tools_list: Optional[str] = None
        self._cached_date: Optional[str] = None
        self._last_cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes cache

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
        # Delegate to centralized prompt manager
        return prompt_manager.get_context_prompt(context, **kwargs)

    def _get_context_instructions(self, context: str, **kwargs) -> str:
        """
        Get context-specific instructions.

        This method is kept for backward compatibility but is no longer used
        since context prompts are now handled by the centralized prompt manager.
        """
        # Delegate to centralized prompt manager
        return prompt_manager.get_context_prompt(context, **kwargs)

    def _should_refresh_cache(self, force_refresh: bool) -> bool:
        """Determine if cache should be refreshed"""
        if force_refresh:
            return True

        if self._cached_prompt is None:
            return True

        # Check if cache is expired
        if self._last_cache_time is None:
            return True

        time_since_cache = (
            datetime.now() - self._last_cache_time
        ).total_seconds()
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

        # Generate the full system prompt using centralized prompt manager
        self._cached_prompt = prompt_manager.get_main_prompt(tools_list)
        self._cached_tools_list = tools_list
        self._cached_date = current_date
        self._last_cache_time = datetime.now()

        logging.debug("System prompt cache refreshed")

    def _build_system_prompt(
        self,
        tools_list: Optional[str] = None,
        context_instructions: Optional[str] = None,
    ) -> str:
        """
        Build the complete system prompt - now delegates to centralized prompts

        This method is kept for backward compatibility but delegates to the
        centralized prompt manager.

        Args:
            tools_list: The available tools list (if None, will fetch it)
            context_instructions: Optional context-specific instructions

        Returns:
            Complete system prompt
        """
        # Get tools list if not provided
        if tools_list is None:
            tools_list = self._get_available_tools_list()

        # If context instructions are provided, this is a context-specific prompt
        # For now, just use the main prompt as the centralized system handles context differently
        return prompt_manager.get_main_prompt(tools_list)

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
                    logging.error("Failed to initialize tools: %s", init_e)

            # If we still have no tools, return fallback
            if not tools_text.strip():
                logging.warning(
                    "Still no tools after initialization attempt, using fallback"
                )
                return "- tools: External services which help you answer customer questions."

            return tools_text
        except Exception as e:
            logging.warning("Could not auto-generate tools list: %s", e)
            # Fallback to manual list
            return "- tools: External services which help you answer customer questions."

    def get_available_contexts(self) -> List[str]:
        """
        Get all available contexts from registered tools

        Returns:
            List of available context names
        """
        try:
            from tools.registry import get_all_supported_contexts

            return get_all_supported_contexts()
        except Exception as e:
            logging.warning("Could not get available contexts: %s", e)
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


# Dynamic system prompt with current date and tool list
def get_system_prompt() -> str:
    """
    Generate the system prompt dynamically with current tools list

    Returns:
        The complete system prompt with available tools
    """
    logging.debug(system_prompt_manager.get_system_prompt())
    return system_prompt_manager.get_system_prompt()


def get_context_system_prompt(context: str, **kwargs) -> str:
    """
    Get a context-specific system prompt while maintaining the core persona

    Args:
        context: The context type (e.g., 'translation', 'text_processing')
        **kwargs: Context-specific parameters

    Returns:
        Context-specific system prompt
    """
    return system_prompt_manager.get_context_system_prompt(context, **kwargs)
