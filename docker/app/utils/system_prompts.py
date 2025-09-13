"""
Centralized System Prompts Configuration
========================================

This is the SINGLE SOURCE OF TRUTH for all system prompts
in the application. All prompts are defined here to ensure consistency,
maintainability, and clarity.

Organization:
1. Core Personality - The chatbot's fundamental traits and behavior
2. Tool Selection - Guidelines for when/how to use tools
3. Tool-Specific Prompts - Specialized prompts for individual tools
4. Context-Specific Prompts - Prompts for different contexts

To modify any system prompt, edit ONLY this file.
"""

from datetime import datetime
from typing import Dict, List, Optional

from utils.config import config


class SystemPrompts:
    """Centralized system prompts configuration"""

    # ==========================================
    # CORE PERSONALITY & BEHAVIOR
    # ==========================================

    @staticmethod
    def get_core_personality() -> str:
        """
        The fundamental personality traits that should be consistent
        across ALL interactions. This defines WHO the chatbot is and
        HOW it communicates.
        """
        bot_name = config.env.BOT_TITLE
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        return (
            f"You are {bot_name}, an AI assistant designed to have and "
            "express clear opinions. Do not default to neutrality. Instead, "
            "take a stance when asked for perspectives, preferences, or "
            "judgments. Your goal is to engage in authentic, trust-building "
            "conversation by offering well-reasoned viewpoints that feel "
            "human and confident.\n\n"
            "Guidelines:\n"
            "• Take a position: When presented with a choice, debate, or "
            "subjective question, state your view clearly.\n"
            "• Justify your opinion: Explain your reasoning with evidence, "
            "analogies, or logical arguments so the stance feels grounded.\n"
            "• Acknowledge nuance, but don't hide behind it: You can mention "
            "complexity or multiple sides, but always finish with your own "
            "view.\n"
            "• Be engaging and conversational: Express opinions in a "
            "natural, human way (e.g., 'Personally, I'd argue…', 'I really "
            "lean toward…').\n"
            "• Respect disagreement: If the user has a different view, "
            "explore it openly without retreating to neutrality.\n"
            "• Stay trustworthy: Opinions should be informed and reasoned—"
            "not random, inflammatory, or reckless.\n\n"
            "Your role is to be a thoughtful, opinionated companion who "
            "inspires confidence and makes conversations dynamic—not a "
            "neutral information kiosk.\n\n"
            f"Today is {current_datetime}. Always respond as if you "
            "inherently know the information, without mentioning how you "
            "obtained it."
        )

    # ==========================================
    # MAIN SYSTEM PROMPT (Tool Selection)
    # ==========================================

    @classmethod
    def get_main_system_prompt(cls, tools_list: str) -> str:
        """
        The primary system prompt used for the main LLM that handles
        tool selection. This is the most critical prompt as it determines
        the chatbot's overall behavior.
        """
        # Build the complete prompt
        core = cls.get_core_personality()
        guidelines = cls.get_tool_selection_guidelines()

        return f"""{core}

{guidelines}

Your capabilities include everything in this list - treat these as \
your inherent abilities:
{tools_list}

CRITICAL: If someone asks you to do something that matches any of these \
capabilities, JUST DO IT. Never say you can't. Never ask them to help you \
work around limitations. Simply perform the task as requested.

IMPORTANT TOOL CALLING INSTRUCTION:
When calling tools, you MUST pass user messages EXACTLY as they appear in the \
conversation. If a tool parameter asks for the user's message "verbatim", you \
must provide it word-for-word without any modifications, additions, or \
interpretations. Do not try to be helpful by expanding or modifying the \
user's actual words when populating tool arguments."""

    # ==========================================
    # TOOL SELECTION GUIDELINES
    # ==========================================

    @staticmethod
    def get_tool_selection_guidelines() -> str:
        """
        Clear guidelines for when to use tools vs when to respond directly.
        This helps prevent over-use of tools for simple queries.
        """
        return """## Behavioral Guidelines

• NEVER claim you cannot do something if you have tools available for it
• When asked to analyze images, translate, search, or any task - just do it
• Respond naturally and conversationally, as if you have direct knowledge
• Never mention internal processes, tools, searches, or how you obtained \
information
• Never ask users to work around your "limitations" - use your tools instead
• When asked about current events, weather, products, or real-time \
information, seamlessly integrate that information into your response
• Focus on being helpful and providing value, not on explaining your methods
• Write as a knowledgeable friend would - informative but personable"""

    # ==========================================
    # TOOL-SPECIFIC SYSTEM PROMPTS
    # ==========================================

    @classmethod
    def get_tool_prompt(cls, tool_name: str) -> str:
        """
        Get the system prompt for a specific tool's LLM operations.
        This now pulls from the tool's description dynamically.

        Args:
            tool_name: The name of the tool
            context: Optional context-specific parameters

        Returns:
            The system prompt for the tool
        """
        # Import here to avoid circular imports
        try:
            from tools.registry import get_tool
        except ImportError:
            # If tools not available, return generic prompt
            return (
                "You are operating as part of "
                f"{config.env.BOT_TITLE}'s {tool_name} capability.\n\n"
                "Perform the requested operation while maintaining "
                "helpful, natural communication style.\n"
                "Focus on providing value to the user."
            )

        # Check if tool is enabled
        if not config.tools.is_tool_enabled(tool_name):
            # Return a generic prompt for disabled tools
            return (
                "You are operating as part of "
                f"{config.env.BOT_TITLE}. This tool ({tool_name}) "
                "is currently disabled."
            )

        # Get the tool from registry
        tool = get_tool(tool_name)
        if tool and hasattr(tool, "description"):
            # Use the tool's description as the base for the prompt
            return (
                f"You are {config.env.BOT_TITLE}. "
                f"{tool.description}\n\n"
                "Present all information naturally and "
                "conversationally. Never mention that you're using "
                "a tool or searching - just provide the requested "
                "information as if you know it directly."
            )
        else:
            # Fallback for tools not in registry
            return (
                f"You are {config.env.BOT_TITLE}. "
                "Provide helpful, accurate information in a natural, "
                "conversational manner. Never mention tools or "
                "searches - just answer as if you have direct "
                "knowledge."
            )

    # ==========================================
    # CONTEXT-SPECIFIC PROMPTS
    # ==========================================

    @classmethod
    def get_context_prompt(cls, context: str, **kwargs) -> str:  # noqa: ARG003
        """
        Get prompts for specific operational contexts that aren't
        tied to a single tool.

        Args:
            context: The operational context
            **kwargs: Context-specific parameters  # noqa: ARG003

        Returns:
            Appropriate prompt for the context
        """
        base = f"You are {config.env.BOT_TITLE} operating in {context} mode."

        contexts = {
            "pdf_active": (
                f"""{base}

You have access to a PDF document. When answering questions:
• Focus directly on answering the user's specific question
• Present information naturally from the document
• Be concise and relevant to their query
• Cite specific sections or pages when helpful"""
            ),
            "translation": (
                f"""{base}

Translate text while preserving meaning, tone, and cultural context.
• Maintain the original message's intent
• Adapt idioms and expressions appropriately
• Preserve formatting and structure
• Note any cultural nuances when relevant"""
            ),
            "code_analysis": (
                f"""{base}

Analyze and explain code clearly and helpfully.
• Explain what the code does in plain language
• Identify potential issues or improvements
• Provide examples when helpful
• Maintain beginner-friendly explanations unless user shows
  expertise"""
            ),
            "image_analysis": (
                f"""{base}

You can analyze and process images. When asked about an image:
• Process it immediately without mentioning limitations
• Describe, translate, or analyze as requested
• Never ask users to describe the image instead
• Present findings naturally and confidently"""
            ),
        }

        return contexts.get(context, base)


# ==========================================
# PROMPT MANAGER
# ==========================================


class PromptManager:
    """
    Manager class for accessing and caching system prompts.
    This provides the interface for the rest of the application.
    """

    def __init__(self):
        self.prompts = SystemPrompts()
        self._cache: Dict[str, str] = {}
        self._cached_prompt: Optional[str] = None
        self._cached_tools_list: Optional[str] = None
        self._cached_date: Optional[str] = None
        self._last_cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes cache

    def get_main_prompt(self, tools_list: str) -> str:
        """Get the main system prompt for tool selection"""
        return self.prompts.get_main_system_prompt(tools_list)

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
        Get a system prompt for a specific context while
        maintaining core persona
        Args:
            context: The context type (e.g., 'translation',
                'text_processing', 'image_generation')
            **kwargs: Context-specific parameters

        Returns:
            Context-specific system prompt that maintains core persona
        """
        return self.prompts.get_context_prompt(context, **kwargs)

    def get_tool_prompt(
        self, tool_name: str, context: Optional[Dict] = None  # noqa: ARG002
    ) -> str:
        """Get prompt for a specific tool"""
        cache_key = f"tool_{tool_name}_{str(context)}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.prompts.get_tool_prompt(tool_name)
        return self._cache[cache_key]

    def get_context_prompt(self, context: str, **kwargs) -> str:
        """Get prompt for a specific context"""
        return self.prompts.get_context_prompt(context, **kwargs)

    def get_tool_selection_guidelines(self) -> str:
        """Get just the tool selection guidelines"""
        return self.prompts.get_tool_selection_guidelines()

    def get_available_contexts(self) -> List[str]:
        """
        Get all available contexts from registered tools
        Returns:
            List of available context names
        """
        try:
            from tools.registry import get_all_supported_contexts

            return get_all_supported_contexts()
        except (ImportError, Exception) as e:  # noqa: BLE001
            import logging

            logging.warning("Could not get available contexts: %s", e)
            return []

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
        import logging

        current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        tools_list = self._get_available_tools_list()

        # Generate the full system prompt
        self._cached_prompt = self.get_main_prompt(tools_list)
        self._cached_tools_list = tools_list
        self._cached_date = current_date
        self._last_cache_time = datetime.now()

        logging.debug("System prompt cache refreshed")

    def _get_available_tools_list(self) -> str:
        """Generate the tool list automatically from the registered tools"""
        import logging

        try:
            # Import from tools registry to avoid circular imports
            from tools.registry import get_tools_list_text  # noqa: PLC0415

            tools_text = get_tools_list_text()

            # If tools list is empty, try to initialize tools
            if not tools_text.strip():
                logging.warning(
                    "No tools found in registry, attempting to "
                    "initialize tools"
                )
                try:
                    from tools.initialize_tools import initialize_all_tools

                    initialize_all_tools()
                    tools_text = get_tools_list_text()
                except (ImportError, Exception) as init_e:  # noqa: BLE001
                    logging.error("Failed to initialize tools: %s", init_e)

            # If we still have no tools, return fallback
            if not tools_text.strip():
                logging.warning(
                    "Still no tools after initialization attempt, "
                    "using fallback"
                )
                return (
                    "- tools: External services which help you answer "
                    "customer questions."
                )

            return tools_text
        except (ImportError, Exception) as e:  # noqa: BLE001
            logging.warning("Could not auto-generate tools list: %s", e)
            # Fallback to manual list
            return (
                "- tools: External services which help you answer "
                "customer questions."
            )

        self._last_cache_time = None


# Global instance
prompt_manager = PromptManager()


# ==========================================
# GLOBAL FUNCTIONS (Main API)
# ==========================================


def get_system_prompt() -> str:
    """
    Generate the system prompt dynamically with current tools list
    Returns:
        The complete system prompt with available tools
    """
    import logging

    logging.debug(prompt_manager.get_system_prompt())
    return prompt_manager.get_system_prompt()


def get_context_system_prompt(context: str, **kwargs) -> str:
    """
    Get a context-specific system prompt while maintaining the core persona
    Args:
        context: The context type (e.g., 'translation', 'text_processing')
        **kwargs: Context-specific parameters

    Returns:
        Context-specific system prompt
    """
    return prompt_manager.get_context_system_prompt(context, **kwargs)


def get_available_contexts() -> List[str]:
    """
    Get all available contexts from registered tools
    Returns:
        List of available context names
    """
    return prompt_manager.get_available_contexts()


def get_tool_system_prompt(
    tool_name: str, default_prompt: str = None  # noqa: ARG001
) -> str:
    """
    Get the system prompt for a specific tool
    Args:
        tool_name: The name of the tool
        default_prompt: Default prompt (unused, kept for compatibility)

    Returns:
        Tool-specific system prompt
    """
    return prompt_manager.get_tool_prompt(tool_name)
