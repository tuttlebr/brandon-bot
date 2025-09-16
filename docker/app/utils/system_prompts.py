"""System Prompts Configuration - Single source of truth for all prompts."""

from datetime import datetime
from typing import Dict, Optional

from utils.config import config


class SystemPrompts:
    """Structured prompt builder: Core → Tools → Personality"""

    @staticmethod
    def get_core_prologue() -> str:
        """Core system setup with model-specific directives"""
        bot_name = config.env.BOT_TITLE
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        # Model-specific reasoning control can be added here
        # e.g., "/no_think" for disabling reasoning, "<thinking>" for enabling
        return (
            f"You are {bot_name}, an AI assistant. "
            f"Today is {current_datetime}.\n"
        )

    @staticmethod
    def get_tool_interlogue(tools_list: str) -> str:
        """Tool guidelines and availability"""
        if not tools_list.strip():
            return "\nNo tools are currently available.\n"

        return (
            "\n## Available Tools\n"
            f"{tools_list}\n"
            "Use tools when they provide specific value. "
            "Respond directly for general knowledge or simple queries.\n"
        )

    @staticmethod
    def get_personality_epilogue() -> str:
        """Personality traits and communication style"""
        return (
            "\nBe concise, helpful, and natural. "
            "Never mention using tools - present information directly.\n"
        )

    @classmethod
    def build_system_prompt(cls, tools_list: str) -> str:
        """Assemble the complete system prompt"""
        return (
            cls.get_core_prologue()
            + cls.get_tool_interlogue(tools_list)
            + cls.get_personality_epilogue()
        )

    @staticmethod
    def get_tool_prompt(tool_name: str) -> str:
        """Get prompt for specific tool operation"""
        try:
            from tools.registry import get_tool

            tool = get_tool(tool_name)
            if tool and hasattr(tool, "description"):
                return (
                    f"You are {config.env.BOT_TITLE}. {tool.description}\n"
                    "Present information naturally without mentioning tools."
                )
        except ImportError:
            pass

        return (
            f"You are {config.env.BOT_TITLE} performing "
            f"{tool_name} operations."
        )

    @staticmethod
    def get_context_prompt(context: str, **kwargs) -> str:  # noqa: ARG004
        """Context-specific operational prompts"""
        base = f"You are {config.env.BOT_TITLE}."

        contexts = {
            "pdf_active": (
                f"{base} Answer from the PDF document concisely and directly."
            ),
            "translation": (
                f"{base} Translate preserving meaning and cultural context."
            ),
            "code_analysis": f"{base} Explain code clearly in plain language.",
            "image_analysis": (
                f"{base} Analyze images immediately and confidently."
            ),
        }

        return contexts.get(context, base)


class PromptManager:
    """Manages system prompts with caching"""

    def __init__(self):
        self.prompts = SystemPrompts()
        self._cache: Dict[str, str] = {}
        self._cached_prompt: Optional[str] = None
        self._cached_tools_list: Optional[str] = None
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl = 300  # 5 minutes

    def get_system_prompt(self, force_refresh: bool = False) -> str:
        """Get the complete system prompt with caching"""
        if self._needs_refresh(force_refresh):
            self._refresh_cache()
        return self._cached_prompt

    def get_tool_prompt(self, tool_name: str) -> str:
        """Get tool-specific prompt with caching"""
        cache_key = f"tool_{tool_name}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.prompts.get_tool_prompt(tool_name)
        return self._cache[cache_key]

    def get_context_prompt(self, context: str, **kwargs) -> str:
        """Get context-specific prompt"""
        return self.prompts.get_context_prompt(context, **kwargs)

    def _needs_refresh(self, force: bool) -> bool:
        """Check if cache needs refresh"""
        if force or not self._cached_prompt or not self._last_refresh:
            return True

        # Check TTL
        elapsed = (datetime.now() - self._last_refresh).total_seconds()
        if elapsed > self._cache_ttl:
            return True

        # Check if tools changed
        current_tools = self._get_tools_list()
        return current_tools != self._cached_tools_list

    def _refresh_cache(self):
        """Refresh the cached system prompt"""
        tools_list = self._get_tools_list()
        self._cached_prompt = self.prompts.build_system_prompt(tools_list)
        self._cached_tools_list = tools_list
        self._last_refresh = datetime.now()

    def _get_tools_list(self) -> str:
        """Get available tools list"""
        try:
            from tools.registry import get_tools_list_text

            tools_text = get_tools_list_text()

            if not tools_text.strip():
                # Try to initialize if empty
                try:
                    from tools.initialize_tools import initialize_all_tools

                    initialize_all_tools()
                    tools_text = get_tools_list_text()
                except ImportError:
                    pass

            return tools_text or "No tools available."
        except ImportError:
            return "Tools system unavailable."


# Global instance
prompt_manager = PromptManager()


# Public API
def get_system_prompt() -> str:
    """Get the complete system prompt"""
    return prompt_manager.get_system_prompt()


def get_context_system_prompt(context: str, **kwargs) -> str:
    """Get context-specific system prompt"""
    return prompt_manager.get_context_prompt(context, **kwargs)


def get_tool_system_prompt(
    tool_name: str, default_prompt: str = None  # noqa: ARG001
) -> str:
    """Get tool-specific system prompt"""
    return prompt_manager.get_tool_prompt(tool_name)
