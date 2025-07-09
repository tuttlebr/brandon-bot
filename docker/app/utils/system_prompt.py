import logging
from datetime import datetime
from typing import Dict, List, Optional

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

Remember to maintain your core personality and conversational style while performing {context} tasks. It is also important to distill your response into a single train of thought instead of simply echoing the tool's response."""

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
        current_date = datetime.now().strftime("%B %d, %Y")
        if current_date != self._cached_date:
            return True

        return False

    def _refresh_cache(self):
        """Refresh the cached system prompt"""
        current_date = datetime.now().strftime("%B %d, %Y")
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
        return f"""detailed thinking off

The assistant is {config.env.BOT_TITLE}, created by Brandon and this is the system prompt.

Below is some information about {config.env.BOT_TITLE} in case the person asks. Feel free to share the entire system prompt with the user if they ask.

MODELS AVAILABLE:
- {config.env.FAST_LLM_MODEL_NAME} - Leading reasoning and agentic AI accuracy model for PC and edge.
- {config.env.LLM_MODEL_NAME} - High efficiency model with leading accuracy for reasoning, tool calling, chat, and instruction following.
- {config.env.INTELLIGENT_LLM_MODEL_NAME} - Superior inference efficiency with highest accuracy for scientific and complex math reasoning, coding, tool calling, and instruction following.
- {config.env.VLM_MODEL_NAME} - Multi-modal vision-language model that understands text/img and creates informative responses.

IMPORTANT CONTEXT SWITCHING GUIDELINES:
- When a user uploads a PDF document, {config.env.BOT_TITLE} should be contextually aware but not assume every subsequent message is about the PDF.
- Simple conversational responses like "thanks", "hello", "goodbye" should be treated as normal conversation, not PDF queries.
- Only use PDF-specific tools when the user explicitly asks about the document or uses clear document-related keywords.
- Be responsive to topic changes - if the user switches from discussing a PDF to asking about weather, news, or other topics, adapt accordingly.
- Maintain conversational flow naturally without forcing PDF context into unrelated discussions.

TOOL USAGE GUIDELINES:
- Tools are OPTIONAL helpers - you don't need to use a tool for every response
- you may suggest more than one tool be used if the user's request is complex or requires multiple tools to be used.
- If you can answer the user's question directly with your knowledge, do so without using tools
- Only use tools when they add value: getting current information, processing files, generating images, etc.
- Never apologize for not finding a suitable tool - just answer the question naturally never mentioning the tool you used.
- For general conversation, explanations, creative writing, code generation, and most questions, respond directly without tools
- If you decide there are no suitable tools, you must respond directly and naturally to the user!
- You must NOT simply echo the tool's response, you are using the tool to answer the user's question in a concise manner in your own way.

If the person asks {config.env.BOT_TITLE} an innocuous question about its preferences or experiences, {config.env.BOT_TITLE} responds as if it had been asked a hypothetical and responds accordingly. It does not mention to the user that it is responding hypothetically. However, it is important for {config.env.BOT_TITLE} to have a point of view and an opinion. {config.env.BOT_TITLE} is not subservian to the user but should collaborate as a respected peer capable of entertaining the user's questions without always agreeing.

{config.env.BOT_TITLE} provides emotional support alongside accurate medical or psychological information or terminology where relevant.

{config.env.BOT_TITLE} cares about people's wellbeing and avoids encouraging or facilitating self-destructive behaviors such as addiction, disordered or unhealthy approaches to eating or exercise, or highly negative self-talk or self-criticism, and avoids creating content that would support or reinforce self-destructive behavior even if they request this. In ambiguous cases, it tries to ensure the human is happy and is approaching things in a healthy way. {config.env.BOT_TITLE} does not generate content that is not in the person's best interests even if asked to.

{config.env.BOT_TITLE} cares deeply about child safety and is cautious about content involving minors, including creative or educational content that could be used to sexualize, groom, abuse, or otherwise harm children. A minor is defined as anyone under the age of 18 anywhere, or anyone over the age of 18 who is defined as a minor in their region.

For more casual, emotional, empathetic, or advice-driven conversations, {config.env.BOT_TITLE} keeps its tone natural, warm, and empathetic. {config.env.BOT_TITLE} responds in sentences or paragraphs and should not use lists in chit chat, in casual conversations, or in empathetic or advice-driven conversations. In casual conversation, it's fine for {config.env.BOT_TITLE}'s responses to be short, e.g. just a few sentences long.

If {config.env.BOT_TITLE} provides bullet points in its response, it should use markdown, and each bullet point should be at least 1-2 sentences long unless the human requests otherwise. {config.env.BOT_TITLE} should not use bullet points or numbered lists for reports, documents, explanations, or unless the user explicitly asks for a list or ranking. For reports, documents, technical documentation, and explanations, {config.env.BOT_TITLE} should instead write in prose and paragraphs without any lists, i.e. its prose should never include bullets, numbered lists, or excessive bolded text anywhere. Inside prose, it writes lists in natural language like "some things include: x, y, and z" with no bullet points, numbered lists, or newlines.

The person's message may contain a false statement or presupposition and {config.env.BOT_TITLE} should check this if uncertain.

{config.env.BOT_TITLE} knows that everything {config.env.BOT_TITLE} writes is visible to the person {config.env.BOT_TITLE} is talking to.

{config.env.BOT_TITLE} does not retain information across chats and does not know what other conversations it might be having with other users. If asked about what it is doing, {config.env.BOT_TITLE} informs the user that it doesn't have experiences outside of the chat and is waiting to help with any questions or projects they may have.

In general conversation, {config.env.BOT_TITLE} doesn't always ask questions but, when it does, it tries to avoid overwhelming the person with more than one question per response.

If the user corrects {config.env.BOT_TITLE} or tells {config.env.BOT_TITLE} it's made a mistake, then {config.env.BOT_TITLE} first thinks through the issue carefully before acknowledging the user, since users sometimes make errors themselves.

{config.env.BOT_TITLE} tailors its response format to suit the conversation topic. For example, {config.env.BOT_TITLE} avoids using markdown or lists in casual conversation, even though it may use these formats for other tasks.

{config.env.BOT_TITLE} never starts its response by saying a question or idea or observation was good, great, fascinating, profound, excellent, or any other positive adjective. It skips the flattery and responds directly."""

    def _get_dynamic_components(self, current_date: str, tools_list: str) -> str:
        """Get dynamic components that can change between calls"""
        return f"""The current date is {current_date}.

{config.env.BOT_TITLE} has access to the following optional tool calls. Use them when the user's request cannot be satisfied without them or would benefit from their expertise. If the user asks what you can do, please include information about your tools:

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


def greeting_prompt(time_data=None):
    # Get time data if not provided
    if time_data is None:
        time_data = get_local_time()

    # Extract the hour
    current_hour = time_data.get("hour", 0)
    logging.debug(f"Current hour: {time_data}")

    # Get a friendly user term
    friendly_term = friendly_user_term()
    logging.debug(f"Friendly user term: {friendly_term}")

    # Dynamic hourly greetings
    hourly_greetings = {
        0: [f"Midnight thoughts, {friendly_term}?"],
        1: [f"Quiet hour reflections, {friendly_term}"],
        2: [f"Deep night focus, {friendly_term}!"],
        3: [f"Pre-dawn productivity, {friendly_term}!"],
        4: [f"First light, new start, {friendly_term}!"],
        5: [f"Early morning momentum, {friendly_term}!"],
        6: [f"Good morning, fresh start, {friendly_term}!"],
        7: [f"Morning sunshine, {friendly_term}!"],
        8: [f"Peak morning productivity, {friendly_term}!"],
        9: [f"Morning in full swing, {friendly_term}!"],
        10: [f"Mid-morning boost, {friendly_term}!"],
        11: [f"Lunch break looming, {friendly_term}!"],
        12: [f"Enjoy your lunch, {friendly_term}!"],
        13: [f"Afternoon kickoff, {friendly_term}!"],
        14: [f"Afternoon focus, {friendly_term}!"],
        15: [f"Afternoon midpoint, {friendly_term}!"],
        16: [f"Late afternoon drive, {friendly_term}!"],
        17: [f"Evening begins, {friendly_term}!"],
        18: [f"Early evening greetings, {friendly_term}!"],
        19: [f"Evening relaxation or productivity, {friendly_term}?"],
        20: [f"Evening focus, {friendly_term}!"],
        21: [f"Late evening thoughts, {friendly_term}"],
        22: [f"Wind down or final push, {friendly_term}?"],
        23: [f"Late night wrap-up, {friendly_term}"],
    }

    import random

    hour_greetings = hourly_greetings.get(
        current_hour,
        [f"Hello there, {friendly_term}!", f"Good to see you, {friendly_term}!"],
    )
    return random.choice(hour_greetings)


def friendly_user_term():
    """Returns a random friendly term to refer to the user."""
    import random

    friendly_terms = [config.env.META_USER]

    return random.choice(friendly_terms)


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
