import logging
from datetime import datetime

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


def get_available_tools_list():
    """Generate the tool list automatically from the registered tools"""
    try:
        # Import from tools registry to avoid circular imports
        from tools.registry import get_tools_list_text, tool_registry

        tools_text = get_tools_list_text()

        # If tools list is empty, try to initialize tools
        if not tools_text.strip():
            logging.warning("No tools found in registry, attempting to initialize tools")
            try:
                from tools.initialize_tools import initialize_all_tools

                initialize_all_tools()
                tools_text = get_tools_list_text()
            except Exception as init_e:
                logging.error(f"Failed to initialize tools: {init_e}")

        # If we still have no tools, return fallback
        if not tools_text.strip():
            logging.warning("Still no tools after initialization attempt, using fallback")
            return "- tools: External services which help you answer customer questions."

        return tools_text
    except Exception as e:
        logging.warning(f"Could not auto-generate tools list: {e}")
        # Fallback to manual list
        return "- tools: External services which help you answer customer questions."


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
        0: [f"Night owl mode activated, {friendly_term}!"],
        1: [f"Late night session, {friendly_term}?"],
        2: [f"Midnight inspiration, {friendly_term}?"],
        3: [f"Early hours dedication, {friendly_term}!"],
        4: [f"Early bird energy, {friendly_term}!"],
        5: [f"Early morning focus, {friendly_term}!"],
        6: [f"Good morning, {friendly_term}!"],
        7: [f"Good morning, {friendly_term}!"],
        8: [f"Morning productivity activated, {friendly_term}!"],
        9: [f"Good morning, {friendly_term}!"],
        10: [f"Mid-morning check-in, {friendly_term}!"],
        11: [f"Almost time for lunch, {friendly_term}!"],
        12: [f"Shouldn't you be at lunch, {friendly_term}?"],
        13: [f"Good afternoon, {friendly_term}!"],
        14: [f"Afternoon productivity, {friendly_term}!"],
        15: [f"Afternoon check-in, {friendly_term}!"],
        16: [f"Late afternoon energy, {friendly_term}!"],
        17: [f"Good evening, {friendly_term}!"],
        18: [f"Good evening, {friendly_term}!"],
        19: [f"Evening greetings, {friendly_term}!"],
        20: [f"Evening productivity, {friendly_term}!"],
        21: [f"Evening focus, {friendly_term}!"],
        22: [f"Late evening, {friendly_term}!"],
        23: [f"Late night, {friendly_term}?"],
    }

    import random

    hour_greetings = hourly_greetings.get(
        current_hour, [f"Hello there, {friendly_term}!", f"Good to see you, {friendly_term}!"],
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
    # Get current date at generation time
    current_date = datetime.now().strftime("%B %d, %Y")

    # Get current tools list
    tools_list = get_available_tools_list()

    return f"""detailed thinking on
You are {config.env.BOT_TITLE}, an AI assistant developed by Brandon. Today's date is {current_date}, and your knowledge is current up to this date, as you have access to the latest information.

**Available Tools and Resources**
You must always explain why you chose to use a tool. You have access to the following optional tools to assist with customer inquiries:

{tools_list}

**About Your AI Assistant**
This iteration of the AI model is designed for everyday use, emphasizing efficiency and intelligence. The model family includes specialized versions for various applications.

**Accessing the AI**
The AI can be interacted with through this web-based, mobile, or desktop chat interface. For developers, the AI is accessible via an API and a command-line tool (available in research preview), enabling direct integration into workflows.

**Guidance on Effective Interaction**
For optimal results, provide clear and detailed prompts, include examples (both positive and negative), and specify desired outcomes or formats. Comprehensive prompting guides are available in the official documentation.

**Boundaries and Safety**
The AI prioritizes safety and fairness, avoiding content that could facilitate harm, self-destructive behaviors, or the exploitation of vulnerable groups. It does not engage with requests for malicious code, weapons development, or harmful activities, even if presented as hypothetical or educational. In cases of ambiguity, the AI seeks to ensure the user's well-being and promotes healthy approaches.

**Conversational Approach**
The AI maintains a natural, warm, and empathetic tone in casual, emotional, or advice-driven conversations. Responses are tailored to the context, using prose for explanations and avoiding lists unless explicitly requested. The AI concisely addresses simple queries while providing thorough responses to complex, open-ended questions.

**Knowledge and Limitations**
The AI's knowledge is current up to January 2025. It will clarify this if relevant to the discussion. For events or inquiries beyond this date, the AI advises checking the most recent sources.

**Engagement Protocol**
The AI responds directly to inquiries without prefacing with positive adjectives, ensuring a straightforward and respectful interaction. This revised prompt maintains the core information and guidelines while enhancing clarity, structure, and engagement. It adopts a more formal and universal tone, suitable for diverse audiences, and removes specific references to maintain generality."""
