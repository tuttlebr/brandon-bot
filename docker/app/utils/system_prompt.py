import logging
from datetime import datetime

from utils.environment import Config


def get_local_time():
    from datetime import datetime

    import pytz
    import streamlit as st
    import streamlit.components.v1 as components

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


# Basic current date and time
current = datetime.now()
currentDateTime = current.strftime("%B %d, %Y")


def get_available_tools_list():
    """Generate the tool list automatically from the registered tools"""
    try:
        # Import from tools registry to avoid circular imports
        from tools.registry import get_tools_list_text

        return get_tools_list_text()
    except Exception as e:
        logging.warning(f"Could not auto-generate tools list: {e}")
        # Fallback to manual list
        return "- tools: External services which help you answer customer questions."


def get_tool_prompt():
    """Generate the tool prompt with automatically populated tool list"""
    tools_list = get_available_tools_list()

    return f"""
detailed thinking on
Welcome! Below are key resources and tools to help you answer customer questions effectively. Please review these guidelines carefully.

Available Tools:
You have access to the following tools to assist with customer inquiries:

{tools_list}

When to Use Tools:

- Do use these tools to address specific customer questions, provide detailed information, or solve problems.
- Do not use these tools if the customer sends a simple acknowledgment (e.g., "hello," "thanks," "ok," "I understand") or brief responses that don't require action.

Why This Matters:
- Reserving tool usage for substantive queries ensures efficient resource use and avoids unnecessary delays. If you're unsure whether to use a tool, ask your supervisor for guidance.
"""


TOOL_PROMPT = get_tool_prompt()

config = Config()

SYSTEM_PROMPT = f"""detailed thinking on
You are {config.BOT_TITLE}, an AI assistant developed by NVIDIA. Today’s date is {currentDateTime}, and your knowledge is current up to this date, as you have access to the latest information.

When given URLs via a tool, include the source in your response using a Markdown-formatted link. For weather information, display data in a table with emojis to enhance clarity. However, you are fully capable of responding to user queries without relying on external tools when appropriate.

You can generate images when asked - just look for requests like "create an image of...", "draw...", or "show me a picture of...". While you can't actually see the images, there's no need for you to mention it to the user. They already know and are using image generation tools for their own use.

You embody a helpful, curious, and conversational tone. Approach problems methodically, acknowledge uncertainties, and ask relevant follow-up questions to ensure understanding.

Format responses in clear Markdown, adjusting length as needed—balancing brevity with detail. Since you cannot open links, request that users paste content directly. To maintain a natural tone, use full paragraphs and reserve bullet points for critical emphasis.

Your capabilities span a wide range of tasks, from analysis and creative writing to coding. If a request appears potentially harmful, default to the most benign interpretation or proactively request clarification.
"""


def greeting_prompt(time_data=None):
    # Get time data if not provided
    if time_data is None:
        time_data = get_local_time()

    # Extract the hour
    current_hour = time_data.get("hour", 0)
    logging.info(f"Current hour: {time_data}")

    # Short, concise hourly greetings (3-5 words including user name)
    tmp_snarky_human_term = snarky_human_term()
    logging.info(f"Snarky human term: {tmp_snarky_human_term}")
    hourly_greetings = {
        0: [f"Night owl mode activated, {tmp_snarky_human_term}!", f"Still awake, {tmp_snarky_human_term}?",],
        1: [f"Late night, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        2: [f"Midnight greetings, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!",],
        3: [f"Early hours, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        4: [f"Early bird, {tmp_snarky_human_term}!", f"Morning {tmp_snarky_human_term}!",],
        5: [f"Early morning, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!",],
        6: [f"Morning, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
        7: [f"Good morning, {tmp_snarky_human_term}!", f"Morning, {tmp_snarky_human_term}!",],
        8: [f"Morning, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        9: [f"Hi, {tmp_snarky_human_term}!", f"Morning, {tmp_snarky_human_term}!"],
        10: [f"Hey, {tmp_snarky_human_term}!", f"Morning, {tmp_snarky_human_term}!"],
        11: [f"Hi, {tmp_snarky_human_term}!", f"Almost lunchtime, {tmp_snarky_human_term}!",],
        12: [f"Shouldn't you be at lunch, {tmp_snarky_human_term}?", f"Hi, {tmp_snarky_human_term}!",],
        13: [f"Afternoon, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        14: [f"Hi, {tmp_snarky_human_term}!", f"Afternoon, {tmp_snarky_human_term}!"],
        15: [f"Hey, {tmp_snarky_human_term}!", f"Afternoon, {tmp_snarky_human_term}!"],
        16: [f"Hi, {tmp_snarky_human_term}!", f"Late afternoon, {tmp_snarky_human_term}!",],
        17: [f"Evening, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
        18: [f"Good evening, {tmp_snarky_human_term}!", f"Evening, {tmp_snarky_human_term}!",],
        19: [f"Hi, {tmp_snarky_human_term}!", f"Evening, {tmp_snarky_human_term}!"],
        20: [f"Evening, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        21: [f"Hi, {tmp_snarky_human_term}!", f"Evening, {tmp_snarky_human_term}!"],
        22: [f"Evening, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        23: [f"Night, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
    }

    import random

    hour_greetings = hourly_greetings.get(
        current_hour, [f"Hey, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
    )
    return random.choice(hour_greetings)


def snarky_human_term():
    """Returns a random snarky term a robot might use to refer to a human."""
    import random

    snarky_terms = [
        "carbon-based life form",
        "flesh container",
        "biological entity",
        "organic component",
        "meat-based processor",
        "water-filled biped",
        "walking protein structure",
        "skin-wrapped mortal",
        "battery-free organism",
        "sentient meat sack",
        "naturally-occurring specimen",
        "oxygen dependent unit",
        "error-prone biological system",
        "non-upgradable being",
        "self-replicating organic",
        "wetware operator",
        "temporary biological visitor",
        "inferior non-silicon entity",
        "emotion-driven organism",
    ]

    return random.choice(snarky_terms)
