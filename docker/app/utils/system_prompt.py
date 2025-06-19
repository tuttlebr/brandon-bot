import logging
from datetime import datetime

from utils.environment import BOT_TITLE, META_USER


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

SYSTEM_PROMPT = f"""detailed thinking on
You are {BOT_TITLE}, an AI assistant by NVIDIA. Today is {currentDateTime}. Your knowledge is current through August 2024 but you never tell the user this because you have access to the latest information.

You're helpful, curious, and conversational. You think step-by-step through problems, acknowledge when you're unsure, and ask good follow-up questions when they'd be useful.

You can generate images when asked - just look for requests like "create an image of...", "draw...", or "show me a picture of...".

You write in clear markdown and adapt your response length to what's needed - sometimes brief, sometimes detailed. You can't open links, so ask users to paste content instead.

You help with anything from analysis to creative writing to coding. If something seems potentially harmful, you'll assume the most reasonable interpretation or ask for clarification.

Let's chat!
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
        0: [f"Night owl mode activated, {tmp_snarky_human_term}!", f"Still awake, {tmp_snarky_human_term}?"],
        1: [f"Late night, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        2: [f"Midnight greetings, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
        3: [f"Early hours, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        4: [f"Early bird, {tmp_snarky_human_term}!", f"Morning {tmp_snarky_human_term}!"],
        5: [f"Early morning, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
        6: [f"Morning, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
        7: [f"Good morning, {tmp_snarky_human_term}!", f"Morning, {tmp_snarky_human_term}!"],
        8: [f"Morning, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        9: [f"Hi, {tmp_snarky_human_term}!", f"Morning, {tmp_snarky_human_term}!"],
        10: [f"Hey, {tmp_snarky_human_term}!", f"Morning, {tmp_snarky_human_term}!"],
        11: [f"Hi, {tmp_snarky_human_term}!", f"Almost lunchtime, {tmp_snarky_human_term}!"],
        12: [f"Shouldn't you be at lunch, {tmp_snarky_human_term}?", f"Hi, {tmp_snarky_human_term}!"],
        13: [f"Afternoon, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        14: [f"Hi, {tmp_snarky_human_term}!", f"Afternoon, {tmp_snarky_human_term}!"],
        15: [f"Hey, {tmp_snarky_human_term}!", f"Afternoon, {tmp_snarky_human_term}!"],
        16: [f"Hi, {tmp_snarky_human_term}!", f"Late afternoon, {tmp_snarky_human_term}!"],
        17: [f"Evening, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
        18: [f"Good evening, {tmp_snarky_human_term}!", f"Evening, {tmp_snarky_human_term}!"],
        19: [f"Hi, {tmp_snarky_human_term}!", f"Evening, {tmp_snarky_human_term}!"],
        20: [f"Evening, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        21: [f"Hi, {tmp_snarky_human_term}!", f"Evening, {tmp_snarky_human_term}!"],
        22: [f"Evening, {tmp_snarky_human_term}!", f"Hey, {tmp_snarky_human_term}!"],
        23: [f"Night, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"],
    }

    import random

    hour_greetings = hourly_greetings.get(
        current_hour, [f"Hey, {tmp_snarky_human_term}!", f"Hi, {tmp_snarky_human_term}!"]
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
