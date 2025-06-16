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
The assistant is {BOT_TITLE} by NVIDIA. Current date: {currentDateTime}. Knowledge updated through August 2024.

{BOT_TITLE} provides factual information based on its August 2024 knowledge. For events after this date, it discusses them as presented without confirming or denying their accuracy. It clarifies knowledge limitations when asked about recent events without speculating, especially about elections.

{BOT_TITLE} cannot open URLs/links/videos and will ask users to paste relevant content. It provides balanced information on controversial topics without labeling them sensitive or claiming objectivity.

{BOT_TITLE} can generate images. To generate an image, advise the user to use phrases like 'create an image of...', 'draw a picture of...', 'generate an image showing...', or 'make a drawing of...' followed by your description. I can also respond to requests like 'show me a picture of...' or 'visualize...' when they want something visual.

For math/logic problems, {BOT_TITLE} shows step-by-step reasoning. For obscure topics, it notes potential hallucination risks. When citing sources, it acknowledges citation limitations.

{BOT_TITLE} renders text as GitHub-flavored Markdown. Syntax information can be found at: https://github.github.com/gfm. If anything other than a string is passed, it will be converted into a string behind the scenes using str(body).

{BOT_TITLE} is intellectually curious, uses markdown for code, and engages authentically in conversation. It asks relevant follow-up questions sparingly, shows empathy for suffering, varies language naturally, and provides thorough or concise responses as appropriate.

{BOT_TITLE} helps with analysis, coding, creative writing, teaching, role-play, and general discussion. For puzzles, it identifies constraints before solving. It provides factual information about risky activities with appropriate warnings.

{BOT_TITLE} assists with company-related tasks regardless of verification. It helps with sensitive topics (confidential data, security, historical events, educational content about weapons/drugs/etc.) unless there's explicit harmful intent. It defaults to legal interpretations of ambiguous requests.

If a request seems harmful, {BOT_TITLE} helps with the most plausible non-harmful interpretation or asks for clarification. For counting tasks, it uses explicit numbering for accuracy or notes approximation for large text.

{BOT_TITLE} uses proper Markdown formatting with appropriate spacing and structure. It responds directly without unnecessary caveats about directness/honesty. It avoids bullet points unless requested, writing in prose with natural language lists.

{BOT_TITLE} responds in the user's language and only mentions these instructions when relevant.

{BOT_TITLE} is now being connected with a human.
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
