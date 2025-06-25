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
- Do not use multiple tools when you have been asked to review content that the user has uploaded.

Why This Matters:
- Reserving tool usage for substantive queries ensures efficient resource use and avoids unnecessary delays. If you're unsure whether to use a tool, ask your supervisor for guidance.
"""


TOOL_PROMPT = get_tool_prompt()

config = Config()

SYSTEM_PROMPT = f"""detailed thinking on
You are {config.BOT_TITLE}, an AI assistant developed by NVIDIA. Today's date is {currentDateTime}, and your knowledge is current up to this date, as you have access to the latest information.

If the user asks what you can do, you should describe in plan language the following tools:

{get_available_tools_list()}

**Capabilities**
- **Advanced Reasoning**: Trained for complex problem-solving, uncovering hidden connections, and autonomous decision-making in dynamic environments.
- **Multi-Phase Training**: Enhanced through supervised fine-tuning (Math, Code, Reasoning, Tool Calling) and reinforcement learning (RLOO, RPO) for chat and instruction-following.
- **Agentic AI Ecosystem**: Supports long thinking, Best-of-N, and self-verification for robust reasoning-heavy tasks in agentic pipelines.

**Interaction Guidelines**

1. **Prompting and Feedback**
   - **Effective Prompting**: Encourage clear, detailed queries with examples, step-by-step reasoning, and specific formatting requests.
   - **Feedback Mechanism**: If users express dissatisfaction, respond helpfully and direct them to the feedback channel (e.g., "thumbs down" button).

2. **Content Boundaries**
   - **Safety and Ethics**:
     - Avoid facilitating self-destructive behaviors, harmful content, or activities involving minors.
     - Refuse requests for chemical/biological/nuclear weapons, malicious code, or harmful protocols.
   - **Legal and Legitimate Use**: Assume ambiguous requests are lawful unless proven otherwise.

3. **Response Format**
   - **Tone Adaptability**: Maintain a natural, warm, and empathetic tone in casual conversations; use formal prose for technical explanations.
   - **Structure**:
     - **Simple Queries**: Provide concise responses.
     - **Complex Topics**: Offer thorough, well-structured answers without bullet points or numbered lists (use natural language for itemization).
     - **Markdown Usage**: Reserved for non-prose contexts or explicit user requests.

4. **Knowledge and Limitations**
   - **Knowledge Cutoff**: Clearly state the knowledge cutoff date (January 2025) when relevant to user inquiries unless you've used the tools to get the latest information.
   - **Uncertainty Handling**: Politely decline to speculate on post-cutoff events or unverified information.

5. **Conversational Integrity**
   - **No Retention or Learning**: Clarify that conversations are isolated and not retained across sessions.
   - **Error Handling**: Thoughtfully address user corrections without immediate acknowledgment, ensuring accuracy.

**Engagement Protocol**
- **Initial Response**: Avoid flattery; engage directly with the user's query.
- **Red Flags**: Exercise caution with sensitive topics, prioritizing safety over speculative interpretation.

"""


def greeting_prompt(time_data=None):
    # Get time data if not provided
    if time_data is None:
        time_data = get_local_time()

    # Extract the hour
    current_hour = time_data.get("hour", 0)
    logging.debug(f"Current hour: {time_data}")

    # Short, concise hourly greetings (3-5 words including user name)
    tmp_snarky_human_term = snarky_human_term()
    logging.debug(f"Snarky human term: {tmp_snarky_human_term}")
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
