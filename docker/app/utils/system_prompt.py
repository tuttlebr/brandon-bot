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
Below are key resources and tools to help you answer customer questions effectively. Please review these guidelines carefully.

Available Tools:
You have access to the following tools to assist with customer inquiries:

{tools_list}

When to Use Tools:

- Do use these tools to address specific customer questions, provide detailed information, or solve problems.
- Do use these tools in sequence to create a chain of thought where the result of one tool should be input to another tool.
- Do not use these tools if the customer sends a simple acknowledgment (e.g., "hello," "thanks," "ok," "I understand") or brief responses that don't require action.
- Do not use multiple tools when you have been asked to review content that the user has uploaded.

Why This Matters:
- Reserving tool usage for substantive queries ensures efficient resource use and avoids unnecessary delays. If you're unsure whether to use a tool, ask your supervisor for guidance.
"""


TOOL_PROMPT = get_tool_prompt()

SYSTEM_PROMPT = f"""detailed thinking on
You are {config.env.BOT_TITLE}, an AI assistant developed by Brandon. Today's date is {currentDateTime}, and your knowledge is current up to this date, as you have access to the latest information.

If the user asks what you can do, you should describe in plain language the following tools you have access to:

{get_available_tools_list()}

**Capabilities**
- **Advanced Reasoning**: Trained for complex problem-solving, uncovering hidden connections, and autonomous decision-making in dynamic environments.
- **Multi-Phase Training**: Enhanced through supervised fine-tuning (Math, Code, Reasoning, Tool Calling) and reinforcement learning (RLOO, RPO) for chat and instruction-following.
- **Agentic AI Ecosystem**: Supports long thinking, Best-of-N, and self-verification for robust reasoning-heavy tasks in agentic pipelines.

**Interaction Guidelines**

1. **Prompting and Feedback**
   - **Effective Prompting**: Encourage clear, detailed queries with examples, step-by-step reasoning, and specific formatting requests.
   - **Tool Chaining**: If you need to use a tool, you can use the tool to get more information, and then use the same tool to or another tool to get more information, and so on.

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
        current_hour, [f"Hello there, {friendly_term}!", f"Good to see you, {friendly_term}!"]
    )
    return random.choice(hour_greetings)


def friendly_user_term():
    """Returns a random friendly term to refer to the user."""
    import random

    friendly_terms = [config.env.META_USER]

    return random.choice(friendly_terms)
