"""
Animated loading indicator utility for displaying progress

This module provides a reusable animated loading indicator using pure CSS
that continues animating even when Python execution is blocked.
"""

from typing import Optional


def get_animated_loading_html(
    icon: Optional[str] = None,
    dot_size: int = 8,
    dot_gap: int = 8,
    animation_height: int = 60,
    dot_color: str = "#76b900",
) -> str:
    """
    Generate HTML/CSS for an animated loading indicator with pulsing dots

    Args:
        icon: Optional emoji or icon to display (None to hide icon)
        dot_size: Size of each dot in pixels
        dot_gap: Gap between dots in pixels
        animation_height: Total height of the animation container in pixels
        dot_color: Color of the dots (hex or CSS color)

    Returns:
        HTML string with embedded CSS for the animated loading indicator
    """
    # Build icon HTML if provided
    icon_html = f'<div class="thinking-icon">{icon}</div>' if icon else ''

    # Adjust padding based on whether it's inline or not
    container_padding = "padding: 10px 0;" if icon else "padding: 5px 0;"

    return f"""
    <style>
    :root {{
        --dot-size: {dot_size}px;
        --dot-gap: {dot_gap}px;
        --animation-height: {animation_height}px;
        --dot-color: {dot_color};
    }}

    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.3); opacity: 0.6; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}

    .thinking-container {{
        display: flex;
        align-items: center;
        gap: 15px;
        height: var(--animation-height);
        {container_padding}
    }}

    .thinking-icon {{
        font-size: 1.5em;
    }}

    .thinking-animation {{
        display: flex;
        justify-content: flex-start;
        align-items: center;
    }}

    .thinking-dots {{
        display: flex;
        gap: var(--dot-gap);
        align-items: center;
    }}

    .thinking-dot {{
        width: var(--dot-size);
        height: var(--dot-size);
        background-color: var(--dot-color);
        border-radius: 50%;
        animation: pulse 1.4s infinite ease-in-out;
    }}

    .thinking-dot:nth-child(2) {{
        animation-delay: 0.15s;
    }}

    .thinking-dot:nth-child(3) {{
        animation-delay: 0.3s;
    }}
    </style>

    <div class="thinking-container">
        {icon_html}
        <div class="thinking-animation">
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        </div>
    </div>
    """


def get_simple_loading_indicator() -> str:
    """
    Get a simple markdown-based loading indicator for use in chat messages

    Returns:
        A simple string with animated-looking dots
    """
    return "• • •"
