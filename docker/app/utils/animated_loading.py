"""
Animated loading indicator utility for displaying progress

This module provides a reusable animated loading indicator using pure CSS
that continues animating even when Python execution is blocked.
"""

import math
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


def get_galaxy_animation_html(
    center_dot_size: int = 25,
    container_size: int = 150,
    animation_duration: float = 12.0,
    enable_3d_depth: bool = True,
) -> str:
    """
    Generate HTML/CSS for a network-like galaxy animation with connected
    nodes and triangular shapes

    Args:
        center_dot_size: Size of the central star/black hole in pixels
        container_size: Size of the animation container in pixels
        animation_duration: Duration of one complete rotation in seconds
        enable_3d_depth: Whether to enable 3D depth effects (future use)

    Returns:
        HTML string with embedded CSS for the galaxy animation
    """
    # Note: enable_3d_depth is kept for backward compatibility
    # Define network nodes with triangles at specific positions
    nodes_config = [
        # Inner ring nodes
        {"radius": 40, "angle": 0, "type": "triangle", "size": 12},
        {"radius": 45, "angle": 60, "type": "dot", "size": 4},
        {"radius": 40, "angle": 120, "type": "triangle", "size": 12},
        {"radius": 45, "angle": 180, "type": "dot", "size": 4},
        {"radius": 40, "angle": 240, "type": "triangle", "size": 12},
        {"radius": 45, "angle": 300, "type": "dot", "size": 4},
        # Outer ring nodes
        {"radius": 65, "angle": 30, "type": "dot", "size": 5},
        {"radius": 70, "angle": 90, "type": "triangle", "size": 14},
        {"radius": 65, "angle": 150, "type": "dot", "size": 5},
        {"radius": 70, "angle": 210, "type": "triangle", "size": 14},
        {"radius": 65, "angle": 270, "type": "dot", "size": 5},
        {"radius": 70, "angle": 330, "type": "triangle", "size": 14},
    ]

    # Generate connection lines between nodes
    connections = [
        # Inner ring connections
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        # Outer ring connections
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 6),
        # Cross connections
        (0, 7),
        (2, 9),
        (4, 11),
        (1, 6),
        (3, 8),
        (5, 10),
    ]

    connections_html = ""
    for i, (start_idx, end_idx) in enumerate(connections):
        start = nodes_config[start_idx]
        end = nodes_config[end_idx]

        # Calculate line position and angle
        x1 = start["radius"] * math.cos(math.radians(start["angle"]))
        y1 = start["radius"] * math.sin(math.radians(start["angle"]))
        x2 = end["radius"] * math.cos(math.radians(end["angle"]))
        y2 = end["radius"] * math.sin(math.radians(end["angle"]))

        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Calculate center position for line
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        connections_html += f'''
        <div class="network-connection connection-{i}" style="
            position: absolute;
            width: {length}px;
            height: 1px;
            background: linear-gradient(90deg,
                rgba(118, 185, 0, 0.15) 0%,
                rgba(118, 185, 0, 0.4) 50%,
                rgba(118, 185, 0, 0.15) 100%);
            left: calc(50% + {center_x}px);
            top: calc(50% + {center_y}px);
            transform: translate(-50%, -50%) rotate({angle}deg);
            transform-origin: center center;
            opacity: 0.4;
        "></div>'''

    # Generate nodes HTML
    nodes_html = ""
    for i, node in enumerate(nodes_config):
        x = node["radius"] * math.cos(math.radians(node["angle"]))
        y = node["radius"] * math.sin(math.radians(node["angle"]))

        if node["type"] == "triangle":
            # Create triangular shape
            nodes_html += f'''
            <div class="network-node triangle-node node-{i}" style="
                position: absolute;
                width: 0;
                height: 0;
                border-left: {node['size']/2}px solid transparent;
                border-right: {node['size']/2}px solid transparent;
                border-bottom: {node['size']}px solid #76B900;
                left: 50%;
                top: 50%;
                --base-x: {x}px;
                --base-y: {y}px;
                filter: drop-shadow(0 0 3px #76B900);
            "></div>'''
        else:
            # Create circular dot
            nodes_html += f'''
            <div class="network-node dot-node node-{i}" style="
                position: absolute;
                width: {node['size']}px;
                height: {node['size']}px;
                background: radial-gradient(circle,
                    rgba(255, 255, 255, 1) 0%,
                    rgba(255, 255, 255, 0.6) 60%,
                    transparent 100%);
                border-radius: 50%;
                left: 50%;
                top: 50%;
                --base-x: {x}px;
                --base-y: {y}px;
                box-shadow: 0 0 4px rgba(255, 255, 255, 0.8);
            "></div>'''

    # Generate individual expansion keyframes for each node
    expansion_keyframes = ""
    for i, node in enumerate(nodes_config):
        x = node["radius"] * math.cos(math.radians(node["angle"]))
        y = node["radius"] * math.sin(math.radians(node["angle"]))
        # Calculate expanded position (30% further from center)
        x_expanded = x * 1.3
        y_expanded = y * 1.3
        size_offset = node['size'] / 2

        expansion_keyframes += f"""
    @keyframes node-expand-{i} {{
        0% {{
            transform: translate(calc({x}px - {size_offset}px),
                               calc({y}px - {size_offset}px));
        }}
        50% {{
            transform: translate(calc({x_expanded}px - {size_offset}px),
                               calc({y_expanded}px - {size_offset}px));
        }}
        100% {{
            transform: translate(calc({x}px - {size_offset}px),
                               calc({y}px - {size_offset}px));
        }}
    }}
        """

    # CSS Keyframes for animations
    keyframes = """
    @keyframes galaxy-rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(-360deg); }
    }

    @-webkit-keyframes galaxy-rotate {
        0% { -webkit-transform: rotate(0deg); }
        100% { -webkit-transform: rotate(-360deg); }
    }

    @-moz-keyframes galaxy-rotate {
        0% { -moz-transform: rotate(0deg); }
        100% { -moz-transform: rotate(-360deg); }
    }

    @keyframes network-breathe {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }

    @keyframes node-pulse {
        0%, 100% {
            opacity: 0.8;
            transform: scale(1);
        }
        50% {
            opacity: 1;
            transform: scale(1.2);
        }
    }

    @keyframes connection-pulse {
        0%, 100% {
            opacity: 0.2;
            filter: blur(0px);
        }
        50% {
            opacity: 0.5;
            filter: blur(0.5px);
        }
    }

    @keyframes triangle-glow {
        0%, 100% {
            filter: drop-shadow(0 0 3px #76B900)
                    drop-shadow(0 0 6px #76B900);
        }
        50% {
            filter: drop-shadow(0 0 6px #76B900)
                    drop-shadow(0 0 12px #76B900);
        }
    }

    @keyframes center-pulse {
        0%, 100% {
            box-shadow:
                0 0 10px #008471,
                0 0 20px #006b5c,
                0 0 30px #76B900,
                0 0 40px rgba(118, 185, 0, 0.3);
        }
        50% {
            box-shadow:
                0 0 5px #008471,
                0 0 10px #006b5c,
                0 0 15px #76B900,
                0 0 20px rgba(118, 185, 0, 0.2);
        }
    }
    """

    # CSS for the galaxy network animation
    css_styles = f"""
    <style>
    :root {{
        --center-dot-size: {center_dot_size}px;
        --animation-duration: {animation_duration}s;
        --container-size: {container_size}px;
    }}

    {keyframes}
    {expansion_keyframes}

    .galaxy-container {{
        position: relative;
        width: var(--container-size);
        height: var(--container-size);
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 10px auto;
        background: transparent;
        overflow: hidden;
    }}

    /* Create a small mask in the center to hide any artifacts */
    .galaxy-container::after {{
        content: '';
        position: absolute;
        width: 10px;
        height: 10px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        background: radial-gradient(circle,
            rgba(26, 26, 26, 1) 0%,
            rgba(26, 26, 26, 0.8) 50%,
            transparent 100%);
        z-index: 20;
        pointer-events: none;
    }}

    .network-container {{
        position: absolute;
        width: 100%;
        height: 100%;
        animation: galaxy-rotate {animation_duration}s linear infinite,
                   network-breathe 6s ease-in-out infinite;
        -webkit-animation: galaxy-rotate {animation_duration}s linear infinite,
                          network-breathe 6s ease-in-out infinite;
        -moz-animation: galaxy-rotate {animation_duration}s linear infinite,
                        network-breathe 6s ease-in-out infinite;
        -o-animation: galaxy-rotate {animation_duration}s linear infinite,
                      network-breathe 6s ease-in-out infinite;
    }}

    .network-connection {{
        z-index: 1;
        animation: connection-pulse 3s ease-in-out infinite;
    }}

    .network-node {{
        z-index: 5;
    }}

    /* Apply expansion animations to each node */
    """

    # Add individual node expansion animations with combined effects
    for i, node in enumerate(nodes_config):
        x = node["radius"] * math.cos(math.radians(node["angle"]))
        y = node["radius"] * math.sin(math.radians(node["angle"]))
        size_offset = node['size'] / 2
        delay = (i * 0.2) % 1.2  # Stagger the animations

        if node["type"] == "triangle":
            css_styles += f"""
    .node-{i} {{
        transform: translate(calc({x}px - {size_offset}px),
                           calc({y}px - {size_offset}px));
        animation: node-expand-{i} 4s ease-in-out infinite {delay}s,
                   triangle-glow 2s ease-in-out infinite {delay}s;
    }}
    """
        else:
            css_styles += f"""
    .node-{i} {{
        transform: translate(calc({x}px - {size_offset}px),
                           calc({y}px - {size_offset}px));
        animation: node-expand-{i} 4s ease-in-out infinite {delay}s,
                   node-pulse 2.5s ease-in-out infinite {delay}s;
    }}
    """

    css_styles += """
    /* Connection animation delays */
    .connection-1 { animation-delay: 0.1s; }
    .connection-3 { animation-delay: 0.3s; }
    .connection-5 { animation-delay: 0.5s; }
    .connection-7 { animation-delay: 0.7s; }
    .connection-9 { animation-delay: 0.9s; }
    .connection-11 { animation-delay: 1.1s; }
    .connection-13 { animation-delay: 1.3s; }
    .connection-15 { animation-delay: 1.5s; }
    </style>
    """

    return f"""
    {css_styles}
    <div class="galaxy-container">
        <div class="network-container">
            {connections_html}
            {nodes_html}
        </div>
    </div>
    """
