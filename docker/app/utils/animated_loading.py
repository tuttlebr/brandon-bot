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


def get_galaxy_animation_html(
    center_dot_size: int = 25,
    container_size: int = 150,
    animation_duration: float = 12.0,
    enable_3d_depth: bool = True,
) -> str:
    """
    Generate HTML/CSS for a beautiful galaxy-like orbital animation with multiple rings and varied stars

    Args:
        center_dot_size: Size of the central star/black hole in pixels
        container_size: Size of the animation container in pixels
        animation_duration: Duration of one complete galaxy rotation in seconds
        enable_3d_depth: Whether to enable 3D depth effects (elliptical orbit, scaling, opacity)

    Returns:
        HTML string with embedded CSS for the galaxy animation
    """
    # Define galaxy rings with different properties
    galaxy_rings = [
        {"radius": 60, "dots": 2, "size": 8, "speed": 1.0, "color": "#ffffff"},
        {"radius": 65, "dots": 3, "size": 6, "speed": 0.8, "color": "#e3f2fd"},
        {"radius": 70, "dots": 2, "size": 4, "speed": 0.6, "color": "#bbdefb"},
        {"radius": 75, "dots": 1, "size": 3, "speed": 0.4, "color": "#90caf9"},
    ]

    # Generate stars for all rings
    all_stars_html = ""
    star_id = 0

    for ring in galaxy_rings:
        for i in range(ring["dots"]):
            angle_offset = (360 / ring["dots"]) * i
            # Add some spiral offset for galaxy arms
            spiral_offset = (ring["radius"] - 25) * 1.5  # More spiral for outer rings
            total_angle = angle_offset + spiral_offset

            delay = (animation_duration * ring["speed"] / ring["dots"]) * i
            individual_duration = animation_duration / ring["speed"]

            all_stars_html += f'''
            <div class="galaxy-star galaxy-star-{star_id}" style="
                width: {ring['size']}px;
                height: {ring['size']}px;
                margin-top: -{ring['size']//2}px;
                margin-left: -{ring['size']//2}px;
                background: radial-gradient(circle, {ring['color']} 0%, transparent 70%);
                box-shadow: 0 0 2px {ring['color']}, 0 0 4px {ring['color']};
                animation: galaxy-orbit-{star_id} {individual_duration}s linear infinite, twinkle {2 + (star_id % 3)}s ease-in-out infinite;
                animation-delay: -{delay}s, -{(star_id * 0.3) % 2}s;
            "></div>'''
            star_id += 1

    # Generate individual keyframes for each star
    individual_keyframes = ""
    star_id = 0

    for ring in galaxy_rings:
        for i in range(ring["dots"]):
            angle_offset = (360 / ring["dots"]) * i
            spiral_offset = (ring["radius"] - 25) * 1.5
            total_angle = angle_offset + spiral_offset

            if enable_3d_depth:
                individual_keyframes += f"""
    @keyframes galaxy-orbit-{star_id} {{
        0% {{
            transform:
                rotate({total_angle}deg)
                translateX({ring['radius']}px)
                scaleY(0.3)
                scale(0.6)
                rotate({-total_angle}deg);
            opacity: 0.3;
            z-index: 1;
        }}
        25% {{
            transform:
                rotate({total_angle + 90}deg)
                translateX({ring['radius']}px)
                scaleY(0.7)
                scale(1.2)
                rotate({-(total_angle + 90)}deg);
            opacity: 1;
            z-index: 3;
        }}
        50% {{
            transform:
                rotate({total_angle + 180}deg)
                translateX({ring['radius']}px)
                scaleY(0.3)
                scale(0.6)
                rotate({-(total_angle + 180)}deg);
            opacity: 0.3;
            z-index: 1;
        }}
        75% {{
            transform:
                rotate({total_angle + 270}deg)
                translateX({ring['radius']}px)
                scaleY(0.7)
                scale(1.2)
                rotate({-(total_angle + 270)}deg);
            opacity: 1;
            z-index: 3;
        }}
        100% {{
            transform:
                rotate({total_angle + 360}deg)
                translateX({ring['radius']}px)
                scaleY(0.3)
                scale(0.6)
                rotate({-(total_angle + 360)}deg);
            opacity: 0.3;
            z-index: 1;
        }}
    }}
                """
            else:
                individual_keyframes += f"""
    @keyframes galaxy-orbit-{star_id} {{
        0% {{
            transform: rotate({total_angle}deg) translateX({ring['radius']}px) rotate({-total_angle}deg);
        }}
        100% {{
            transform: rotate({total_angle + 360}deg) translateX({ring['radius']}px) rotate({-(total_angle + 360)}deg);
        }}
    }}
                """
            star_id += 1

    # Base keyframes for galactic core and twinkle effects
    base_keyframes = (
        f"""
    @keyframes twinkle {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}

         @keyframes galactic-pulse {{
         0%, 100% {{
             box-shadow:
                 0 0 15px #008471,
                 0 0 30px #006b5c,
                 0 0 45px #76B900,
                 0 0 60px rgba(118, 185, 0, 0.3);
         }}
         50% {{
             box-shadow:
                 0 0 8px #008471,
                 0 0 16px #006b5c,
                 0 0 24px #76B900,
                 0 0 35px rgba(118, 185, 0, 0.2);
         }}
     }}
    """
        if enable_3d_depth
        else """
    @keyframes twinkle {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}
    """
    )

    all_keyframes = individual_keyframes + base_keyframes

    center_glow = (
        """
        background: radial-gradient(circle,
            #00a085 0%,
            #008471 25%,
            #006b5c 45%,
            #005947 65%,
            #4a8c2a 80%,
            #76B900 100%);
        box-shadow:
            0 0 15px #008471,
            0 0 30px #006b5c,
            0 0 45px #76B900,
            0 0 60px rgba(118, 185, 0, 0.3);
        animation: galactic-pulse 3s ease-in-out infinite;
    """
        if enable_3d_depth
        else """background: radial-gradient(circle,
        #00a085 0%,
        #008471 25%,
        #006b5c 45%,
        #005947 65%,
        #4a8c2a 80%,
        #76B900 100%);"""
    )

    return f"""
    <style>
    :root {{
        --center-dot-size: {center_dot_size}px;
        --animation-duration: {animation_duration}s;
        --container-size: {container_size}px;
    }}

    {all_keyframes}

    .galaxy-container {{
        position: relative;
        width: var(--container-size);
        height: var(--container-size);
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 10px auto;
        perspective: 300px;
        background: transparent;
        border-radius: 50%;
        overflow: visible;
    }}

    .galactic-core {{
        position: absolute;
        width: var(--center-dot-size);
        height: var(--center-dot-size);
        {center_glow}
        border-radius: 50%;
        z-index: 10;
    }}

    .galaxy-arms {{
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
    }}

    .galaxy-star {{
        position: absolute;
        border-radius: 50%;
        top: 50%;
        left: 50%;
        transform-origin: 0 0;
    }}
    </style>

    <div class="galaxy-container">
        <div class="galactic-core"></div>
        <div class="galaxy-arms">
            {all_stars_html}
        </div>
    </div>
    """
