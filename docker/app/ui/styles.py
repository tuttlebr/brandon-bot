import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit application"""
    st.markdown(
        """
    <style>
        /* Hide Streamlit decoration */
        header {visibility: hidden;}

        /* Global refinements */
        .stApp {
            background-color: rgba(24, 25, 26, 0.98);
        }

        /* Smooth text streaming animation - silky like butter */
        .element-container .stMarkdown p {
            animation: textFadeIn 0.8s cubic-bezier(0.22, 0.61, 0.36, 1);
            transition: all 0.5s cubic-bezier(0.22, 0.61, 0.36, 1);
            letter-spacing: 0.01em;
            line-height: 1.6em;
        }

        /* Text within chat messages */
        .stChatMessage .stMarkdown p {
            font-size: 1.02em;
            line-height: 1.65;
            margin-bottom: 0.9em;
            text-align: left;
            opacity: 0.98;
            font-weight: 400;
        }

        /* Code blocks in chat */
        .stChatMessage code {
            border-radius: 4px !important;
            padding: 2px 5px !important;
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
            background-color: rgba(40, 42, 46, 0.95) !important;
        }

        /* Inline code */
        .stChatMessage p code {
            background-color: rgba(40, 42, 46, 0.7) !important;
            color: #e0e0e0 !important;
        }

        /* Block code containers */
        .stChatMessage pre {
            background-color: rgba(40, 42, 46, 0.95) !important;
            border-radius: 8px !important;
            margin: 10px 0 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        }

        .stChatMessage pre code {
            padding: 15px !important;
            background-color: transparent !important; /* Remove background from code inside pre */
            border-radius: 0 !important; /* Remove border radius from inner code */
            box-shadow: none !important;
            display: block;
            overflow-x: auto;
            color: #e0e0e0 !important;
        }

        /* Fix for markdown rendered code blocks */
        .stMarkdown pre {
            background-color: rgba(40, 42, 46, 0.95) !important;
            border-radius: 8px !important;
            padding: 0 !important;
            margin: 10px 0 !important;
        }

        /* Refined fade-in animation for text */
        @keyframes textFadeIn {
            0% {
                opacity: 0.01;
                transform: translateY(1px);
                filter: blur(0.4px);
            }
            40% {
                opacity: 0.5;
                filter: blur(0.2px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
                filter: blur(0);
            }
        }

        /* Enhanced typing indicator animation */
        .typing-animation {
            display: inline-flex;
            align-items: center;
            background-color: rgba(64, 62, 65, 0.8);
            padding: 12px 20px;
            border-radius: 18px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            transform-origin: left center;
            animation: pulseIn 0.3s cubic-bezier(0.22, 0.61, 0.36, 1);
        }

        @keyframes pulseIn {
            0% { transform: scale(0.95); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .typing-animation span {
            display: inline-block;
            width: 7px;
            height: 7px;
            margin: 0 3px;
            background-color: rgba(118, 185, 0, 0.7);
            border-radius: 50%;
            animation: typingBounce 1.4s infinite both;
            animation-delay: calc(0.25s * var(--i));
        }

        @keyframes typingBounce {
            0%, 80%, 100% {
                transform: translateY(0);
                opacity: 0.3;
            }
            40% {
                transform: translateY(-4px);
                opacity: 1;
            }
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(30, 30, 30, 0.1);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(118, 185, 0, 0.3);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(118, 185, 0, 0.5);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def get_typing_indicator_html() -> str:
    """Get the HTML for the typing indicator animation"""
    return """
    <div class="typing-animation">
        <span style="--i:1"></span>
        <span style="--i:2"></span>
        <span style="--i:3"></span>
    </div>
    """
