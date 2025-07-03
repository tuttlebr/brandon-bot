import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styling to the Streamlit application"""
    # st.markdown(
    #     """
    # <style>
    #     /* Hide Streamlit decoration */
    #     header {visibility: hidden;}

    #     /* Global refinements with smooth transitions */
    #     .stApp {
    #         background-color: rgba(24, 25, 26, 0.98);
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     /* Smooth page transition for st.rerun() */
    #     html, body, .stApp > div {
    #         transition: opacity 0.2s ease-in-out, transform 0.2s ease-in-out;
    #     }

    #     /* Fade-in animation for the entire app content */
    #     .main .block-container {
    #         animation: pageTransition 0.25s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     @keyframes pageTransition {
    #         0% {
    #             opacity: 0.7;
    #             transform: translateY(2px);
    #         }
    #         100% {
    #             opacity: 1;
    #             transform: translateY(0);
    #         }
    #     }

    #     /* Smooth chat message container transitions */
    #     .stChatMessage {
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #         animation: chatMessageSlideIn 0.4s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     @keyframes chatMessageSlideIn {
    #         0% {
    #             opacity: 0;
    #             transform: translateY(8px) scale(0.98);
    #         }
    #         60% {
    #             opacity: 0.8;
    #             transform: translateY(2px) scale(0.99);
    #         }
    #         100% {
    #             opacity: 1;
    #             transform: translateY(0) scale(1);
    #         }
    #     }

    #     /* Enhanced text streaming animation - silky like butter */
    #     .element-container .stMarkdown p {
    #         animation: textFadeIn 0.6s cubic-bezier(0.22, 0.61, 0.36, 1);
    #         transition: all 0.4s cubic-bezier(0.22, 0.61, 0.36, 1);
    #         letter-spacing: 0.01em;
    #         line-height: 1.6em;
    #     }

    #     /* Text within chat messages with smooth appearance */
    #     .stChatMessage .stMarkdown p {
    #         font-size: 1.02em;
    #         line-height: 1.65;
    #         margin-bottom: 0.9em;
    #         text-align: left;
    #         opacity: 0.98;
    #         font-weight: 400;
    #         transition: opacity 0.3s ease-in-out;
    #     }

    #     /* Smooth transitions for chat input area */
    #     .stChatInput {
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     /* Smooth container transitions */
    #     .element-container {
    #         transition: opacity 0.2s ease-in-out, transform 0.2s ease-in-out;
    #     }

    #     /* Code blocks in chat with smooth transitions */
    #     .stChatMessage code {
    #         border-radius: 4px !important;
    #         padding: 2px 5px !important;
    #         font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    #         background-color: rgba(40, 42, 46, 0.95) !important;
    #         transition: background-color 0.2s ease-in-out;
    #     }

    #     /* Inline code */
    #     .stChatMessage p code {
    #         background-color: rgba(40, 42, 46, 0.7) !important;
    #         color: #e0e0e0 !important;
    #         transition: all 0.2s ease-in-out;
    #     }

    #     /* Block code containers */
    #     .stChatMessage pre {
    #         background-color: rgba(40, 42, 46, 0.95) !important;
    #         border-radius: 8px !important;
    #         margin: 10px 0 !important;
    #         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     .stChatMessage pre code {
    #         padding: 15px !important;
    #         background-color: transparent !important; /* Remove background from code inside pre */
    #         border-radius: 0 !important; /* Remove border radius from inner code */
    #         box-shadow: none !important;
    #         display: block;
    #         overflow-x: auto;
    #         color: #e0e0e0 !important;
    #     }

    #     /* Fix for markdown rendered code blocks */
    #     .stMarkdown pre {
    #         background-color: rgba(40, 42, 46, 0.95) !important;
    #         border-radius: 8px !important;
    #         padding: 0 !important;
    #         margin: 10px 0 !important;
    #         transition: all 0.2s ease-in-out;
    #     }

    #     /* Refined fade-in animation for text */
    #     @keyframes textFadeIn {
    #         0% {
    #             opacity: 0.01;
    #             transform: translateY(1px);
    #             filter: blur(0.4px);
    #         }
    #         40% {
    #             opacity: 0.5;
    #             filter: blur(0.2px);
    #         }
    #         100% {
    #             opacity: 1;
    #             transform: translateY(0);
    #             filter: blur(0);
    #         }
    #     }

    #     /* Enhanced typing indicator animation with smooth transitions */
    #     .typing-animation {
    #         display: inline-flex;
    #         align-items: center;
    #         background-color: rgba(64, 62, 65, 0.8);
    #         padding: 12px 20px;
    #         border-radius: 18px;
    #         box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    #         margin: 10px 0;
    #         transform-origin: left center;
    #         animation: pulseIn 0.3s cubic-bezier(0.22, 0.61, 0.36, 1);
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     @keyframes pulseIn {
    #         0% { transform: scale(0.95); opacity: 0; }
    #         100% { transform: scale(1); opacity: 1; }
    #     }

    #     .typing-animation span {
    #         display: inline-block;
    #         width: 7px;
    #         height: 7px;
    #         margin: 0 3px;
    #         background-color: rgba(118, 185, 0, 0.7);
    #         border-radius: 50%;
    #         animation: typingBounce 1.4s infinite both;
    #         animation-delay: calc(0.25s * var(--i));
    #         transition: background-color 0.2s ease-in-out;
    #     }

    #     @keyframes typingBounce {
    #         0%, 80%, 100% {
    #             transform: translateY(0);
    #             opacity: 0.3;
    #         }
    #         40% {
    #             transform: translateY(-4px);
    #             opacity: 1;
    #         }
    #     }

    #     /* Smooth spinner transitions */
    #     .stSpinner {
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     /* Smooth expander transitions */
    #     .streamlit-expander {
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     /* Smooth button transitions */
    #     .stButton button {
    #         transition: all 0.2s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     .stButton button:hover {
    #         transform: translateY(-1px);
    #         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    #     }

    #     /* Scrollbar styling with smooth transitions */
    #     ::-webkit-scrollbar {
    #         width: 6px;
    #         height: 6px;
    #         transition: all 0.2s ease-in-out;
    #     }

    #     ::-webkit-scrollbar-track {
    #         background: rgba(30, 30, 30, 0.1);
    #         border-radius: 10px;
    #     }

    #     ::-webkit-scrollbar-thumb {
    #         background: rgba(118, 185, 0, 0.3);
    #         border-radius: 10px;
    #         transition: background 0.2s ease-in-out;
    #     }

    #     ::-webkit-scrollbar-thumb:hover {
    #         background: rgba(118, 185, 0, 0.5);
    #     }

    #     /* Reduce flicker on rerun by smoothing all transitions */
    #     * {
    #         -webkit-font-smoothing: antialiased;
    #         -moz-osx-font-smoothing: grayscale;
    #     }

    #     /* Smooth transition for entire document */
    #     html {
    #         scroll-behavior: smooth;
    #     }

    #     /* Styled sidebar with custom colors and spacing */
    #     section[data-testid="stSidebar"] {
    #         background: linear-gradient(180deg, rgba(30, 31, 32, 0.98) 0%, rgba(24, 25, 26, 0.98) 100%);
    #         width: 300px;
    #         padding: 25px;
    #         box-shadow: 2px 0 10px rgba(0, 0, 0, 0.2);
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #         border-right: 1px solid rgba(118, 185, 0, 0.1);
    #     }

    #     /* Style widgets inside the sidebar */
    #     section[data-testid="stSidebar"] .element-container {
    #         margin-bottom: 20px;
    #         transition: all 0.2s ease-in-out;
    #     }

    #     /* Persistent sidebar toggle button styling - UPDATED */
    #     /* Target both open and closed states of the toggle button */
    #     button[data-testid="collapsedControl"],
    #     button[data-testid="baseButton-header"] {
    #         position: fixed !important;
    #         top: 20px !important;
    #         left: 20px !important;
    #         z-index: 999999 !important;
    #         background: rgba(40, 42, 46, 0.95) !important;
    #         border: 1px solid rgba(118, 185, 0, 0.3) !important;
    #         border-radius: 8px !important;
    #         padding: 8px 12px !important;
    #         color: #e0e0e0 !important;
    #         transition: all 0.2s cubic-bezier(0.4, 0.0, 0.2, 1) !important;
    #         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    #         opacity: 1 !important;
    #         visibility: visible !important;
    #         display: flex !important;
    #         align-items: center !important;
    #         justify-content: center !important;
    #         min-width: 40px !important;
    #         min-height: 40px !important;
    #     }

    #     /* Ensure toggle stays visible in all states */
    #     div[data-testid="collapsedControl"] button,
    #     header[data-testid="stHeader"] button[kind="header"] {
    #         position: fixed !important;
    #         top: 20px !important;
    #         left: 20px !important;
    #         z-index: 999999 !important;
    #         opacity: 1 !important;
    #         visibility: visible !important;
    #     }

    #     /* Hover effect for toggle button */
    #     button[data-testid="collapsedControl"]:hover,
    #     button[data-testid="baseButton-header"]:hover,
    #     div[data-testid="collapsedControl"] button:hover,
    #     header[data-testid="stHeader"] button[kind="header"]:hover {
    #         background: rgba(118, 185, 0, 0.2) !important;
    #         border-color: rgba(118, 185, 0, 0.5) !important;
    #         transform: translateY(-1px) !important;
    #         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    #         opacity: 1 !important;
    #     }

    #     /* Active/pressed state */
    #     button[data-testid="collapsedControl"]:active,
    #     button[data-testid="baseButton-header"]:active {
    #         transform: translateY(0) !important;
    #         box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
    #     }

    #     /* Adjust main content area when sidebar is open */
    #     section[data-testid="stSidebar"] + div {
    #         transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     /* Make sure main content has proper padding to not overlap with toggle */
    #     .main > div:first-child {
    #         padding-left: 70px !important;
    #     }

    #     /* Ensure the header doesn't hide the toggle button */
    #     header[data-testid="stHeader"] {
    #         z-index: 999998 !important; /* Just below the toggle button */
    #     }

    #     /* Sidebar header styling */
    #     section[data-testid="stSidebar"] h1,
    #     section[data-testid="stSidebar"] h2,
    #     section[data-testid="stSidebar"] h3 {
    #         color: #e0e0e0 !important;
    #         margin-bottom: 15px;
    #     }

    #     /* Sidebar text styling */
    #     section[data-testid="stSidebar"] p,
    #     section[data-testid="stSidebar"] span,
    #     section[data-testid="stSidebar"] label {
    #         color: rgba(224, 224, 224, 0.9) !important;
    #     }

    #     /* Sidebar input fields */
    #     section[data-testid="stSidebar"] input,
    #     section[data-testid="stSidebar"] textarea,
    #     section[data-testid="stSidebar"] select {
    #         background-color: rgba(40, 42, 46, 0.5) !important;
    #         border: 1px solid rgba(118, 185, 0, 0.2) !important;
    #         color: #e0e0e0 !important;
    #         transition: all 0.2s ease-in-out;
    #     }

    #     section[data-testid="stSidebar"] input:focus,
    #     section[data-testid="stSidebar"] textarea:focus,
    #     section[data-testid="stSidebar"] select:focus {
    #         border-color: rgba(118, 185, 0, 0.5) !important;
    #         box-shadow: 0 0 0 2px rgba(118, 185, 0, 0.1) !important;
    #     }

    #     /* Sidebar buttons */
    #     section[data-testid="stSidebar"] button {
    #         background-color: rgba(118, 185, 0, 0.15) !important;
    #         border: 1px solid rgba(118, 185, 0, 0.3) !important;
    #         color: #e0e0e0 !important;
    #         transition: all 0.2s cubic-bezier(0.4, 0.0, 0.2, 1);
    #     }

    #     section[data-testid="stSidebar"] button:hover {
    #         background-color: rgba(118, 185, 0, 0.25) !important;
    #         border-color: rgba(118, 185, 0, 0.5) !important;
    #         transform: translateY(-1px);
    #     }

    #     /* Additional styles to ensure toggle button persistence */
    #     /* Target Streamlit's header buttons specifically */
    #     header button[kind="header"] {
    #         position: fixed !important;
    #         top: 20px !important;
    #         left: 20px !important;
    #         z-index: 999999 !important;
    #         opacity: 1 !important;
    #         visibility: visible !important;
    #     }

    #     /* Override any Streamlit styles that might hide the button */
    #     [data-testid="stDecoration"] {
    #         display: none !important;
    #     }
    # </style>
    # """,
    #     unsafe_allow_html=True,
    # )
