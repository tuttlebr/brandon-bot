"""
Conversation Context Service

This service handles automatic injection of conversation context
into LLM messages to ensure the model always has access to recent
conversation history up to the configured maximum turns.
"""

import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from utils.config import config

logger = logging.getLogger(__name__)


class ConversationContextService:
    """Service for automatically injecting conversation context"""

    def __init__(self, config_obj: ChatConfig):
        """
        Initialize the conversation context service

        Args:
            config_obj: Configuration object
        """
        self.config = config_obj
        self._context_cache = {}

    def should_inject_context(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Determine if conversation context should be injected

        Args:
            messages: Current conversation messages

        Returns:
            True if context should be injected
        """
        # Check if auto-injection is enabled
        if not config.llm.AUTO_INJECT_CONVERSATION_CONTEXT:
            return False

        # Always inject context if we have more than just system messages
        conversation_messages = [
            msg for msg in messages if msg.get("role") not in ["system", "tool"]
        ]

        # Calculate number of turns (2 messages = 1 turn)
        num_turns = len(conversation_messages) // 2

        # Inject context if we have at least the minimum required turns
        return num_turns >= config.llm.MIN_TURNS_FOR_CONTEXT_INJECTION

    def inject_conversation_context(
        self, messages: List[Dict[str, Any]], user_message: str
    ) -> List[Dict[str, Any]]:
        """
        Inject conversation context into messages for LLM processing

        Args:
            messages: Current conversation messages
            user_message: The user's current query

        Returns:
            Messages with conversation context injected
        """
        # Check if we should inject context
        if not self.should_inject_context(messages):
            logger.debug(
                "Conversation context injection not needed - insufficient messages"
            )
            return messages

        # Check if context has already been injected (avoid duplicates)
        for msg in messages:
            if msg.get("role") == "system" and "## Conversation Context" in msg.get(
                "content", ""
            ):
                logger.debug("Conversation context already present, skipping injection")
                return messages

        logger.info("Injecting conversation context for LLM invocation")

        # Get the conversation context
        context_summary = self._get_conversation_summary(messages, user_message)

        if not context_summary:
            logger.warning("Failed to generate conversation context")
            return messages

        # Create system message with conversation context
        context_system_message = self._create_context_system_message(
            context_summary, len(messages)
        )

        # Create a new message list with context injected
        enhanced_messages = []

        # First, add any existing system messages
        for msg in messages:
            if msg.get("role") == "system":
                enhanced_messages.append(msg)

        # Add our conversation context system message
        enhanced_messages.append(context_system_message)

        # Add all other messages
        for msg in messages:
            if msg.get("role") != "system":
                enhanced_messages.append(msg)

        logger.info(
            f"Injected conversation context summary ({len(context_summary)} chars)"
        )
        return enhanced_messages

    def _get_conversation_summary(
        self, messages: List[Dict[str, Any]], user_message: str
    ) -> Optional[str]:
        """
        Generate a conversation summary using the conversation context tool

        Args:
            messages: Conversation messages
            user_message: Current user message

        Returns:
            Conversation summary or None if failed
        """
        try:
            # Prepare messages for context analysis (limit to max turns)
            max_turns = config.llm.SLIDING_WINDOW_MAX_TURNS

            # Filter conversation messages (exclude system/tool messages)
            conversation_messages = [
                msg for msg in messages if msg.get("role") not in ["system", "tool"]
            ]

            # Apply max turns limit (convert turns to messages: 1 turn = 2 messages)
            max_messages = max_turns * 2
            if len(conversation_messages) > max_messages:
                limited_messages = conversation_messages[-max_messages:]
            else:
                limited_messages = conversation_messages

            # Use the conversation context tool to analyze
            params = {
                "query": "conversation_summary",
                "max_messages": len(limited_messages),
                "messages": limited_messages,
                "include_document_content": False,  # We handle documents separately
                "but_why": "Analyzing conversation history to provide relevant context for better response generation",
            }

            # Execute the context analysis using the tool registry
            from tools.registry import execute_tool

            response = execute_tool("conversation_context", params)

            if response and response.success:
                return response.analysis
            else:
                logger.error(
                    f"Context analysis failed: {response.error_message if response else 'Unknown error'}"
                )
                return None

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return None

    def _create_context_system_message(
        self, context_summary: str, total_messages: int
    ) -> Dict[str, str]:
        """
        Create a system message containing conversation context

        Args:
            context_summary: The conversation summary
            total_messages: Total number of messages in conversation

        Returns:
            System message with conversation context
        """
        content = f"""## Conversation Context
You are continuing an ongoing conversation. Here's a summary of the discussion so far:

{context_summary}

Total messages in conversation: {total_messages}
Context window: Last {config.llm.SLIDING_WINDOW_MAX_TURNS} turns

Please maintain continuity with the previous discussion and refer to earlier topics when relevant.
---"""

        return {"role": "system", "content": content}
