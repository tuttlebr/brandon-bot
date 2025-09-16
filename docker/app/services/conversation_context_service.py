"""
Conversation Context Service

This service handles automatic injection of conversation context
into LLM messages to ensure the model always has access to recent
conversation history up to the configured maximum turns.

Caching Implementation:
- Context summaries are cached to avoid regenerating on every message
- Cache key is based on recent message content (SHA256 hash)
- Cache entries expire after 5 minutes (configurable via _cache_ttl)
- Cache is invalidated if conversation grows by more than 2 messages
- Maximum 10 cache entries are kept to prevent memory issues
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from utils.config import config

from utils.logging_config import get_logger

logger = get_logger(__name__)


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
        # Cache TTL in seconds (5 minutes)
        self._cache_ttl = 300
        # Number of messages to consider for cache invalidation
        self._cache_invalidation_threshold = 2

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
            msg
            for msg in messages
            if msg.get("role") not in ["system", "tool"]
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
                "Conversation context injection not needed - insufficient "
                "messages"
            )
            return messages

        # Check if context has already been injected (avoid duplicates)
        for msg in messages:
            if msg.get(
                "role"
            ) == "system" and "## Conversation Context" in msg.get(
                "content", ""
            ):
                logger.debug(
                    "Conversation context already present, skipping injection"
                )
                return messages

        logger.info("Injecting conversation context for LLM invocation")

        # Get the conversation context (with caching)
        context_summary = self._get_conversation_summary(
            messages, user_message
        )

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
            "Injected conversation context summary (%d chars)",
            len(context_summary),
        )
        return enhanced_messages

    def _generate_cache_key(
        self, messages: List[Dict[str, Any]], user_message: str
    ) -> str:
        """
        Generate a cache key based on conversation messages

        Args:
            messages: Conversation messages
            user_message: Current user message

        Returns:
            SHA256 hash as cache key
        """
        # Filter conversation messages (exclude system/tool messages)
        conversation_messages = [
            msg
            for msg in messages
            if msg.get("role") not in ["system", "tool"]
        ]

        # Take last N messages for cache key (to limit key variation)
        # Using more messages than invalidation threshold to ensure stability
        cache_key_size = self._cache_invalidation_threshold * 4
        key_messages = conversation_messages[-cache_key_size:]

        # Create a string representation of messages
        key_parts = []
        for msg in key_messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))[:200]  # Limit content length
            key_parts.append(f"{role}:{content}")

        # Add current user message
        key_parts.append(f"current:{user_message[:200]}")

        # Generate hash
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _is_cache_valid(
        self, cache_entry: Dict[str, Any], messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if cache entry is still valid

        Args:
            cache_entry: Cache entry with timestamp and message count
            messages: Current messages

        Returns:
            True if cache is valid
        """
        # Check TTL
        if time.time() - cache_entry["timestamp"] > self._cache_ttl:
            logger.debug("Cache expired due to TTL")
            return False

        # Check if conversation has grown significantly
        current_msg_count = len(
            [
                msg
                for msg in messages
                if msg.get("role") not in ["system", "tool"]
            ]
        )
        cached_msg_count = cache_entry.get("message_count", 0)

        if (
            current_msg_count - cached_msg_count
            > self._cache_invalidation_threshold
        ):
            logger.debug(
                "Cache invalidated: conversation grew by %d messages",
                current_msg_count - cached_msg_count,
            )
            return False

        return True

    def _get_conversation_summary(
        self, messages: List[Dict[str, Any]], user_message: str
    ) -> Optional[str]:
        """
        Generate a conversation summary using the conversation context tool
        with caching support

        Args:
            messages: Conversation messages
            user_message: Current user message

        Returns:
            Conversation summary or None if failed
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(messages, user_message)

            # Check cache
            if cache_key in self._context_cache:
                cache_entry = self._context_cache[cache_key]
                if self._is_cache_valid(cache_entry, messages):
                    logger.info(
                        "Using cached conversation context (age: %ds)",
                        int(time.time() - cache_entry["timestamp"]),
                    )
                    return cache_entry["summary"]
                else:
                    # Remove invalid cache entry
                    del self._context_cache[cache_key]

            logger.info("Generating new conversation context summary")

            # Prepare messages for context analysis (limit to max turns)
            max_turns = config.llm.SLIDING_WINDOW_MAX_TURNS

            # Filter conversation messages (exclude system/tool messages)
            conversation_messages = [
                msg
                for msg in messages
                if msg.get("role") not in ["system", "tool"]
            ]

            # Apply max turns limit (convert turns to messages: 1 turn = 2
            # messages)
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
                "include_document_content": False,  # We handle documents
                # separately
            }

            # Execute the context analysis using the tool registry
            from tools.registry import execute_tool

            response = execute_tool("conversation_context", params)

            if response and response.success:
                summary = response.analysis

                # Cache the result
                self._context_cache[cache_key] = {
                    "summary": summary,
                    "timestamp": time.time(),
                    "message_count": len(conversation_messages),
                }

                # Clean up old cache entries (keep max 10 entries)
                if len(self._context_cache) > 10:
                    # Remove oldest entries
                    sorted_keys = sorted(
                        self._context_cache.keys(),
                        key=lambda k: self._context_cache[k]["timestamp"],
                    )
                    for old_key in sorted_keys[:-10]:
                        del self._context_cache[old_key]

                logger.info("Cached new conversation context summary")
                return summary
            else:
                logger.warning("Context analysis failed")
                return None

        except Exception as e:
            logger.error("Error generating conversation context: %s", e)
            return None

    def _create_context_system_message(
        self, context_summary: str, total_messages: int
    ) -> Dict[str, str]:
        """
        Create a system message with conversation context

        Args:
            context_summary: Generated conversation summary
            total_messages: Total number of messages in conversation

        Returns:
            System message with context
        """
        return {
            "role": "system",
            "content": (
                f"""## Conversation Context

You are continuing an ongoing conversation. Here's a summary of the
discussion so far:

{context_summary}

Total messages in conversation: {total_messages}

Please maintain continuity with the previous discussion and refer to
earlier topics when relevant."""
            ),
        }

    def clear_cache(self):
        """Clear the conversation context cache"""
        self._context_cache.clear()
        logger.info("Cleared conversation context cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging/monitoring"""
        return {
            "cache_size": len(self._context_cache),
            "cache_ttl": self._cache_ttl,
            "invalidation_threshold": self._cache_invalidation_threshold,
            "entries": [
                {
                    "key": key[:8] + "...",  # Show first 8 chars of hash
                    "age": int(time.time() - entry["timestamp"]),
                    "message_count": entry["message_count"],
                }
                for key, entry in self._context_cache.items()
            ],
        }
