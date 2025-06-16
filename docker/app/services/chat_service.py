import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from models.chat_config import ChatConfig
from models.chat_message import ChatMessage
from utils.retrieval import EmbeddingCreator, SimilaritySearch
from utils.split_context import END_CONTEXT, START_CONTEXT, extract_context_regex


class ChatService:
    """Service for handling chat processing operations"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the chat service

        Args:
            config: Configuration for the chat service
        """
        self.config = config
        self.embedding_creation = self._initialize_embedding_creator()
        self.similarity_search = self._initialize_similarity_search()
        self.verbose_messages = []

    def _initialize_embedding_creator(self) -> Optional[EmbeddingCreator]:
        """Initialize the embedding creator"""
        try:
            return EmbeddingCreator(
                base_url=self.config.embedding_endpoint,
                api_key=self.config.api_key,
                model=self.config.embedding_model,
            )
        except Exception as e:
            logging.error(f"Failed to initialize embedding creator: {e}")
            logging.warning(f"Embedding-based search functionality may be limited: {e}")
            return None

    def _initialize_similarity_search(self) -> Optional[SimilaritySearch]:
        """Initialize the similarity search"""
        try:
            return SimilaritySearch(
                collection_name=self.config.collection_name,
                uri=self.config.database_url,
                db_name=self.config.default_db,
            )
        except Exception as e:
            logging.error(f"Failed to initialize similarity search: {e}")
            logging.warning(f"Search functionality may be limited: {e}")
            return None

    def clean_chat_history_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean up context information and thinking tags from all previous messages in chat history

        Args:
            messages: List of message dictionaries

        Returns:
            Cleaned list of message dictionaries
        """
        cleaned_messages = []

        for message in messages:
            if message["role"] != "system":  # Preserve system prompt as is
                # Check if this is an image message - if so, skip context cleaning
                if isinstance(message["content"], dict) and message["content"].get("type") == "image":
                    # Image messages don't need context cleaning
                    cleaned_messages.append(message)
                    continue

                # Only clean string content
                if isinstance(message["content"], str):
                    # First remove context markers
                    cleaned_content = extract_context_regex(message["content"])

                    # Then remove any thinking tags that might be present
                    cleaned_content = re.sub(r"<think>.*?</think>", "", cleaned_content, flags=re.DOTALL)

                    cleaned_messages.append({"role": message["role"], "content": cleaned_content})
                else:
                    cleaned_messages.append(message)
            else:
                cleaned_messages.append(message)

        logging.debug("Cleaned context and thinking tags from previous chat history")
        return cleaned_messages

    def enhance_prompt_with_context(self, prompt: str) -> Tuple[str, str]:
        """
        Enhance prompt with relevant context using vector search

        Args:
            prompt: The user's original prompt

        Returns:
            Tuple of (original prompt, context information)
        """

        # Skip retrieval if similarity search is not available
        if not self.embedding_creation or not self.similarity_search:
            return prompt, ""

        try:
            # Get vector embedding for the query
            embedding_data = self.embedding_creation.create_formatted_query(prompt)
            # Perform similarity search
            results_dict = self.similarity_search.search(data=embedding_data)

            # Apply reranking if multiple results found
            if results_dict and len(results_dict[0]) > 1:
                results_dict = self.similarity_search.reranker(prompt, results_dict)
                context = self.similarity_search.format_results(results_dict)
                return prompt, context

        except Exception as e:
            logging.error(f"Context retrieval error: {e}")

        return prompt, ""

    def prepare_messages_for_api(self, messages: List[Dict[str, Any]], context: str = "") -> List[Dict[str, Any]]:
        """
        Prepare messages for API call, filtering out image data for LLM

        Args:
            messages: List of message dictionaries
            context: Context information to add to the prompt

        Returns:
            Prepared messages for API call
        """
        # Include all messages but filter out image data for LLM
        self.verbose_messages = []

        for msg in messages:
            chat_message = ChatMessage(msg["role"], msg["content"])

            # For image messages, only include the text content, not the image data
            if chat_message.is_image_message():
                # Only include the text confirmation message, not the image data
                text_content = chat_message.get_display_content()
                self.verbose_messages.append({"role": msg["role"], "content": text_content})
            else:
                # Regular messages go through as normal
                self.verbose_messages.append({"role": msg["role"], "content": msg["content"]})

        # Add context if available
        if context and self.verbose_messages:
            self.verbose_messages[-1]["content"] += f"{START_CONTEXT}{context}{END_CONTEXT}"

        return self.verbose_messages

    def drop_verbose_messages_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean up context information from messages

        Args:
            messages: List of message dictionaries

        Returns:
            Cleaned list of message dictionaries
        """
        cleaned_messages = []

        for message in messages:
            # Handle different message content types
            if isinstance(message["content"], dict) and message["content"].get("type") == "image":
                # Image messages don't need context extraction
                cleaned_messages.append(
                    {"role": message["role"], "content": message["content"]}  # Keep image content as-is
                )
            else:
                # Regular text messages get context cleaned
                cleaned_messages.append(
                    {
                        "role": message["role"],
                        "content": ChatMessage(
                            message["role"], extract_context_regex(message["content"])
                        ).get_display_content(),
                    }
                )

        return cleaned_messages
