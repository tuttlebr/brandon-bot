"""
Translation Service

This service handles language translation operations.
Extracted from the monolithic AssistantTool.
"""

import logging
from typing import Dict, List, Optional

from models.chat_config import ChatConfig
from services.llm_client_service import llm_client_service
from utils.config import config as app_config

logger = logging.getLogger(__name__)

# Supported languages for translation
SUPPORTED_LANGUAGES = [
    "English",
    "German",
    "French",
    "Italian",
    "Portuguese",
    "Hindi",
    "Spanish",
    "Thai",
]


class TranslationService:
    """Service for language translation operations"""

    def __init__(self, config: ChatConfig, llm_type: str = "llm"):
        """
        Initialize translation service

        Args:
            config: Chat configuration
            llm_type: Type of LLM to use
        """
        self.config = config
        self.llm_type = llm_type
        self.supported_languages = SUPPORTED_LANGUAGES

    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict[str, any]:
        """
        Translate text between languages

        Args:
            text: Text to translate
            target_language: Target language
            source_language: Source language (optional, will auto-detect)
            messages: Optional conversation messages for context

        Returns:
            Translation result dictionary
        """
        # Validate languages
        if target_language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Target language '{target_language}' not supported. Supported: {', '.join(self.supported_languages)}",
            }

        if source_language and source_language not in self.supported_languages:
            return {
                "success": False,
                "error": f"Source language '{source_language}' not supported. Supported: {', '.join(self.supported_languages)}",
            }

        try:
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)
            system_prompt = self._get_translation_prompt(
                source_language, target_language
            )

            # Build messages
            if messages:
                final_messages = self._build_messages_with_context(
                    messages, system_prompt, text
                )
            else:
                final_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ]

            logger.debug(
                f"Translating text to {target_language} using {model_name}"
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            translated_text = response.choices[0].message.content.strip()

            return {
                "success": True,
                "result": translated_text,
                "source_language": source_language or "auto-detected",
                "target_language": target_language,
                "processing_notes": f"Translation completed from {source_language or 'auto-detected'} to {target_language}",
            }

        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return {"success": False, "error": str(e)}

    def _get_translation_prompt(
        self, source_language: Optional[str], target_language: str
    ) -> str:
        """Get translation system prompt"""
        from utils.system_prompt import get_context_system_prompt

        # Use the new context-aware system prompt
        return get_context_system_prompt(
            context='translation',
            target_language=target_language,
            source_language=source_language,
        )

    def _build_messages_with_context(
        self, messages: List[Dict], system_prompt: str, text: str
    ) -> List[Dict]:
        """Build messages with proper context injection"""

        # Filter out existing translation system messages
        filtered_messages = [
            msg
            for msg in messages
            if msg.get("role") != "system"
            or "translating" not in msg.get("content", "").lower()
        ]

        return [
            {"role": "system", "content": system_prompt}
        ] + filtered_messages

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()
