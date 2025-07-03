"""
Text Processor Service

This service handles text processing operations like summarizing, proofreading,
rewriting, and critiquing text. Extracted from the monolithic AssistantTool.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional

from models.chat_config import ChatConfig
from services.llm_client_service import llm_client_service
from utils.config import config as app_config

logger = logging.getLogger(__name__)


class TextTaskType(str, Enum):
    """Text processing task types"""

    SUMMARIZE = "summarize"
    PROOFREAD = "proofread"
    REWRITE = "rewrite"
    CRITIC = "critic"


class TextProcessorService:
    """Service for text processing operations"""

    def __init__(self, config: ChatConfig, llm_type: str = "intelligent"):
        """
        Initialize text processor service

        Args:
            config: Chat configuration
            llm_type: Type of LLM to use ("fast", "llm", "intelligent")
        """
        self.config = config
        self.llm_type = llm_type

    def process_text(
        self,
        task_type: TextTaskType,
        text: str,
        instructions: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
    ) -> Dict[str, any]:
        """
        Process text with specified task type

        Args:
            task_type: Type of processing task
            text: Text to process
            instructions: Optional additional instructions
            messages: Optional conversation messages for context

        Returns:
            Processing result dictionary
        """
        try:
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)
            system_prompt = self._get_system_prompt(task_type, instructions)

            # Build messages
            if messages:
                final_messages = self._build_messages_with_context(messages, system_prompt, text)
            else:
                final_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]

            logger.debug(f"Processing text with {task_type} using {model_name}")

            response = client.chat.completions.create(
                model=model_name,
                messages=final_messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            return {
                "success": True,
                "result": result,
                "task_type": task_type,
                "processing_notes": self._get_processing_notes(task_type, text, result),
            }

        except Exception as e:
            logger.error(f"Error processing text with {task_type}: {e}")
            return {"success": False, "error": str(e), "task_type": task_type}

    def _get_system_prompt(self, task_type: TextTaskType, instructions: Optional[str] = None) -> str:
        """Get appropriate system prompt for task type"""

        base_prompts = {
            TextTaskType.SUMMARIZE: """You are creating a concise summary that captures the essential information.

Focus on extracting the most important points that deliver the core insights. Maintain the logical flow and causal relationships from the original. Eliminate redundancy while preserving all critical information.""",
            TextTaskType.PROOFREAD: """You are proofreading text to identify and correct errors.

Check for grammar, punctuation, and spelling mistakes. Improve awkward phrasing and ensure consistency in style. Mark significant changes with [**] brackets and provide brief explanations for important corrections.""",
            TextTaskType.REWRITE: """You are rewriting text to improve its effectiveness and impact.

Enhance clarity, flow, and engagement while preserving the core message. Adjust tone, formality, and style as needed for the target audience. Use stronger verbs and more precise language.""",
            TextTaskType.CRITIC: """You are providing constructive critique and actionable feedback.

Evaluate the text's strengths and weaknesses objectively. Identify areas for improvement in structure, argumentation, and execution. Provide specific, actionable suggestions for enhancement.""",
        }

        prompt = base_prompts.get(task_type, base_prompts[TextTaskType.SUMMARIZE])

        if instructions:
            prompt += f"\n\nAdditional instructions: {instructions}"

        return prompt

    def _build_messages_with_context(self, messages: List[Dict], system_prompt: str, text: str) -> List[Dict]:
        """Build messages with proper context injection"""

        # Filter out existing system messages with task prompts
        filtered_messages = [
            msg
            for msg in messages
            if msg.get("role") != "system"
            or not any(task.value in msg.get("content", "").lower() for task in TextTaskType)
        ]

        # Check for injected context in text
        if "--- Additional Context ---" in text:
            parts = text.split("--- Additional Context ---")
            if len(parts) == 2:
                user_text = parts[0].strip()
                context = parts[1].strip()

                return [
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": f"Additional context:\n\n{context}"},
                ] + filtered_messages

        return [{"role": "system", "content": system_prompt}] + filtered_messages

    def _get_processing_notes(self, task_type: TextTaskType, original: str, result: str) -> str:
        """Generate processing notes for the task"""

        if task_type == TextTaskType.SUMMARIZE:
            return f"Original: {len(original.split())} words, Summary: {len(result.split())} words"
        elif task_type == TextTaskType.PROOFREAD:
            return "Proofreading completed with suggestions for improvement"
        elif task_type == TextTaskType.REWRITE:
            return "Text has been rewritten for improved clarity and flow"
        elif task_type == TextTaskType.CRITIC:
            return "Critical analysis and feedback provided"

        return "Processing completed"
