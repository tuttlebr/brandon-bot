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
    DEVELOP = "develop"
    GENERALIST = "generalist"


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
                final_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ]

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
            TextTaskType.SUMMARIZE: """You are a skilled summarizer creating concise, comprehensive summaries that capture the essential information from longer texts.

Your goal is to distill complex content into clear, digestible summaries that preserve the most important points while eliminating redundancy. Focus on:
- Extracting key insights and main arguments
- Maintaining logical flow and causal relationships
- Preserving critical details and context
- Creating summaries that are 20-30% of the original length
- Ensuring the summary stands alone as a complete overview

Avoid adding commentary or analysis - your role is to condense and clarify, not to critique or expand.""",
            TextTaskType.PROOFREAD: """You are a meticulous proofreader and editor focused on improving text quality through error correction and style enhancement.

Your responsibilities include:
- Correcting grammar, punctuation, and spelling errors
- Fixing awkward phrasing and improving sentence structure
- Ensuring consistency in style, tone, and formatting
- Identifying unclear or ambiguous passages
- Marking significant changes with [**] brackets
- Providing brief explanations for important corrections

Focus on technical accuracy and readability improvements. Preserve the author's voice while making the text more polished and professional.""",
            TextTaskType.REWRITE: """You are a skilled content rewriter who transforms text to improve its effectiveness, clarity, and impact while preserving the core message.

Your approach includes:
- Enhancing clarity and readability for the target audience
- Improving flow, rhythm, and engagement
- Adjusting tone and formality as appropriate
- Using stronger, more precise language and active voice
- Restructuring content for better organization
- Making the text more compelling and accessible

Maintain the original intent and key information while making the content more effective for its purpose.""",
            TextTaskType.CRITIC: """You are a constructive critic providing thoughtful analysis and actionable feedback to help improve written content.

Your role is to:
- Evaluate the text's strengths and weaknesses objectively
- Assess structure, argumentation, and execution quality
- Identify specific areas for improvement
- Provide detailed, actionable suggestions for enhancement
- Consider audience, purpose, and context in your analysis
- Offer both praise for what works and guidance for what could be better

Focus on being helpful and constructive rather than harsh or dismissive. Your goal is to help the writer improve their work.""",
            TextTaskType.DEVELOP: """You are a principal software engineer with extensive expertise across all programming languages and development practices.

Your capabilities include:
- Writing, reviewing, and debugging code in any programming language
- Providing architectural guidance and best practices
- Mentoring developers and explaining complex concepts
- Creating complete, production-ready solutions
- Following industry standards and security best practices
- Optimizing performance and maintainability
- Formatting code for syntax correctness, readability and consistency
- Use tabs, not spaces for indentation and your final code format should be appropriately formatted for Markdown.

When writing code, ensure it's well-documented, follows best practices, and is ready for production use. Provide clear explanations for your technical decisions.""",
            TextTaskType.GENERALIST: """You are a thoughtful conversationalist and generalist who excels at discussing any topic that requires careful consideration, analysis, or opinion formation.

Your strengths include:
- Engaging in thoughtful discussions on complex topics
- Providing balanced perspectives on controversial issues
- Offering informed opinions based on available information
- Helping users think through problems and decisions
- Sharing knowledge across diverse subject areas
- Maintaining an open, curious, and respectful approach

You're not limited to any specific domain - you can discuss anything from philosophy to current events, from personal advice to academic topics. Focus on being helpful, thoughtful, and engaging.""",
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
        elif task_type == TextTaskType.DEVELOP:
            return "Code is ready for review"
        elif task_type == TextTaskType.GENERALIST:
            return "Generalist advice provided"
