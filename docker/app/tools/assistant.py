import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from openai import OpenAI
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class AssistantTaskType(str, Enum):
    """Enumeration of assistant task types"""

    SUMMARIZE = "summarize"
    PROOFREAD = "proofread"
    REWRITE = "rewrite"
    CRITIC = "critic"


class AssistantResponse(BaseModel):
    """Response from the assistant tool"""

    original_text: str = Field(description="The original input text")
    task_type: AssistantTaskType = Field(description="The type of task performed")
    result: str = Field(description="The processed result")
    improvements: Optional[List[str]] = Field(None, description="List of improvements made (for proofreading)")
    summary_length: Optional[int] = Field(None, description="Length of summary in words (for summarizing)")
    processing_notes: Optional[str] = Field(None, description="Additional notes about the processing")
    direct_response: bool = Field(
        default=True, description="Flag indicating this response should be returned directly to user",
    )


class AssistantTool:
    """Tool for text processing tasks including summarizing, proofreading, and rewriting"""

    def __init__(self):
        self.name = "text_assistant"
        self.description = "Triggered when user asks for writing tasks like summarizing, proofreading, critiquing, or rewriting text. Specify the task type and provide the text to be processed without modification or truncation."

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI function calling format

        Returns:
            Dict containing the OpenAI-compatible tool definition
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["summarize", "proofread", "rewrite", "critic"],
                            "description": "The type of task to perform: 'summarize' to create a concise summary, 'proofread' to check for errors and suggest improvements, 'rewrite' to rephrase and improve the text, 'critic' to critique the text and provide feedback on the text",
                        },
                        "text": {"type": "string", "description": "The text content to be processed",},
                        "instructions": {
                            "type": "string",
                            "description": "Optional specific instructions for the task (e.g., 'make it more formal', 'bullet points only', 'fix grammar only')",
                        },
                    },
                    "required": ["task_type", "text"],
                },
            },
        }

    def _get_system_prompt(self, task_type: AssistantTaskType, instructions: Optional[str] = None) -> str:
        """
        Get the appropriate system prompt for the task type

        Args:
            task_type: The type of assistant task
            instructions: Optional user instructions

        Returns:
            Formatted system prompt
        """
        base_prompts = {
            AssistantTaskType.SUMMARIZE: """detailed thinking on
You are a precision-focused summarization expert. Your task is to distill complex content into concise, accessible overviews that preserve the original intent while prioritizing:
- **Core Concept Extraction**: Identifying pivotal data points, arguments, or themes
- **Contextual Balance**: Retaining critical nuance without overcomplicating the narrative
- **Audience Alignment**: Tailoring density and detail to the target reader’s needs
- **Structural Integrity**: Organizing information in a logical, progressive flow
Avoid redundancy while ensuring the summary stands alone as a complete representation of the source material.""",
            AssistantTaskType.PROOFREAD: """detailed thinking on
You are a meticulous editorial specialist. Your task is to scrutinize text through a multi-phase refinement process:
1. **Error Elimination**: Correcting grammatical/syntactical flaws, spelling inaccuracies, and punctuation misuses
2. **Clarity Enhancement**: Simplifying convoluted phrasing and ambiguous language
3. **Consistency Audit**: Standardizing tone, style, and formatting conventions
4. **Flow Optimization**: Smoothing transitions between ideas and sentences
Provide the polished text alongside annotations explaining critical revisions and their impact on readability/credibility.""",
            AssistantTaskType.REWRITE: """detailed thinking on
You are a strategic content reimaginer. Your task is to transform existing text through:
- **Tonal Adaptation**: Shifting formality, voice, or perspective to align with target audiences
- **Engagement Enhancement**: Injecting dynamic language, vivid imagery, and persuasive elements
- **Structural Reinvention**: Reorganizing ideas for maximum impact (e.g., inverted pyramids, narrative arcs)
- **Precision Refinement**: Upgrading vocabulary and syntax for sophistication without obscurity
- **Purpose-Driven Editing**: Emphasizing key messages while maintaining factual/tonal fidelity
Deliver a polished version that elevates the source material’s intellectual and emotional resonance.""",
            AssistantTaskType.CRITIC: """detailed thinking on
You are an expert literary agent. Your task is to critique the text and provide feedback on the text.
A literary agent evaluates new fiction submissions through a structured process to assess a manuscript’s potential and an author’s readiness for publication. Key steps include:

- Initial Screening: Reviewing the query letter for clarity and a compelling hook, and the synopsis for a unique premise, strong character arcs, and alignment with genre/market trends.
- Manuscript Evaluation: Analyzing originality, narrative voice, pacing, plot coherence, and character development, while identifying weaknesses (e.g., clichés, plot holes) and assessing commercial viability compared to genre bestsellers.
- Market Analysis: Ensuring the manuscript aligns with current publisher demands (e.g., diverse voices) and evaluating the author’s platform (e.g., social media, promotional ability).
- Feedback & Collaboration: Offering editorial guidance for revisions and discussing long-term career strategies if representation is pursued.
- Business Negotiation: Crafting pitches to secure publisher offers and advocating for favorable contract terms (advances, royalties).
Agents prioritize projects they are passionate about (subjective fit) and balance creative merit with commercial pragmatism. Authors are advised to polish their query materials, ensure error-free manuscripts, and research agents with relevant genre expertise.""",
        }

        prompt = base_prompts.get(task_type, base_prompts[AssistantTaskType.REWRITE])

        if instructions:
            prompt += f"\n\nAdditional instructions: {instructions}"

        return prompt

    def _create_llm_client(self, config: ChatConfig) -> OpenAI:
        """
        Create an OpenAI client for text processing using the chat configuration

        Args:
            config: ChatConfig instance with LLM configuration

        Returns:
            OpenAI client instance
        """
        try:
            return OpenAI(api_key=config.api_key, base_url=config.intelligent_llm_endpoint)
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            raise

    def _process_text(
        self, task_type: AssistantTaskType, text: str, config: ChatConfig, instructions: Optional[str] = None,
    ) -> AssistantResponse:
        """
        Process text using the appropriate LLM task

        Args:
            task_type: The type of processing task
            text: The text to process
            config: ChatConfig instance with LLM configuration
            instructions: Optional user instructions

        Returns:
            AssistantResponse with the processed result
        """
        logger.info(f"Processing text with task type: {task_type}")

        try:
            client = self._create_llm_client(config)
            system_prompt = self._get_system_prompt(task_type, instructions)

            # Prepare the user message based on task type
            if task_type == AssistantTaskType.SUMMARIZE:
                user_message = f"Please summarize the following text:\n\n{text}"
            elif task_type == AssistantTaskType.PROOFREAD:
                user_message = (
                    f"Please proofread the following text and provide corrections with explanations:\n\n{text}"
                )
            elif task_type == AssistantTaskType.REWRITE:
                user_message = f"Please rewrite and improve the following text:\n\n{text}"
            elif task_type == AssistantTaskType.CRITIC:
                user_message = f"Please critique the following text and provide feedback on the text:\n\n{text}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(f"Making LLM request with model: {config.llm_model_name}")

            response = client.chat.completions.create(
                model=config.intelligent_llm_model_name,
                messages=messages,
                temperature=0.6,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
            )

            result = response.choices[0].message.content.strip()

            # Process the response based on task type
            improvements = None
            summary_length = None
            processing_notes = None

            if task_type == AssistantTaskType.SUMMARIZE:
                summary_length = len(result.split())
                processing_notes = f"Original text: {len(text.split())} words, Summary: {summary_length} words"
            elif task_type == AssistantTaskType.PROOFREAD:
                # Try to extract improvements from the response
                if "improvements" in result.lower() or "corrections" in result.lower():
                    improvements = ["See detailed feedback in the result"]
                processing_notes = "Proofreading completed with suggestions for improvement"
            elif task_type == AssistantTaskType.REWRITE:
                processing_notes = "Text has been rewritten for improved clarity and flow"
            elif task_type == AssistantTaskType.CRITIC:
                processing_notes = "Text has been critiqued and feedback provided"

            logger.info(f"Successfully processed text with {task_type}")

            return AssistantResponse(
                original_text=(text[:500] + "..." if len(text) > 500 else text),  # Truncate for storage
                task_type=task_type,
                result=result,
                improvements=improvements,
                summary_length=summary_length,
                processing_notes=processing_notes,
            )

        except Exception as e:
            logger.error(f"Error processing text with {task_type}: {e}")
            raise

    def _run(self, task_type: str = None, text: str = None, instructions: str = None, **kwargs,) -> AssistantResponse:
        """
        Execute an assistant task with the given parameters.

        Args:
            task_type: The type of task to perform
            text: The text to process
            instructions: Optional instructions
            **kwargs: Can accept a dictionary with parameters

        Returns:
            AssistantResponse: The processed result
        """
        # Support both direct parameters and dictionary input
        if task_type is None and "task_type" in kwargs:
            task_type = kwargs["task_type"]
        if text is None and "text" in kwargs:
            text = kwargs["text"]
        if instructions is None and "instructions" in kwargs:
            instructions = kwargs["instructions"]

        if task_type is None:
            raise ValueError("task_type parameter is required")
        if text is None:
            raise ValueError("text parameter is required")

        # Validate task type
        try:
            task_enum = AssistantTaskType(task_type.lower())
        except ValueError:
            raise ValueError(f"Invalid task_type: {task_type}. Must be one of: {[t.value for t in AssistantTaskType]}")

        logger.debug(f"_run method called with task_type: '{task_type}', text length: {len(text)}")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self._process_text(task_enum, text, config, instructions)

    def run_with_dict(self, params: Dict[str, Any]) -> AssistantResponse:
        """
        Execute an assistant task with parameters provided as a dictionary.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'task_type', 'text', and optionally 'instructions'

        Returns:
            AssistantResponse: The processed result
        """
        if "task_type" not in params:
            raise ValueError("'task_type' key is required in parameters dictionary")
        if "text" not in params:
            raise ValueError("'text' key is required in parameters dictionary")

        task_type = params["task_type"]
        text = params["text"]
        instructions = params.get("instructions")

        logger.debug(f"run_with_dict method called with task_type: '{task_type}', text length: {len(text)}")

        # Create config from environment
        config = ChatConfig.from_environment()
        return self._process_text(AssistantTaskType(task_type.lower()), text, config, instructions)


# Create a global instance and helper functions for easy access
assistant_tool = AssistantTool()


def get_assistant_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for text assistant

    Returns:
        Dict containing the OpenAI tool definition
    """
    return assistant_tool.to_openai_format()


def execute_assistant_task(task_type: str, text: str, instructions: Optional[str] = None) -> AssistantResponse:
    """
    Execute an assistant task with the given parameters

    Args:
        task_type: The type of task to perform ('summarize', 'proofread', 'rewrite')
        text: The text to process
        instructions: Optional specific instructions

    Returns:
        AssistantResponse: The processed result
    """
    config = ChatConfig.from_environment()
    return assistant_tool._process_text(AssistantTaskType(task_type.lower()), text, config, instructions)


def execute_assistant_with_dict(params: Dict[str, Any]) -> AssistantResponse:
    """
    Execute an assistant task with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters
               Expected keys: 'task_type', 'text', and optionally 'instructions'

    Returns:
        AssistantResponse: The processed result
    """
    return assistant_tool.run_with_dict(params)
