"""
Tool Execution Service

This service handles the execution of tools, including parallel and sequential
execution strategies, and tool-specific modifications.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from tools.registry import tool_registry
from utils.exceptions import ToolExecutionError
from utils.streamlit_context import run_with_streamlit_context

logger = logging.getLogger(__name__)


class ToolExecutionService:
    """Service for executing tools with different strategies"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the tool execution service

        Args:
            config: Application configuration
        """
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.last_tool_responses: List[Dict[str, Any]] = []

    async def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        strategy: str = "parallel",
        current_user_message: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls with the specified strategy

        Args:
            tool_calls: List of tool calls to execute
            strategy: Execution strategy ("parallel" or "sequential")
            current_user_message: Original user message
            messages: Full conversation messages

        Returns:
            List of tool responses
        """
        if not tool_calls:
            return []

        # Apply any tool restrictions
        tool_calls = self._apply_tool_restrictions(tool_calls)

        if strategy == "sequential":
            responses = await self._execute_sequential(tool_calls, current_user_message, messages)
        else:
            responses = await self._execute_parallel(tool_calls, current_user_message, messages)

        self.last_tool_responses = responses
        return responses

    async def _execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        current_user_message: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute tools in parallel"""
        logger.info(f"Executing {len(tool_calls)} tools in parallel")

        tasks = []
        for tool_call in tool_calls:
            task = self._execute_single_tool(tool_call, current_user_message, messages)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = tool_calls[i].get("name", "unknown")
                logger.error(f"Tool {tool_name} failed: {result}")
                responses.append(
                    {"role": "tool", "content": f"Error: {str(result)}", "tool_name": tool_name, "error": True}
                )
            else:
                responses.append(result)

        return responses

    async def _execute_sequential(
        self,
        tool_calls: List[Dict[str, Any]],
        current_user_message: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute tools sequentially"""
        logger.info(f"Executing {len(tool_calls)} tools sequentially")

        responses = []
        for i, tool_call in enumerate(tool_calls):
            try:
                result = await self._execute_single_tool(tool_call, current_user_message, messages)
                result["execution_order"] = i + 1
                responses.append(result)

                # For sequential execution, update messages after each tool
                if result.get("role") == "tool":
                    messages.append(result)

            except Exception as e:
                tool_name = tool_call.get("name", "unknown")
                logger.error(f"Tool {tool_name} failed: {e}")
                responses.append(
                    {
                        "role": "tool",
                        "content": f"Error: {str(e)}",
                        "tool_name": tool_name,
                        "execution_order": i + 1,
                        "error": True,
                    }
                )

        return responses

    async def _execute_single_tool(
        self,
        tool_call: Dict[str, Any],
        current_user_message: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Execute a single tool"""
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})

        if not tool_name:
            raise ToolExecutionError("unknown", "Tool name not provided")

        # Apply tool-specific modifications
        modified_args = await self._apply_tool_modifications(tool_name, tool_args, current_user_message, messages)

        # Execute tool through registry with Streamlit context preserved
        loop = asyncio.get_event_loop()

        # Execute in thread pool with context preservation
        result = await loop.run_in_executor(
            self.executor, run_with_streamlit_context, tool_registry.execute_tool, tool_name, modified_args
        )

        # Format response
        if hasattr(result, "direct_response") and result.direct_response:
            # Get the content from the appropriate field
            if hasattr(result, "message"):
                content = result.message
            elif hasattr(result, "result"):
                content = result.result
            else:
                content = str(result)

            return {
                "role": "direct_response",
                "content": content,
                "tool_name": tool_name,
                "tool_result": result,
            }
        else:
            return {
                "role": "tool",
                "content": result.json() if hasattr(result, "json") else str(result),
                "tool_name": tool_name,
            }

    async def _apply_tool_modifications(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        current_user_message: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Apply tool-specific argument modifications"""
        modified_args = tool_args.copy()

        # Add conversation context for specific tools
        context_tools = ["conversation_context", "text_assistant", "generate_image"]
        if tool_name in context_tools and messages:
            modified_args["messages"] = messages

        # Add messages and PDF data for PDF tools
        pdf_tools = ["retrieve_pdf_summary", "process_pdf_text"]
        if tool_name in pdf_tools:
            if messages:
                modified_args["messages"] = messages

            # Also try to get PDF data directly
            try:
                from models.chat_config import ChatConfig
                from services.pdf_context_service import PDFContextService

                config = ChatConfig.from_environment()
                pdf_service = PDFContextService(config)
                pdf_data = pdf_service.get_latest_pdf_data()
                if pdf_data:
                    modified_args["pdf_data"] = pdf_data
                    logger.debug(f"Added PDF data to {tool_name} arguments")
            except Exception as e:
                logger.debug(f"Could not add PDF data to tool: {e}")

        # Add original prompt for assistant tool
        if tool_name == "text_assistant" and current_user_message:
            user_content = current_user_message.get("content", "")

            # If text is not provided, use the user's original message
            if user_content and "text" not in modified_args:
                modified_args["text"] = user_content

            # If text refers to PDF content but no instructions provided, use user's question as instructions
            text_arg = modified_args.get("text", "").lower()
            if ("pdf" in text_arg or "document" in text_arg) and "instructions" not in modified_args and user_content:
                logger.info(f"Adding user question as instructions for PDF analysis: {user_content}")
                modified_args["instructions"] = user_content

        return modified_args

    def _apply_tool_restrictions(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply tool-specific restrictions"""
        # With automatic PDF injection, we no longer need special PDF tool restrictions
        return tool_calls

    def determine_execution_strategy(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Determine the best execution strategy for the given tools

        Args:
            tool_calls: List of tool calls

        Returns:
            "parallel" or "sequential"
        """
        if not tool_calls or len(tool_calls) == 1:
            return "parallel"

        # Tools that should run sequentially
        sequential_tools = {"conversation_context", "retrieval_search"}

        # Check if any tools require sequential execution
        tool_names = {tc.get("name") for tc in tool_calls}
        if tool_names & sequential_tools:
            return "sequential"

        return "parallel"
