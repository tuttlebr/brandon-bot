"""
PDF Parser Tool - Retrieves PDF content when relevant to user queries

This tool allows the LLM to selectively retrieve PDF document content
only when it determines the user is asking questions related to uploaded PDFs.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class PDFContentResult(BaseModel):
    """Individual PDF page content"""

    page_number: int = Field(description="Page number")
    text: str = Field(description="Extracted text content from the page")


class PDFParseResponse(BaseModel):
    """Response model for PDF content retrieval"""

    filename: str = Field(description="Name of the PDF file")
    total_pages: int = Field(description="Total number of pages in the PDF")
    pages_requested: List[int] = Field(description="Page numbers that were requested")
    content: List[PDFContentResult] = Field(description="PDF page content")
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Status or error message")
    formatted_results: Optional[str] = Field(None, description="Formatted text for easy reading")


class PDFParserTool:
    """
    PDF Content Retrieval Tool

    This tool retrieves content from uploaded PDF documents stored in session state.
    It allows the LLM to selectively access PDF content only when relevant to the user's query.
    """

    def __init__(self):
        self.name = "retrieve_pdf_content"
        self.description = """Retrieve complete content from user uploaded documents. You MUST use this if the user has uploaded a document.

This tool retrieves the full text content from all pages of PDF documents that have been uploaded to the session. Use this tool when the user asks questions about or references their uploaded PDF documents. The tool returns the complete content of the entire PDF without any page limitations, making it suitable for comprehensive document analysis and reference."""

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
                "parameters": {"type": "object", "properties": {}, "required": [],},
            },
        }

    def _get_pdf_data_from_messages(self, messages: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve PDF data from messages list

        Args:
            messages: List of conversation messages to search in

        Returns:
            PDF data dictionary if available, None otherwise
        """
        if not messages:
            logger.error("No messages provided to search for PDF data")
            return None

        logger.info(f"Searching for PDF data in {len(messages)} messages")

        # Look for recent tool messages containing PDF data
        for message in reversed(messages):
            if message.get("role") == "tool":
                try:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        tool_data = json.loads(content)
                        if (
                            isinstance(tool_data, dict)
                            and tool_data.get("tool_name") == "process_pdf_document"
                            and tool_data.get("status") == "success"
                        ):
                            logger.info(f"✅ Found PDF data for: {tool_data.get('filename', 'Unknown')}")
                            return tool_data
                except (json.JSONDecodeError, TypeError):
                    continue

        logger.warning("❌ No PDF data found in messages")
        return None

    def _format_content_for_display(self, content: List[PDFContentResult], filename: str) -> str:
        """
        Format PDF content for easy reading

        Args:
            content: List of PDF page content
            filename: Name of the PDF file

        Returns:
            Formatted string for display
        """
        if not content:
            return f"No content found in {filename}"

        formatted_parts = [f"Content from PDF: {filename}\n" + "=" * 50]

        for page_content in content:
            formatted_parts.append(f"\n📄 Page {page_content.page_number}:")
            formatted_parts.append("-" * 30)
            # Limit content length for readability
            text = page_content.text
            if len(text) > 2000:
                text = text[:2000] + "\n... [Content truncated for readability] ..."
            formatted_parts.append(text)

        return "\n".join(formatted_parts)

    def run_with_dict(self, params: Dict[str, Any]) -> PDFParseResponse:
        """
        Execute PDF content retrieval - returns all pages from the uploaded PDF

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'messages'

        Returns:
            PDFParseResponse: PDF content retrieval response
        """
        messages = params.get("messages", None)  # Get messages from parameters

        logger.info(f"PDF content retrieval called, messages_provided: {messages is not None}")

        # Get PDF data from messages
        pdf_data = self._get_pdf_data_from_messages(messages)

        if not pdf_data:
            return PDFParseResponse(
                filename="No PDF found",
                total_pages=0,
                pages_requested=[],
                content=[],
                success=False,
                message="No PDF document found in the current session. Please upload a PDF document first.",
            )

        filename = pdf_data.get("filename", "Unknown PDF")
        pages = pdf_data.get("pages", [])
        total_pages = len(pages)

        if not pages:
            return PDFParseResponse(
                filename=filename,
                total_pages=0,
                pages_requested=[],
                content=[],
                success=False,
                message="PDF document found but contains no readable pages.",
            )

        # Retrieve all pages from the PDF
        content_results = []
        pages_requested = []

        for i in range(total_pages):
            page_data = pages[i]
            page_num = page_data.get("page", i + 1)
            content_results.append(PDFContentResult(page_number=page_num, text=page_data.get("text", "")))
            pages_requested.append(page_num)

        # Format results for display
        formatted_results = self._format_content_for_display(content_results, filename)

        success_message = f"Retrieved complete content from all {len(content_results)} pages of '{filename}'"

        return PDFParseResponse(
            filename=filename,
            total_pages=total_pages,
            pages_requested=pages_requested,
            content=content_results,
            success=True,
            message=success_message,
            formatted_results=formatted_results,
        )


# Create a global instance and helper functions for easy access
pdf_parser_tool = PDFParserTool()


def get_pdf_parser_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for PDF content retrieval

    Returns:
        Dict containing the OpenAI tool definition
    """
    return pdf_parser_tool.to_openai_format()


def execute_pdf_parse_with_dict(params: Dict[str, Any]) -> PDFParseResponse:
    """
    Execute PDF content retrieval with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters

    Returns:
        PDFParseResponse: PDF content retrieval response
    """
    return pdf_parser_tool.run_with_dict(params)
