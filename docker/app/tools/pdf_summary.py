"""
PDF Summary Tool

This tool generates or retrieves document summaries for uploaded PDFs.
It will create summaries on-demand if they don't already exist.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from pydantic import Field
from services.pdf_summarization_service import PDFSummarizationService
from tools.base import BaseTool, BaseToolResponse

# Configure logger
logger = logging.getLogger(__name__)


class PDFSummaryResponse(BaseToolResponse):
    """Response from PDF summary retrieval tool"""

    success: bool = Field(description="Whether the retrieval was successful")
    filename: str = Field(description="Name of the PDF file")
    summary_type: str = Field(description="Type of summary retrieved")
    document_summary: Optional[str] = Field(None, description="Overall document summary")
    page_summaries: Optional[List[Dict[str, Any]]] = Field(None, description="Page-level summaries")
    message: str = Field(description="Status message")
    direct_response: bool = Field(default=True, description="This provides a direct response to the user")


class PDFSummaryTool(BaseTool):
    """
    PDF Summary Tool

    This tool generates or retrieves summaries for PDF documents.
    It will create summaries on-demand if they don't already exist.
    """

    def __init__(self):
        super().__init__()
        self.name = "retrieve_pdf_summary"
        self.description = "Use this tool ONLY when the user explicitly asks for a 'summary', 'overview', 'brief', or 'condensed version' of a PDF. This generates AI summaries for large PDFs (>10 pages). Do NOT use for: general PDF questions, finding specific information, or when users ask what's in a PDF (use retrieve_pdf_content instead)."
        self.summarization_service = None  # Will be initialized on first use

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert the tool to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Optional filename to get summary for. If not provided, will use the most recent PDF.",
                        },
                        "summary_type": {
                            "type": "string",
                            "enum": ["document", "pages", "all"],
                            "default": "document",
                            "description": "Type of summary to retrieve: 'document' for overall summary, 'pages' for page-level summaries, 'all' for both",
                        },
                    },
                    "required": [],
                },
            },
        }

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for BaseTool interface"""
        return self.to_openai_format()

    def execute(self, params: Dict[str, Any]) -> PDFSummaryResponse:
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def run_with_dict(self, params: Dict[str, Any]) -> PDFSummaryResponse:
        """
        Execute summary retrieval

        Args:
            params: Dictionary containing parameters

        Returns:
            PDFSummaryResponse
        """
        filename = params.get("filename", None)
        summary_type = params.get("summary_type", "document")

        # Initialize session controller to access PDFs
        try:
            from controllers.session_controller import SessionController
            from models.chat_config import ChatConfig

            config = ChatConfig.from_environment()
            session_controller = SessionController(config)

            # Get PDF documents from session
            pdf_documents = session_controller.get_pdf_documents()

            if not pdf_documents:
                return PDFSummaryResponse(
                    success=False,
                    filename=filename or "Unknown",
                    summary_type=summary_type,
                    message="No PDF documents found. Please upload a PDF first.",
                    direct_response=True,
                )

            # Find the PDF to summarize
            pdf_data = None
            pdf_id = None

            if filename:
                # Look for specific filename
                for pid, pdata in pdf_documents.items():
                    if pdata.get("filename", "").lower() == filename.lower():
                        pdf_data = pdata
                        pdf_id = pid
                        break
            else:
                # Get the most recent PDF
                pdf_ids = list(pdf_documents.keys())
                if pdf_ids:
                    pdf_id = pdf_ids[-1]
                    pdf_data = pdf_documents[pdf_id]

            if not pdf_data:
                return PDFSummaryResponse(
                    success=False,
                    filename=filename or "Unknown",
                    summary_type=summary_type,
                    message=f"PDF document '{filename}' not found." if filename else "No PDF documents available.",
                    direct_response=True,
                )

            actual_filename = pdf_data.get("filename", "Unknown")

        except Exception as e:
            logger.error(f"Error accessing session controller: {e}")
            return PDFSummaryResponse(
                success=False,
                filename=filename or "Unknown",
                summary_type=summary_type,
                message=f"Error accessing PDF documents: {str(e)}",
                direct_response=True,
            )

        # Check if summarization is complete
        if not pdf_data.get("summarization_complete", False):
            total_pages = len(pdf_data.get("pages", []))

            # For small documents, inform user that summarization isn't needed
            if total_pages <= 10:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message=f"**{actual_filename}** has only {total_pages} pages. For documents this size, I can analyze the content directly without needing a summary. Please ask specific questions about the document instead.",
                    direct_response=True,
                )

            # For large documents, generate summary on-demand
            logger.info(f"Generating on-demand summary for {actual_filename} ({total_pages} pages)")

            try:
                # Initialize summarization service if needed
                if self.summarization_service is None:
                    config = ChatConfig.from_environment()
                    self.summarization_service = PDFSummarizationService(config)

                # Show progress message
                import streamlit as st

                with st.spinner(f"📝 Generating summary for **{actual_filename}** ({total_pages} pages)..."):
                    # Perform synchronous summarization for immediate results
                    enhanced_pdf_data = self.summarization_service.summarize_pdf_sync(pdf_data)

                # Update the PDF data in storage
                from services.file_storage_service import FileStorageService

                file_storage = FileStorageService()

                if file_storage.update_pdf(pdf_id, enhanced_pdf_data):
                    logger.info(f"Updated PDF '{actual_filename}' with summarization data")
                    pdf_data = enhanced_pdf_data
                else:
                    logger.error(f"Failed to update PDF '{actual_filename}' in storage")

            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message=f"❌ Error generating summary for **{actual_filename}**: {str(e)}",
                    direct_response=True,
                )

        # Retrieve the requested summary
        document_summary = pdf_data.get("document_summary", None)
        page_summaries = pdf_data.get("page_summaries", None)

        # Format the response based on summary type
        if summary_type == "document":
            if document_summary:
                return PDFSummaryResponse(
                    success=True,
                    filename=actual_filename,
                    summary_type=summary_type,
                    document_summary=document_summary,
                    message=f"## Summary of {actual_filename}\n\n{document_summary}",
                    direct_response=True,
                )
            else:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message="Document summary not available.",
                    direct_response=True,
                )

        elif summary_type == "pages":
            if page_summaries:
                formatted_summaries = self._format_page_summaries(page_summaries)
                return PDFSummaryResponse(
                    success=True,
                    filename=actual_filename,
                    summary_type=summary_type,
                    page_summaries=page_summaries,
                    message=f"## Page Summaries for {actual_filename}\n\n{formatted_summaries}",
                    direct_response=True,
                )
            else:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message="Page summaries not available.",
                    direct_response=True,
                )

        else:  # "all"
            parts = [f"## Complete Summary of {actual_filename}\n"]

            if document_summary:
                parts.append("### Document Overview\n")
                parts.append(document_summary)
                parts.append("\n")

            if page_summaries:
                parts.append("### Detailed Page Summaries\n")
                parts.append(self._format_page_summaries(page_summaries))

            if not document_summary and not page_summaries:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message="No summaries available for this document.",
                    direct_response=True,
                )

            return PDFSummaryResponse(
                success=True,
                filename=actual_filename,
                summary_type=summary_type,
                document_summary=document_summary,
                page_summaries=page_summaries,
                message="\n".join(parts),
                direct_response=True,
            )

    def _format_page_summaries(self, page_summaries: List[Dict[str, Any]]) -> str:
        """Format page summaries for display"""
        formatted = []

        for summary in page_summaries:
            page_range = summary.get("page_range", "Unknown")
            summary_text = summary.get("summary", "No summary available")
            formatted.append(f"**Pages {page_range}:**\n{summary_text}\n")

        return "\n".join(formatted)


# Create global instance
pdf_summary_tool = PDFSummaryTool()


def get_pdf_summary_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    return pdf_summary_tool.to_openai_format()


def execute_pdf_summary_with_dict(params: Dict[str, Any]) -> PDFSummaryResponse:
    """Execute PDF summary retrieval with parameters as dictionary"""
    return pdf_summary_tool.run_with_dict(params)
