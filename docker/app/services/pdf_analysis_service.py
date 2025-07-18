"""
PDF Analysis Service

This service handles intelligent analysis of PDF documents to answer user questions
by processing pages in batches and combining relevant findings.
"""

import asyncio
import logging
from typing import Dict, List

from models.chat_config import ChatConfig
from utils.config import config
from utils.executor_pool import get_shared_executor
from utils.streamlit_context import run_with_streamlit_context

logger = logging.getLogger(__name__)


class PDFAnalysisService:
    """Service for intelligent PDF document analysis and Q&A"""

    def __init__(self, config_obj: ChatConfig):
        """
        Initialize the PDF analysis service

        Args:
            config_obj: Configuration for the service
        """
        self.config = config_obj
        self.batch_size = (
            config.file_processing.PDF_SUMMARIZATION_BATCH_SIZE
        )  # Reuse batch size
        self.executor = get_shared_executor()

    async def _synthesize_batch_results(
        self, batch_results: List[Dict], user_query: str, filename: str
    ) -> str:
        """Synthesize results from multiple batch analyses"""
        try:
            # Format batch results for synthesis
            formatted_results = []
            for result in batch_results:
                if "page_range" in result:
                    formatted_results.append(
                        f"Analysis of pages {result['page_range']}:\n{result['analysis']}"
                    )
                elif "pages" in result:
                    page_list = ", ".join(map(str, result["pages"]))
                    formatted_results.append(
                        f"Pages {page_list}:\n{result['analysis']}"
                    )
                else:
                    formatted_results.append(result["analysis"])

            combined_findings = "\n\n".join(formatted_results)

            synthesis_params = {
                "task_type": "analyze",
                "text": combined_findings,
                "instructions": f"Based on these analyses from '{filename}', provide a concise answer to: {user_query}. Synthesize all relevant information into a cohesive response.",
            }

            # Import locally to avoid circular imports
            from tools.registry import execute_tool

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                run_with_streamlit_context,
                execute_tool,
                "text_assistant",
                synthesis_params,
            )

            return result.result

        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            # Fallback: return all individual results
            return "\n\n".join(
                [result.get("analysis", "") for result in batch_results if result]
            )
