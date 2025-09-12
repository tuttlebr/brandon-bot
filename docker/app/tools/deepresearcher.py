"""
Deep Researcher Tool - Multi-turn Research Assistant

This tool performs deep research by making multiple iterations and calls to
other tools until it achieves sufficient depth and quality in answering
research questions.
"""

import asyncio
import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from models.chat_config import ChatConfig
from pydantic import BaseModel, Field
from services.llm_client_service import LLMClientService
from tools.base import (
    BaseTool,
    BaseToolResponse,
    StreamingToolResponse,
    ToolController,
    ToolView,
)
from tools.registry import execute_tool
from utils.text_processing import strip_think_tags

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for research findings"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_float(cls, value: float) -> "ConfidenceLevel":
        """Convert float confidence to categorical level"""
        if value >= 0.8:
            return cls.HIGH
        elif value >= 0.6:
            return cls.MEDIUM
        else:
            return cls.LOW

    def to_float(self) -> float:
        """Convert categorical level to float for backward compatibility"""
        if self == ConfidenceLevel.HIGH:
            return 0.85
        elif self == ConfidenceLevel.MEDIUM:
            return 0.7
        else:
            return 0.4


FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["questions"],
}

GAPS_SCHEMA = {
    "type": "object",
    "properties": {"gaps": {"type": "array", "items": {"type": "string"}}},
    "required": ["gaps"],
}

# Test with most basic schema first
COMPLETENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_more_research": {"type": "boolean"},
        "next_phase": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": [
        "needs_more_research",
        "next_phase",
        "confidence",
        "reasoning",
    ],
}

# Basic schema for synthesis
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "synthesis": {"type": "string"},
        "confidence_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "citations_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "synthesis",
        "confidence_level",
        "key_findings",
        "citations_used",
    ],
}


class ResearchPhase(str, Enum):
    """Phases of deep research"""

    INITIAL_SEARCH = "initial_search"
    DEEP_DIVE = "deep_dive"
    SYNTHESIS = "synthesis"
    QUALITY_CHECK = "quality_check"


class ResearchSource(BaseModel):
    """Information about a research source"""

    source_type: str = Field(description="Type of source (web, pdf, etc)")
    url: Optional[str] = Field(None, description="URL if applicable")
    title: str = Field(description="Title or description of the source")
    content: str = Field(description="Relevant content from the source")
    relevance_score: float = Field(0.0, description="Relevance score 0-1")
    extracted_facts: List[str] = Field(
        default_factory=list, description="Key facts extracted"
    )
    citation_id: Optional[str] = Field(
        None, description="Unique citation identifier (e.g., [1], [2])"
    )
    author: Optional[str] = Field(None, description="Author if available")
    date: Optional[str] = Field(
        None, description="Publication date if available"
    )
    domain: Optional[str] = Field(None, description="Domain/publisher")


class ResearchIteration(BaseModel):
    """Record of a single research iteration"""

    iteration_number: int
    phase: ResearchPhase
    query: str
    tools_used: List[str]
    sources_found: List[ResearchSource]
    synthesis: str
    needs_more_research: bool
    next_questions: List[str] = Field(default_factory=list)


class DeepResearchResponse(BaseToolResponse):
    """Response from the deep researcher tool"""

    original_query: str = Field(description="The original research query")
    iterations: List[ResearchIteration] = Field(
        default_factory=list, description="Research iterations performed"
    )
    final_synthesis: str = Field(
        description="Final synthesized research answer"
    )
    total_sources: int = Field(
        0, description="Total number of sources consulted"
    )
    confidence_level: ConfidenceLevel = Field(
        ConfidenceLevel.MEDIUM,
        description="Confidence in the answer (low, medium, high)",
    )
    research_depth: str = Field(
        "basic",
        description="Depth achieved: basic, moderate, deep, comprehensive",
    )
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings from the research"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Limitations or gaps in the research"
    )
    references: List[str] = Field(
        default_factory=list,
        description="List of cited references in academic format",
    )
    direct_response: bool = Field(
        default=True,
        description="Response should be returned directly to user",
    )


class StreamingDeepResearchResponse(StreamingToolResponse):
    """Streaming response from the deep researcher tool"""

    original_query: str = Field(description="The original research query")
    direct_response: bool = Field(
        default=True,
        description="Response should be returned directly to user",
    )


class DeepResearchController(ToolController):
    """Controller handling deep research logic"""

    MAX_ITERATIONS = 5
    MIN_CONFIDENCE_THRESHOLD = ConfidenceLevel.MEDIUM

    def __init__(self, config: ChatConfig, llm_type: str):
        self.config = config
        self.llm_type = llm_type
        self.llm_client_service = LLMClientService()
        self.llm_client_service.initialize(config)

    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process synchronously by delegating to async method"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.process_async(params))

    async def process_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the deep research request asynchronously"""
        query = params["query"]
        max_iterations = params.get("max_iterations", self.MAX_ITERATIONS)
        target_depth = params.get("target_depth", "moderate")
        stream = params.get("stream", True)  # Default back to streaming

        logger.info("Starting deep research for query: %s", query)

        # If streaming is requested, return a streaming response
        if stream:
            return {
                "original_query": query,
                "is_streaming": True,
                "content_generator": self._stream_research_process(
                    query, max_iterations, target_depth
                ),
            }

        # Otherwise, run the non-streaming version
        return await self._process_non_streaming(
            query, max_iterations, target_depth
        )

    async def _stream_research_process(
        self, query: str, max_iterations: int, target_depth: str
    ):
        """Stream research progress updates to the user"""
        # Start with a clean status message
        yield "_Researching..._\n\n"

        iterations = []
        all_sources = []
        current_synthesis = ""
        confidence_level = ConfidenceLevel.MEDIUM
        extracted_urls = set()  # Track URLs that have been extracted

        # Provide real-time updates during the search
        yield "> _Phase 1: Initial research phase_\n\n"
        iteration = await self._perform_research_iteration(
            query=query,
            phase=ResearchPhase.INITIAL_SEARCH,
            iteration_number=1,
            previous_synthesis=current_synthesis,
            all_sources=all_sources,
            extracted_urls=extracted_urls,
        )
        iterations.append(iteration)
        all_sources.extend(iteration.sources_found)
        current_synthesis = iteration.synthesis

        # Update status after first iteration
        yield f"> > _{len(iteration.sources_found)} initial sources_\n\n"

        # Continue iterations with progress updates
        iteration_count = 1
        while iteration_count < max_iterations:
            # Update status for iteration start
            yield f"\n> _Phase {iteration_count + 1}: Deep research iteration {iteration_count + 1}/{max_iterations}_\n\n"

            # Check if we need more research
            needs_more, next_phase = await self._assess_research_completeness(
                current_synthesis=current_synthesis,
                sources=all_sources,
                target_depth=target_depth,
                iteration_count=iteration_count,
            )

            if not needs_more:
                # Update status to show we're done with iterations
                yield f"\n_Research complete after {iteration_count} iterations_\n\n"
                break

            # Continue with next iteration
            iteration_count += 1

            # Generate follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(
                original_query=query,
                current_synthesis=current_synthesis,
                gaps=iteration.next_questions if iterations else [],
            )

            if follow_up_questions:
                # Update status for follow-up questions
                num_questions = min(2, len(follow_up_questions))
                yield f"> > _Found {num_questions} follow-up questions to explore_\n\n"

            # Execute follow-up research
            questions_to_research = follow_up_questions[:2]
            for i, question in enumerate(questions_to_research, 1):
                # Update status for sub-question
                yield f"> > > _Researching sub-question {i}/{len(questions_to_research)}..._\n\n"

                iteration = await self._perform_research_iteration(
                    query=question,
                    phase=next_phase,
                    iteration_number=iteration_count,
                    previous_synthesis=current_synthesis,
                    all_sources=all_sources,
                    extracted_urls=extracted_urls,
                )
                iterations.append(iteration)
                all_sources.extend(iteration.sources_found)

                # Update synthesis
                current_synthesis = await self._update_synthesis(
                    previous_synthesis=current_synthesis,
                    new_iteration=iteration,
                    original_query=query,
                )

        # Final synthesis
        yield "\n_Creating final synthesis..._\n\n"

        result = await self._create_final_synthesis(
            query=query,
            iterations=iterations,
            all_sources=all_sources,
            current_synthesis=current_synthesis,
        )
        final_synthesis, confidence_level, key_findings, citations_used = (
            result
        )

        # Review and clean up markdown formatting
        # yield "_Reviewing markdown formatting..._\n\n"
        final_synthesis = await self._review_markdown_formatting(
            final_synthesis
        )

        # Determine research depth (not used in streaming mode, but computed for logging)
        # research_depth = self._determine_research_depth(
        #     iterations=len(iterations),
        #     sources=len(all_sources),
        #     confidence=confidence)

        # Final report heading
        # yield "---\n\n"
        # yield f"_Research Depth:_ {research_depth.upper()}\n"
        # yield f"_Confidence Level:_ {confidence:.1%}\n"
        # yield f"_Key Findings:_ {len(key_findings)}\n\n"

        if key_findings:
            yield "### Key Findings:\n\n"
            for finding in key_findings:
                yield f"- {finding}\n\n"
            yield "\n"

        yield final_synthesis

        # Add references section (cited sources)
        references = self._generate_references(all_sources, citations_used)
        if references:
            yield "\n\n> _References (Cited):_\n\n"
            for ref in references:
                yield f"> >{ref}\n\n"

        # Add all consulted sources if there are uncited ones
        uncited_sources = [
            s
            for s in all_sources
            if s.citation_id and s.citation_id not in citations_used
        ]
        if uncited_sources:
            yield "\n\n> _Additional Sources Consulted:_\n\n"
            # Sort by citation ID
            uncited_sources.sort(
                key=lambda x: (
                    int(x.citation_id.strip("[]")) if x.citation_id else 999
                )
            )
            for source in uncited_sources:
                ref = self._format_single_reference(source)
                yield f"> >{ref}\n\n"

        # Add limitations if any
        limitations = await self._identify_limitations(
            query=query, sources=all_sources, synthesis=final_synthesis
        )

        if limitations:
            yield "\n\n>_Research Limitations:_\n\n"
            for limitation in limitations:
                yield f"> > _{limitation}_\n"

    async def _process_non_streaming(
        self, query: str, max_iterations: int, target_depth: str
    ) -> Dict[str, Any]:
        """Process research without streaming (original implementation)"""
        iterations = []
        all_sources = []
        current_synthesis = ""
        confidence_level = ConfidenceLevel.MEDIUM
        extracted_urls = set()  # Track URLs that have been extracted

        # Phase 1: Initial broad search
        logger.info(
            "=== ITERATION 1/%d === Starting initial search phase",
            max_iterations,
        )
        iteration = await self._perform_research_iteration(
            query=query,
            phase=ResearchPhase.INITIAL_SEARCH,
            iteration_number=1,
            previous_synthesis=current_synthesis,
            all_sources=all_sources,
            extracted_urls=extracted_urls,
        )
        iterations.append(iteration)
        all_sources.extend(iteration.sources_found)
        current_synthesis = iteration.synthesis

        logger.info(
            "← Iteration 1 complete: Found %d sources, synthesis length: %d chars",
            len(iteration.sources_found),
            len(current_synthesis),
        )

        # Continue iterations until we have sufficient depth or hit max iterations
        iteration_count = 1
        while iteration_count < max_iterations:
            logger.info(
                "=== ITERATION %d/%d === Current depth: %s, Sources: %d",
                iteration_count + 1,
                max_iterations,
                self._determine_research_depth(
                    iterations=len(iterations),
                    sources=len(all_sources),
                    confidence_level=ConfidenceLevel.LOW,
                ),
                len(all_sources),
            )

            # Check if we need more research
            logger.info(
                "Assessing research completeness after %d iterations",
                iteration_count,
            )
            needs_more, next_phase = await self._assess_research_completeness(
                current_synthesis=current_synthesis,
                sources=all_sources,
                target_depth=target_depth,
                iteration_count=iteration_count,
            )

            if not needs_more:
                logger.info(
                    "Research COMPLETE after %d iterations - Sufficient depth achieved",
                    iteration_count,
                )
                break

            # Perform next iteration
            iteration_count += 1
            logger.info(
                "→ Continuing to iteration %d - Next phase: %s",
                iteration_count,
                next_phase.value,
            )

            # Generate focused follow-up questions
            logger.info("Generating follow-up questions for deeper research")
            follow_up_questions = await self._generate_follow_up_questions(
                original_query=query,
                current_synthesis=current_synthesis,
                gaps=iteration.next_questions if iterations else [],
            )
            logger.info(
                "← Generated %d follow-up questions", len(follow_up_questions)
            )
            if follow_up_questions:
                for i, q in enumerate(follow_up_questions[:10], 1):
                    logger.debug("  %d. %s", i, q[:80])

            # Execute follow-up research
            # Limit to 2 questions per iteration
            questions_to_research = follow_up_questions[:2]
            logger.info(
                "→ Researching %d follow-up questions",
                len(questions_to_research),
            )

            for i, question in enumerate(questions_to_research, 1):
                logger.info("  → Question %d: %s", i, question[:100])
                iteration = await self._perform_research_iteration(
                    query=question,
                    phase=next_phase,
                    iteration_number=iteration_count,
                    previous_synthesis=current_synthesis,
                    all_sources=all_sources,
                    extracted_urls=extracted_urls,
                )
                iterations.append(iteration)
                all_sources.extend(iteration.sources_found)

                # Update synthesis with new information
                current_synthesis = await self._update_synthesis(
                    previous_synthesis=current_synthesis,
                    new_iteration=iteration,
                    original_query=query,
                )

                logger.info(
                    "← Sub-question %d complete: Added %d sources",
                    i,
                    len(iteration.sources_found),
                )

        # Final quality check and synthesis
        logger.info(
            "Creating final synthesis after %d iterations with %d sources",
            len(iterations),
            len(all_sources),
        )
        result = await self._create_final_synthesis(
            query=query,
            iterations=iterations,
            all_sources=all_sources,
            current_synthesis=current_synthesis,
        )
        final_synthesis, confidence_level, key_findings, citations_used = (
            result
        )
        logger.info(
            "Final synthesis complete - confidence: %s, key findings: %d",
            confidence_level.value,
            len(key_findings),
        )

        # Review and clean up markdown formatting
        logger.info("Reviewing markdown formatting...")
        final_synthesis = await self._review_markdown_formatting(
            final_synthesis
        )
        logger.info("Markdown review complete")

        # Determine research depth achieved
        research_depth = self._determine_research_depth(
            iterations=len(iterations),
            sources=len(all_sources),
            confidence_level=confidence_level,
        )
        logger.info(
            "Research depth achieved: %s (iterations=%d, sources=%d, confidence=%s)",
            research_depth.upper(),
            len(iterations),
            len(all_sources),
            confidence_level.value,
        )

        # Generate references
        logger.info("→ Generating references for cited sources")
        references = self._generate_references(all_sources, citations_used)
        logger.info("← Generated %d references", len(references))

        # Identify limitations
        logger.info("→ Identifying research limitations")
        limitations = await self._identify_limitations(
            query=query, sources=all_sources, synthesis=final_synthesis
        )
        logger.info("← Identified %d limitations", len(limitations))

        result = {
            "original_query": query,
            "iterations": iterations,
            "final_synthesis": final_synthesis,
            "total_sources": len(all_sources),
            "confidence_level": confidence_level,
            "research_depth": research_depth,
            "key_findings": key_findings,
            "limitations": limitations,
            "references": references,
        }

        logger.info(
            "\n╔═══ DEEP RESEARCH COMPLETE ═══╗\n"
            "║ Total iterations: %-10d ║\n"
            "║ Total sources: %-13d ║\n"
            "║ Research depth: %-12s ║\n"
            "║ Confidence: %-16s ║\n"
            "║ Key findings: %-14d ║\n"
            "╚══════════════════════════════╝",
            len(iterations),
            len(all_sources),
            research_depth.upper(),
            confidence_level.value.upper(),
            len(key_findings),
        )

        # Format the synthesis with research summary - clean and simple
        # Separate cited and uncited sources
        uncited_sources = [
            s
            for s in all_sources
            if s.citation_id and s.citation_id not in citations_used
        ]
        uncited_sources.sort(
            key=lambda x: (
                int(x.citation_id.strip("[]")) if x.citation_id else 999
            )
        )

        # Build additional sources section if needed
        additional_sources_section = ""
        if uncited_sources:
            additional_refs = []
            for source in uncited_sources:
                additional_refs.append(self._format_single_reference(source))
            additional_sources_section = f"\n\n### Additional Sources Consulted\n\n{chr(10).join(additional_refs)}"

        research_summary = f"""## Deep Research Complete

**Research Statistics:**
- Iterations performed: {len(iterations)}
- Sources analyzed: {len(all_sources)}
- Sources cited: {len(references)}
- Research depth: {research_depth.upper()}
- Confidence level: {confidence_level.value.upper()}

**Key Findings:** {len(key_findings)}

{final_synthesis}

### References (Cited)

{chr(10).join(references)}{additional_sources_section}"""

        # Review the complete research summary for markdown formatting
        research_summary = await self._review_markdown_formatting(
            research_summary
        )

        result["final_synthesis"] = research_summary

        return result

    async def _perform_research_iteration(
        self,
        query: str,
        phase: ResearchPhase,
        iteration_number: int,
        previous_synthesis: str,
        all_sources: List[ResearchSource],
        extracted_urls: Optional[set] = None,
    ) -> ResearchIteration:
        """Perform a single research iteration"""
        logger.info(
            "┌─ Iteration %d - Phase: %s\n" "└─ Query: %s",
            iteration_number,
            phase.value,
            query[:100],
        )

        tools_used = []
        sources_found = []
        # Determine which tools to use based on phase
        if phase == ResearchPhase.INITIAL_SEARCH:
            # Broad search using multiple tools
            logger.info("Starting web search via SerpAPI...")
            search_results = await self._execute_tool(
                "serpapi_internet_search",
                {
                    "query": query,
                    "but_why": 5,
                    "location_requested": "Saline, Michigan, United States",
                },
            )
            tools_used.append("serpapi_internet_search")
            logger.info("Web search completed")

            if search_results and hasattr(search_results, "organic_results"):
                logger.info(
                    "Processing %d search results",
                    len(search_results.organic_results[:3]),
                )
                for i, result in enumerate(
                    search_results.organic_results[:3], 1
                ):
                    logger.debug(
                        "Processing result %d/%d: %s",
                        i,
                        min(3, len(search_results.organic_results)),
                        result.title[:80],
                    )
                    # Extract domain from URL
                    domain = None
                    if result.link:
                        try:
                            from urllib.parse import urlparse

                            parsed = urlparse(result.link)
                            domain = parsed.netloc.replace("www.", "")
                        except Exception:
                            pass

                    # Assign citation ID based on total sources
                    citation_id = (
                        f"[{len(all_sources) + len(sources_found) + 1}]"
                    )

                    source = ResearchSource(
                        source_type="web",
                        url=result.link,
                        title=result.title,
                        content=(result.extracted_content or result.snippet),
                        relevance_score=0.8,
                        citation_id=citation_id,
                        domain=domain,
                    )
                    sources_found.append(source)
                    logger.debug(
                        "Added source: %s with citation %s",
                        source.title[:80],
                        citation_id,
                    )
            else:
                logger.warning("No search results found")

            # Also try news search for current events
            logger.info("Attempting news search for recent information...")
            news_results = await self._execute_tool(
                "serpapi_news_search",
                {
                    "query": query,
                    "but_why": 3,
                    "location_requested": "Saline, Michigan, United States",
                },
            )
            tools_used.append("serpapi_news_search")
            logger.info("News search completed")

            if news_results and hasattr(news_results, "news_results"):
                logger.info(
                    "Processing %d news results",
                    len(news_results.news_results[:2]),
                )
                for i, result in enumerate(news_results.news_results[:2], 1):
                    logger.debug(
                        "Processing news %d/%d: %s",
                        i,
                        min(2, len(news_results.news_results)),
                        result.title[:80],
                    )
                    # Extract domain and date from news results
                    domain = None
                    date = None
                    if result.link:
                        try:
                            from urllib.parse import urlparse

                            parsed = urlparse(result.link)
                            domain = parsed.netloc.replace("www.", "")
                        except Exception:
                            pass
                    if hasattr(result, "date"):
                        date = result.date

                    # Assign citation ID
                    citation_id = (
                        f"[{len(all_sources) + len(sources_found) + 1}]"
                    )

                    source = ResearchSource(
                        source_type="news",
                        url=result.link,
                        title=result.title,
                        content=result.snippet,
                        relevance_score=0.7,
                        citation_id=citation_id,
                        domain=domain,
                        date=date,
                    )
                    sources_found.append(source)
                    logger.debug(
                        "Added news source: %s with citation %s",
                        source.title[:80],
                        citation_id,
                    )
            else:
                logger.info("No news results found")
        elif phase == ResearchPhase.DEEP_DIVE:
            # Initialize extracted_urls if not provided
            if extracted_urls is None:
                extracted_urls = set()

            # Extract content from specific URLs if we have them
            urls_to_extract = [
                s.url
                for s in all_sources
                if s.url
                and s.relevance_score > 0.7
                and s.url not in extracted_urls
            ][:3]

            logger.info(
                "Deep dive phase: extracting content from %d URLs (skipping %d already extracted)",
                len(urls_to_extract),
                len(extracted_urls),
            )

            for i, url in enumerate(urls_to_extract, 1):
                logger.info(
                    "Extracting content %d/%d from: %s",
                    i,
                    len(urls_to_extract),
                    url[:100],
                )
                extract_result = await self._execute_tool(
                    "extract_web_content",
                    {
                        "url": url,
                        "but_why": 5,
                        "location_requested": "Saline, Michigan, United States",
                    },
                )
                tools_used.append("extract_web_content")
                logger.info("Extraction completed for URL %d", i)

                # Mark URL as extracted
                extracted_urls.add(url)

                if extract_result and hasattr(extract_result, "content"):
                    # Find the original source and update it
                    for source in sources_found:
                        if source.url == url:
                            source.content = extract_result.content
                            break

        # Extract key facts from sources
        if sources_found:
            logger.info(
                "→ Extracting key facts from %d sources", len(sources_found)
            )
            total_facts = 0
            for i, source in enumerate(sources_found, 1):
                logger.debug(
                    "  Extracting from source %d/%d: %s",
                    i,
                    len(sources_found),
                    source.title[:60],
                )
                facts = await self._extract_key_facts(source.content, query)
                source.extracted_facts = facts
                total_facts += len(facts)
                logger.debug("  → Extracted %d facts", len(facts))
            logger.info("← Total facts extracted: %d", total_facts)
        else:
            logger.info("No sources to extract facts from")

        # Synthesize findings from this iteration
        logger.info(
            "→ Synthesizing findings from iteration %d", iteration_number
        )
        synthesis = await self._synthesize_iteration_findings(
            query=query,
            sources=sources_found,
            previous_synthesis=previous_synthesis,
        )
        logger.info(
            "← Synthesis complete: %d chars (was %d chars)",
            len(synthesis),
            len(previous_synthesis),
        )

        # Determine if more research is needed
        needs_more = len(sources_found) < 2 or phase != ResearchPhase.SYNTHESIS
        # Generate next questions if needed
        next_questions = []
        if needs_more:
            next_questions = await self._identify_knowledge_gaps(
                query=query, current_sources=sources_found, synthesis=synthesis
            )

        result = ResearchIteration(
            iteration_number=iteration_number,
            phase=phase,
            query=query,
            tools_used=tools_used,
            sources_found=sources_found,
            synthesis=synthesis,
            needs_more_research=needs_more,
            next_questions=next_questions,
        )

        logger.info(
            "└─ Iteration %d results: %d sources, %d tools used, "
            "needs_more=%s",
            iteration_number,
            len(sources_found),
            len(tools_used),
            needs_more,
        )

        return result

    async def _execute_tool(
        self, tool_name: str, params: Dict[str, Any]
    ) -> Any:
        """Execute a tool and return its response"""
        try:
            logger.debug(
                "Executing tool: %s with params: %s", tool_name, params
            )
            # Add timeout warning for long operations
            if tool_name == "extract_web_content":
                url = params.get("url", "unknown")
                logger.info(
                    "Web extraction starting for %s - this may take time...",
                    url[:100],
                )

            start_time = asyncio.get_event_loop().time()

            # Use the async-safe execution method to avoid deadlock
            from tools.registry import get_tool

            tool = get_tool(tool_name)

            # Set a reasonable timeout for tool execution
            timeout = 60.0  # 60 seconds default
            if tool_name == "extract_web_content":
                timeout = 30.0  # 30 seconds for web extraction
            elif tool_name == "serpapi_internet_search":
                timeout = 45.0  # 45 seconds for search

            try:
                if tool and hasattr(tool._controller, "process_async"):
                    # Tool has async implementation - execute directly with timeout
                    logger.debug(
                        "Executing %s with async controller (timeout=%ds)",
                        tool_name,
                        timeout,
                    )
                    raw_result = await asyncio.wait_for(
                        tool._controller.process_async(params), timeout=timeout
                    )
                    result = tool._view.format_response(
                        raw_result, tool.get_response_type()
                    )
                else:
                    # Tool is sync-only - run in thread to avoid blocking with timeout
                    logger.debug(
                        "Executing %s in thread (sync tool, timeout=%ds)",
                        tool_name,
                        timeout,
                    )
                    result = await asyncio.wait_for(
                        asyncio.to_thread(execute_tool, tool_name, params),
                        timeout=timeout,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "Tool %s timed out after %d seconds", tool_name, timeout
                )
                return None

            elapsed_time = asyncio.get_event_loop().time() - start_time

            if result:
                logger.info(
                    "Tool %s completed successfully in %.1f seconds",
                    tool_name,
                    elapsed_time,
                )
            else:
                logger.warning(
                    "Tool %s returned None/empty after %.1f seconds",
                    tool_name,
                    elapsed_time,
                )

            return result
        except asyncio.TimeoutError:
            logger.error("Tool %s timed out", tool_name)
            return None
        except Exception as e:
            logger.error(
                "Error executing tool %s: %s", tool_name, e, exc_info=True
            )
            return None

    async def _extract_key_facts(self, content: str, query: str) -> List[str]:
        """Extract key facts relevant to the query from content"""
        client = self.llm_client_service.get_async_client(self.llm_type)

        prompt = f"""Extract 3-5 key facts from the following content that \
are relevant to the query while maintaining first principles: "{query}"

Content:
{content[:16000]}

Return the facts as a JSON object with a "facts" key containing an array of strings.
Example format: {{"facts": ["fact1", "fact2", "fact3"]}}
Focus on specific, verifiable information while maintaining first principles."""

        # Guided JSON (xgrammar) is mandatory; any failure should raise.
        response = await client.chat.completions.create(
            model=self._get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            extra_body={"nvext": {"guided_json": FACTS_SCHEMA}},
        )

        result = json.loads(response.choices[0].message.content)
        # Handle both dict and list responses
        if isinstance(result, dict):
            return result.get("facts", [])
        elif isinstance(result, list):
            return result
        else:
            logger.warning(
                "Unexpected response type from facts extraction: %s",
                type(result),
            )
            return []

    async def _synthesize_iteration_findings(
        self,
        query: str,
        sources: List[ResearchSource],
        previous_synthesis: str,
    ) -> str:
        """Synthesize findings from current iteration"""
        client = self.llm_client_service.get_async_client(self.llm_type)

        # Prepare source summaries
        source_summaries = []
        for source in sources:
            summary = f"Source {source.citation_id}: {source.title}\n"
            summary += f"Type: {source.source_type}\n"
            if source.domain:
                summary += f"Domain: {source.domain}\n"
            if source.date:
                summary += f"Date: {source.date}\n"
            if source.extracted_facts:
                facts_text = "\n".join(
                    f"- {fact}" for fact in source.extracted_facts
                )
                summary += f"Key facts:\n{facts_text}"
            else:
                summary += f"Content: {source.content[:500]}...\n"
            source_summaries.append(summary)

        prompt = f"""Synthesize the research findings to answer: "{query}"

Previous synthesis:
{previous_synthesis if previous_synthesis else "None"}

New sources found:
{chr(10).join(source_summaries)}

Create a coherent synthesis that:
1. Integrates new information with previous findings
2. Highlights agreements and contradictions
3. Identifies what we know with high confidence
4. Notes any gaps or uncertainties
5. IMPORTANT: Use inline citations in the format [1], [2], etc. when referencing specific information from sources

Each source has a citation ID that you should use when referencing information from that source.
Example: "The research indicates that X is true [1], while another study suggests Y [2]."

Keep the synthesis focused and factual while maintaining first principles."""

        try:
            response = await client.chat.completions.create(
                model=self._get_model_name(),
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            synthesis = strip_think_tags(response.choices[0].message.content)
            # Clean any problematic markdown that might cause display issues
            # synthesis = synthesis.replace(">>>", "").replace("<<<", "")
            # Ensure no repeated quote markers
            # synthesis = re.sub(r">{2,}", "> ", synthesis)
            return synthesis
        except Exception as e:
            logger.error("Error synthesizing findings: %s", e)
            return previous_synthesis

    async def _assess_research_completeness(
        self,
        current_synthesis: str,
        sources: List[ResearchSource],
        target_depth: str,
        iteration_count: int,
    ) -> tuple[bool, ResearchPhase]:
        """Assess if research is complete and determine next phase"""
        client = self.llm_client_service.get_async_client(self.llm_type)

        prompt = f"""Assess the completeness of this research:

Target depth: {target_depth}
Iterations completed: {iteration_count}
Sources consulted: {len(sources)}

Current synthesis:
{current_synthesis}

Evaluate:
1. Is the research comprehensive enough for the target depth?
2. Are there significant knowledge gaps?
3. Do we have diverse, credible sources?
4. Is the confidence level sufficient (medium or high)?

Return a JSON object with:
- needs_more_research: boolean
- next_phase: "deep_dive" or "synthesis"
- confidence: "low", "medium", or "high"
- reasoning: string"""

        # Guided JSON (xgrammar) is mandatory; any failure should raise.
        response = await client.chat.completions.create(
            model=self._get_model_name(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            extra_body={"nvext": {"guided_json": COMPLETENESS_SCHEMA}},
        )

        result = json.loads(response.choices[0].message.content)
        needs_more = result.get("needs_more_research", True)

        # Validate next_phase since we removed enum constraint for xgrammar
        next_phase_str = result.get("next_phase", "deep_dive")
        valid_phases = [
            "initial_search",
            "deep_dive",
            "synthesis",
            "quality_check",
        ]
        if next_phase_str not in valid_phases:
            logger.warning(
                "Invalid next_phase '%s', defaulting to 'deep_dive'",
                next_phase_str,
            )
            next_phase_str = "deep_dive"
        next_phase = ResearchPhase(next_phase_str)

        logger.info(
            "Assessment complete: needs_more=%s, next_phase=%s, confidence=%s",
            needs_more,
            next_phase.value,
            result.get("confidence", "medium"),
        )
        return needs_more, next_phase

    async def _generate_follow_up_questions(
        self, original_query: str, current_synthesis: str, gaps: List[str]
    ) -> List[str]:
        """Generate focused follow-up questions"""
        client = self.llm_client_service.get_async_client(self.llm_type)

        prompt = f"""Generate 2-3 focused follow-up questions for \
deeper research.

Original query: {original_query}

Current understanding:
{current_synthesis}

Known gaps:
{chr(10).join(gaps) if gaps else "None identified"}

Create specific questions that:
1. Address knowledge gaps
2. Seek clarification on ambiguities
3. Explore different perspectives
4. Verify key claims
5. Challenge existing assumptions


Return as a JSON object with a "questions" key containing an array of question strings.
Example format: {{"questions": ["question1", "question2", "question3"]}}"""

        # Guided JSON (xgrammar) is mandatory; any failure should raise.
        response = await client.chat.completions.create(
            model=self._get_model_name(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            extra_body={"nvext": {"guided_json": QUESTIONS_SCHEMA}},
        )

        result_content = response.choices[0].message.content
        try:
            result = json.loads(result_content)
        except json.JSONDecodeError:
            logger.warning(
                "Questions JSON parsing failed, attempting fallback text extraction"
            )
            result = {}

        questions: List[str] = []

        # Handle both dict and list responses
        if isinstance(result, dict):
            questions = result.get("questions", []) if result else []
        elif isinstance(result, list):
            questions = result  # Already a list of questions

        # Fallback – extract sentences ending with a question mark
        if not questions:
            text_for_extraction = result_content.strip()
            potential_qs = re.findall(r"[^\n]+\?", text_for_extraction)
            questions = [
                q.strip() for q in potential_qs if len(q.strip()) > 10
            ]

        # Final sanitisation – deduplicate & cap at 3
        questions = list(dict.fromkeys(questions))[:3]
        logger.info("Generated %d follow-up questions", len(questions))
        return questions

    async def _update_synthesis(
        self,
        previous_synthesis: str,
        new_iteration: ResearchIteration,
        original_query: str,
    ) -> str:
        """Update synthesis with new iteration findings"""
        # The iteration already includes integrated synthesis
        return new_iteration.synthesis

    async def _create_final_synthesis(
        self,
        query: str,
        iterations: List[ResearchIteration],
        all_sources: List[ResearchSource],
        current_synthesis: str,
    ) -> tuple[str, float, List[str]]:
        """Create final comprehensive synthesis"""
        client = self.llm_client_service.get_async_client(self.llm_type)

        # Compile all findings
        all_facts = []
        for source in all_sources:
            all_facts.extend(source.extracted_facts)

        prompt = f"""Create a final, comprehensive research synthesis \
for: "{query}"

Research conducted:
- {len(iterations)} iterations performed
- {len(all_sources)} sources consulted
- {len(all_facts)} key facts extracted

Current synthesis:
{current_synthesis}

Sources available:
{chr(10).join(f"{s.citation_id}: {s.title} ({s.domain or 'Unknown source'})" for s in all_sources if s.citation_id)}

Key facts discovered:
{chr(10).join(f"- {fact}" for fact in all_facts[:20])}  # Top 20 facts

Create a comprehensive answer that:
1. Directly addresses the original query
2. Integrates all significant findings
3. Presents information in a clear, logical structure
4. Acknowledges uncertainties or conflicting information
5. Provides specific examples and evidence
6. Concludes with key takeaways
7. IMPORTANT: Use inline citations [1], [2], etc. when referencing specific information
8. CRITICAL: Cite as many sources as possible - aim to use ALL available sources in your synthesis

Example of proper citation usage:
"Recent studies show that X is effective [1], though some researchers argue Y [2]. Additional evidence from [3] supports this, while [4] provides context."

Remember: Every source has valuable information. Try to cite ALL sources listed above.

Also identify 3-5 key findings as bullet points and list which citations were actually used in your synthesis.

Return as JSON with:
- synthesis: string (the comprehensive answer with inline citations)
- confidence_level: "low", "medium", or "high"
- key_findings: array of strings
- citations_used: array of citation IDs that were actually referenced (e.g., ["[1]", "[3]", "[5]"])"""

        # Guided JSON (xgrammar) is mandatory; any failure should raise.
        response = await client.chat.completions.create(
            model=self._get_model_name(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            extra_body={"nvext": {"guided_json": SYNTHESIS_SCHEMA}},
        )

        result_content = response.choices[0].message.content
        try:
            result = json.loads(result_content)
        except json.JSONDecodeError:
            logger.warning(
                "Final synthesis JSON parsing failed, attempting fallback text extraction"
            )
            result = {}

        raw_synthesis = result.get("synthesis", current_synthesis)
        # Clean the synthesis to prevent markdown corruption
        synthesis = strip_think_tags(raw_synthesis)
        # Remove any problematic markdown characters that might cause display issues
        # synthesis = synthesis.replace('">">">">', "\n\n")
        # Ensure no repeated quote markers
        # synthesis = re.sub(r">{2,}", "\n\n", synthesis)

        # Get confidence level from response, default to medium
        confidence_str = result.get("confidence_level", "medium")
        try:
            confidence_level = ConfidenceLevel(confidence_str.lower())
        except ValueError:
            logger.warning(
                f"Invalid confidence level '{confidence_str}', defaulting to medium"
            )
            confidence_level = ConfidenceLevel.MEDIUM

        key_findings = result.get("key_findings", [])
        citations_used = result.get("citations_used", [])

        return synthesis, confidence_level, key_findings, citations_used

    async def _review_markdown_formatting(self, synthesis: str) -> str:
        """Review and clean up markdown formatting in the final synthesis"""
        client = self.llm_client_service.get_async_client(self.llm_type)

        prompt = f"""Review the following research synthesis and fix any \
markdown formatting issues to ensure it renders correctly.

Focus on:
1. Fixing escaped characters that shouldn't be escaped
2. Ensuring proper line breaks and paragraph spacing
3. Fixing any malformed lists or headers
4. Ensuring citations like [1], [2] are properly formatted as superscript
5. Removing any problematic markdown patterns
6. Ensuring quotes and special characters display correctly

Return ONLY the cleaned text without any explanations or meta-commentary.

Text to review:
{synthesis}"""

        response = await client.chat.completions.create(
            model=self._get_model_name(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        cleaned_synthesis = strip_think_tags(
            response.choices[0].message.content
        )
        return cleaned_synthesis

    def _generate_references(
        self, all_sources: List[ResearchSource], citations_used: List[str]
    ) -> List[str]:
        """Generate a list of references in academic format"""
        references = []

        # Sort sources by citation ID
        cited_sources = [
            s for s in all_sources if s.citation_id in citations_used
        ]
        cited_sources.sort(
            key=lambda x: (
                int(x.citation_id.strip("[]")) if x.citation_id else 999
            )
        )

        for source in cited_sources:
            # Format: [1] Author/Domain. (Date). Title. URL
            ref_parts = [source.citation_id]

            if source.author:
                ref_parts.append(f" {source.author}.")
            elif source.domain:
                ref_parts.append(f" {source.domain}.")
            else:
                ref_parts.append(" Unknown source.")

            if source.date:
                ref_parts.append(f" ({source.date}).")

            ref_parts.append(f" {source.title}.")

            if source.url:
                ref_parts.append(f" Retrieved from {source.url}")

            references.append("".join(ref_parts))

        return references

    def _format_single_reference(self, source: ResearchSource) -> str:
        """Format a single source into a reference string"""
        ref_parts = [source.citation_id if source.citation_id else "[?]"]

        if source.author:
            ref_parts.append(f" {source.author}.")
        elif source.domain:
            ref_parts.append(f" {source.domain}.")
        else:
            ref_parts.append(" Unknown source.")

        if source.date:
            ref_parts.append(f" ({source.date}).")

        ref_parts.append(f" {source.title}.")

        if source.url:
            ref_parts.append(f" Retrieved from {source.url}")

        return "".join(ref_parts)

    def _determine_research_depth(
        self, iterations: int, sources: int, confidence_level: ConfidenceLevel
    ) -> str:
        """Determine the depth of research achieved"""
        if (
            iterations >= 4
            and sources >= 8
            and confidence_level == ConfidenceLevel.HIGH
        ):
            depth = "comprehensive"
        elif (
            iterations >= 3
            and sources >= 5
            and confidence_level
            in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
        ):
            depth = "deep"
        elif (
            iterations >= 2
            and sources >= 3
            and confidence_level == ConfidenceLevel.MEDIUM
        ):
            depth = "moderate"
        else:
            depth = "basic"

        logger.debug(
            "Depth calculation: %s (i=%d, s=%d, c=%s)",
            depth,
            iterations,
            sources,
            confidence_level.value,
        )
        return depth

    async def _identify_knowledge_gaps(
        self, query: str, current_sources: List[ResearchSource], synthesis: str
    ) -> List[str]:
        """Identify gaps in current knowledge"""
        client = self.llm_client_service.get_async_client(self.llm_type)

        prompt = f"""Identify knowledge gaps in the research for: "{query}"

Current synthesis:
{synthesis}

Sources consulted: {len(current_sources)}

What key aspects of the query remain unanswered or need more evidence?
Return as a JSON object with a "gaps" key containing an array of gap descriptions.
Example format: {{"gaps": ["gap1", "gap2", "gap3"]}}"""

        # Guided JSON (xgrammar) is mandatory; any failure should raise.
        response = await client.chat.completions.create(
            model=self._get_model_name(),
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            extra_body={"nvext": {"guided_json": GAPS_SCHEMA}},
        )

        result_content = response.choices[0].message.content
        logger.info("Gaps identification result: %s", result_content)
        try:
            result = json.loads(result_content)
        except json.JSONDecodeError:
            logger.warning(
                "Gaps JSON parsing failed, attempting fallback text extraction"
            )
            result = {}

        # Handle both dict and list responses
        if isinstance(result, dict):
            return result.get("gaps", [])
        elif isinstance(result, list):
            return result
        else:
            logger.warning(
                "Unexpected response type from gaps identification: %s",
                type(result),
            )
            return []

    async def _identify_limitations(
        self, query: str, sources: List[ResearchSource], synthesis: str
    ) -> List[str]:
        """Identify limitations in the research"""
        limitations = []

        # Check source diversity
        source_types = set(s.source_type for s in sources)
        if len(source_types) < 2:
            limitations.append(
                "Limited source diversity - primarily relied on "
                + list(source_types)[0]
                + " sources"
            )

        # Check for recent information
        news_sources = [s for s in sources if s.source_type == "news"]
        if not news_sources and "current" in query.lower():
            limitations.append("Limited access to very recent information")

        # Check for primary sources
        has_primary = any(
            "study" in s.title.lower() or "research" in s.title.lower()
            for s in sources
        )
        if not has_primary:
            limitations.append("No primary research sources consulted")

        # Always acknowledge potential biases
        limitations.append(
            "Potential biases in source selection and synthesis"
        )

        return limitations

    def _get_model_name(self) -> str:
        """Get the appropriate model name based on llm_type"""
        if self.llm_type == "fast":
            return self.config.fast_llm_model_name
        elif self.llm_type == "intelligent":
            return self.config.intelligent_llm_model_name
        else:
            return self.config.llm_model_name


class DeepResearchView(ToolView):
    """View for formatting deep research responses"""

    def format_response(
        self, data: Dict[str, Any], response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format raw data into DeepResearchResponse"""
        try:
            # Check if this is a streaming response
            if data.get("is_streaming") and data.get("content_generator"):
                return StreamingDeepResearchResponse(**data)

            # Convert dict iterations to ResearchIteration objects if needed
            if "iterations" in data:
                iterations = []
                for it in data["iterations"]:
                    if isinstance(it, dict):
                        # Convert sources if needed
                        if "sources_found" in it:
                            sources = []
                            for s in it["sources_found"]:
                                if isinstance(s, dict):
                                    sources.append(ResearchSource(**s))
                                else:
                                    sources.append(s)
                            it["sources_found"] = sources
                        iterations.append(ResearchIteration(**it))
                    else:
                        iterations.append(it)
                data["iterations"] = iterations

            return DeepResearchResponse(**data)
        except Exception as e:
            logger.error("Error formatting deep research response: %s", e)
            return DeepResearchResponse(
                original_query=data.get("original_query", ""),
                final_synthesis=data.get("final_synthesis", "Research failed"),
                success=False,
                error_message=f"Response formatting error: {str(e)}",
                error_code="FORMAT_ERROR",
            )

    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error into DeepResearchResponse"""
        error_code = "UNKNOWN_ERROR"
        if isinstance(error, ValueError):
            error_code = "VALIDATION_ERROR"
        elif isinstance(error, TimeoutError):
            error_code = "TIMEOUT_ERROR"

        return DeepResearchResponse(
            original_query="",
            final_synthesis="",
            success=False,
            error_message=str(error),
            error_code=error_code,
        )


class DeepResearcherTool(BaseTool):
    """
    Deep Research Tool that performs multi-turn, iterative research

    This tool orchestrates multiple research iterations using various tools
    to provide comprehensive, well-researched answers to complex queries.
    """

    def __init__(self):
        super().__init__()
        self.name = "deep_researcher"
        self.description = (
            "Perform deep, multi-turn research on complex topics. "
            "This tool makes multiple iterations using search, extraction, "
            "and analysis tools to provide comprehensive, well-researched "
            "answers with citations. Use for questions requiring in-depth "
            "research, fact-checking, or comprehensive analysis."
        )
        self.supported_contexts = ["research", "analysis", "fact_checking"]
        self.timeout = 120.0  # Longer timeout for deep research

    def _initialize_mvc(self):
        """Initialize MVC components"""
        config = ChatConfig.from_environment()
        self._controller = DeepResearchController(config, self.llm_type)
        self._view = DeepResearchView()

    def get_definition(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The research question or topic to "
                                "investigate deeply"
                            ),
                        },
                        "target_depth": {
                            "type": "string",
                            "enum": [
                                "basic",
                                "moderate",
                                "deep",
                                "comprehensive",
                            ],
                            "description": (
                                "Target depth of research "
                                "(default: moderate)"
                            ),
                            "default": "moderate",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": (
                                "Maximum number of research iterations "
                                "(default: 5)"
                            ),
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                        "but_why": {
                            "type": "integer",
                            "description": (
                                "Confidence level 1-5 that deep "
                                "research is needed"
                            ),
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": ["query", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[BaseToolResponse]:
        """Get the response type for this tool"""
        return DeepResearchResponse

    def supports_streaming(self) -> bool:
        """Indicate that this tool supports streaming responses"""
        return True


# Helper function for backward compatibility
def get_deep_researcher_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for deep researcher"""
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("deep_researcher", DeepResearcherTool)

    # Get the tool instance and return its definition
    tool = get_tool("deep_researcher")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get deep researcher tool definition")
