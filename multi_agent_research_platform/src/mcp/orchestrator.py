"""
MCP Orchestrator for Cross-Service Integration

Provides sophisticated orchestration across multiple MCP servers including
Perplexity, Tavily, Brave, and Omnisearch with intelligent routing, result
aggregation, and cross-validation capabilities.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

from .base import MCPServer, MCPOperationType, MCPServerResult
from .servers.perplexity import PerplexityServer, PerplexitySearchResult
from .servers.tavily import TavilyServer, TavilySearchResult
from .servers.brave import BraveServer, BraveSearchResult
from .servers.omnisearch import OmnisearchServer, OmnisearchResult
from ..context import ToolContextPattern, MemoryAccessPattern
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


class SearchStrategy(str, Enum):
    """Search strategies for MCP orchestration."""
    SINGLE_BEST = "single_best"           # Use best single source
    PARALLEL_ALL = "parallel_all"         # Query all sources in parallel
    SEQUENTIAL = "sequential"             # Query sources sequentially
    ADAPTIVE = "adaptive"                 # Adapt based on query type
    HYBRID_VALIDATION = "hybrid_validation"  # Cross-validate results
    COST_OPTIMIZED = "cost_optimized"    # Optimize for cost efficiency
    SPEED_OPTIMIZED = "speed_optimized"  # Optimize for speed
    QUALITY_OPTIMIZED = "quality_optimized"  # Optimize for result quality


class QueryType(str, Enum):
    """Types of queries for intelligent routing."""
    FACTUAL = "factual"                   # Factual questions
    RESEARCH = "research"                 # Research queries
    NEWS = "news"                         # Current news
    ACADEMIC = "academic"                 # Academic research
    TECHNICAL = "technical"               # Technical documentation
    COMMERCIAL = "commercial"             # Commercial/product info
    LOCAL = "local"                       # Local/geographic queries
    CREATIVE = "creative"                 # Creative/subjective queries


@dataclass
class SearchContext:
    """Context information for search orchestration."""
    query: str
    query_type: Optional[QueryType] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    cost_budget: Optional[float] = None
    time_budget: Optional[float] = None
    quality_threshold: float = 0.7
    required_sources: Optional[List[str]] = None
    excluded_sources: Optional[List[str]] = None
    language: str = "en"
    region: str = "US"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "query_type": self.query_type.value if self.query_type else None,
            "user_preferences": self.user_preferences,
            "cost_budget": self.cost_budget,
            "time_budget": self.time_budget,
            "quality_threshold": self.quality_threshold,
            "required_sources": self.required_sources,
            "excluded_sources": self.excluded_sources,
            "language": self.language,
            "region": self.region,
        }


@dataclass
class MultiSourceSearchResult:
    """Aggregated result from multiple MCP sources."""
    query: str
    strategy_used: SearchStrategy
    perplexity_result: Optional[PerplexitySearchResult] = None
    tavily_result: Optional[TavilySearchResult] = None
    brave_result: Optional[BraveSearchResult] = None
    omnisearch_result: Optional[OmnisearchResult] = None
    aggregated_results: List[Dict[str, Any]] = field(default_factory=list)
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    consensus_answer: str = ""
    confidence_score: float = 0.0
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    total_execution_time_ms: float = 0.0
    
    @property
    def total_results(self) -> int:
        """Total number of results across all sources."""
        total = 0
        for result in [self.perplexity_result, self.tavily_result, self.brave_result, self.omnisearch_result]:
            if hasattr(result, 'total_results'):
                total += result.total_results
            elif hasattr(result, 'source_count'):
                total += result.source_count
            elif hasattr(result, 'web_results'):
                total += len(result.web_results)
        return total
    
    @property
    def sources_used(self) -> List[str]:
        """List of sources that returned results."""
        sources = []
        if self.perplexity_result and getattr(self.perplexity_result, 'answer', ''):
            sources.append("perplexity")
        if self.tavily_result and getattr(self.tavily_result, 'sources', []):
            sources.append("tavily")
        if self.brave_result and getattr(self.brave_result, 'web_results', []):
            sources.append("brave")
        if self.omnisearch_result and getattr(self.omnisearch_result, 'results', []):
            sources.append("omnisearch")
        return sources
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "strategy_used": self.strategy_used.value,
            "perplexity_result": self.perplexity_result.to_dict() if self.perplexity_result else None,
            "tavily_result": self.tavily_result.to_dict() if self.tavily_result else None,
            "brave_result": self.brave_result.to_dict() if self.brave_result else None,
            "omnisearch_result": self.omnisearch_result.to_dict() if self.omnisearch_result else None,
            "aggregated_results": self.aggregated_results,
            "cross_validation": self.cross_validation,
            "consensus_answer": self.consensus_answer,
            "confidence_score": self.confidence_score,
            "execution_metadata": self.execution_metadata,
            "total_execution_time_ms": self.total_execution_time_ms,
            "total_results": self.total_results,
            "sources_used": self.sources_used,
        }


class MCPOrchestrator:
    """
    MCP Orchestrator for sophisticated cross-service integration.
    
    Provides intelligent routing, result aggregation, cross-validation,
    and optimization across multiple MCP servers.
    """
    
    def __init__(self,
                 perplexity_server: Optional[PerplexityServer] = None,
                 tavily_server: Optional[TavilyServer] = None,
                 brave_server: Optional[BraveServer] = None,
                 omnisearch_server: Optional[OmnisearchServer] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        self.perplexity = perplexity_server
        self.tavily = tavily_server
        self.brave = brave_server
        self.omnisearch = omnisearch_server
        
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Context patterns for integration
        self.tool_pattern = ToolContextPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        self.memory_pattern = MemoryAccessPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        # Available servers
        self.available_servers = {
            "perplexity": self.perplexity,
            "tavily": self.tavily,
            "brave": self.brave,
            "omnisearch": self.omnisearch,
        }
        
        # Query type routing preferences
        self.routing_preferences = {
            QueryType.FACTUAL: ["perplexity", "brave", "tavily"],
            QueryType.RESEARCH: ["perplexity", "tavily", "omnisearch"],
            QueryType.NEWS: ["tavily", "brave", "perplexity"],
            QueryType.ACADEMIC: ["perplexity", "omnisearch", "tavily"],
            QueryType.TECHNICAL: ["perplexity", "omnisearch", "brave"],
            QueryType.COMMERCIAL: ["brave", "tavily", "omnisearch"],
            QueryType.LOCAL: ["brave", "tavily"],
            QueryType.CREATIVE: ["perplexity", "omnisearch"],
        }
        
        # Strategy configurations
        self.strategy_configs = {
            SearchStrategy.SINGLE_BEST: {"max_sources": 1, "parallel": False},
            SearchStrategy.PARALLEL_ALL: {"max_sources": 4, "parallel": True},
            SearchStrategy.SEQUENTIAL: {"max_sources": 4, "parallel": False},
            SearchStrategy.ADAPTIVE: {"max_sources": 3, "parallel": True},
            SearchStrategy.HYBRID_VALIDATION: {"max_sources": 3, "parallel": True, "cross_validate": True},
            SearchStrategy.COST_OPTIMIZED: {"max_sources": 2, "parallel": False, "cost_aware": True},
            SearchStrategy.SPEED_OPTIMIZED: {"max_sources": 2, "parallel": True, "timeout": 10},
            SearchStrategy.QUALITY_OPTIMIZED: {"max_sources": 4, "parallel": True, "quality_threshold": 0.8},
        }
    
    async def search(self, 
                    search_context: SearchContext,
                    strategy: SearchStrategy = SearchStrategy.ADAPTIVE) -> MultiSourceSearchResult:
        """
        Execute orchestrated search across multiple MCP sources.
        
        Args:
            search_context: Search context with query and preferences
            strategy: Search strategy to use
            
        Returns:
            Aggregated results from multiple sources
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(f"Starting orchestrated search: {search_context.query[:100]}")
        
        # Determine optimal sources based on query type and strategy
        selected_sources = self._select_sources(search_context, strategy)
        
        # Execute search based on strategy
        source_results = await self._execute_strategy(search_context, strategy, selected_sources)
        
        # Aggregate and process results
        aggregated_result = await self._aggregate_results(
            search_context.query, strategy, source_results, start_time
        )
        
        # Store orchestrated results in memory
        if self.memory_service:
            await self._store_orchestrated_results(search_context, aggregated_result)
        
        if self.logger:
            self.logger.info(f"Completed orchestrated search in {aggregated_result.total_execution_time_ms:.2f}ms")
        
        return aggregated_result
    
    async def research_workflow(self,
                              topic: str,
                              research_depth: str = "comprehensive",
                              include_perspectives: bool = True) -> Dict[str, MultiSourceSearchResult]:
        """
        Execute comprehensive research workflow using multiple strategies.
        
        Args:
            topic: Research topic
            research_depth: Depth of research ("basic", "comprehensive", "exhaustive")
            include_perspectives: Whether to include multi-perspective analysis
            
        Returns:
            Dictionary of research phases and their results
        """
        research_results = {}
        
        # Phase 1: Initial exploration
        initial_context = SearchContext(
            query=f"overview of {topic}",
            query_type=QueryType.RESEARCH,
            quality_threshold=0.8,
        )
        
        initial_result = await self.search(initial_context, SearchStrategy.QUALITY_OPTIMIZED)
        research_results["initial_exploration"] = initial_result
        
        # Phase 2: Deep dive research
        deep_context = SearchContext(
            query=f"comprehensive analysis {topic}",
            query_type=QueryType.ACADEMIC,
            quality_threshold=0.9,
        )
        
        deep_result = await self.search(deep_context, SearchStrategy.HYBRID_VALIDATION)
        research_results["deep_analysis"] = deep_result
        
        # Phase 3: Current developments
        news_context = SearchContext(
            query=f"recent developments {topic}",
            query_type=QueryType.NEWS,
            time_budget=15.0,
        )
        
        news_result = await self.search(news_context, SearchStrategy.SPEED_OPTIMIZED)
        research_results["current_developments"] = news_result
        
        # Phase 4: Multi-perspective analysis (if requested)
        if include_perspectives:
            perspectives = ["academic", "industry", "policy", "technical"]
            perspective_results = {}
            
            for perspective in perspectives:
                perspective_context = SearchContext(
                    query=f"{topic} from {perspective} perspective",
                    query_type=QueryType.RESEARCH,
                )
                
                perspective_result = await self.search(perspective_context, SearchStrategy.ADAPTIVE)
                perspective_results[perspective] = perspective_result
            
            research_results["perspectives"] = perspective_results
        
        return research_results
    
    async def fact_check_workflow(self,
                                claim: str,
                                verification_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute fact-checking workflow using cross-validation.
        
        Args:
            claim: Claim to fact-check
            verification_sources: Preferred verification sources
            
        Returns:
            Fact-checking results with verification status
        """
        # Search for information about the claim
        fact_check_context = SearchContext(
            query=f"fact check verify: {claim}",
            query_type=QueryType.FACTUAL,
            quality_threshold=0.9,
            required_sources=verification_sources,
        )
        
        verification_result = await self.search(
            fact_check_context, 
            SearchStrategy.HYBRID_VALIDATION
        )
        
        # Analyze cross-validation results
        fact_check_analysis = {
            "claim": claim,
            "verification_result": verification_result.to_dict(),
            "consensus_found": verification_result.confidence_score > 0.7,
            "supporting_sources": len(verification_result.sources_used),
            "cross_validation": verification_result.cross_validation,
            "fact_check_status": self._determine_fact_check_status(verification_result),
            "reliability_score": verification_result.confidence_score,
        }
        
        return fact_check_analysis
    
    def _select_sources(self, 
                       search_context: SearchContext, 
                       strategy: SearchStrategy) -> List[str]:
        """Select optimal sources based on context and strategy."""
        strategy_config = self.strategy_configs.get(strategy, {})
        max_sources = strategy_config.get("max_sources", 2)
        
        # Start with available sources
        available = [name for name, server in self.available_servers.items() if server is not None]
        
        # Apply user preferences
        if search_context.required_sources:
            available = [s for s in available if s in search_context.required_sources]
        
        if search_context.excluded_sources:
            available = [s for s in available if s not in search_context.excluded_sources]
        
        # Apply query type routing if detected
        if search_context.query_type:
            preferred = self.routing_preferences.get(search_context.query_type, available)
            # Reorder available sources by preference
            available = [s for s in preferred if s in available] + [s for s in available if s not in preferred]
        
        # Apply strategy-specific selection
        if strategy == SearchStrategy.SINGLE_BEST:
            return available[:1]
        elif strategy == SearchStrategy.COST_OPTIMIZED:
            # Prefer free/cheaper sources
            cost_order = ["brave", "tavily", "perplexity", "omnisearch"]
            available = [s for s in cost_order if s in available] + [s for s in available if s not in cost_order]
        elif strategy == SearchStrategy.SPEED_OPTIMIZED:
            # Prefer faster sources
            speed_order = ["brave", "tavily", "perplexity", "omnisearch"]
            available = [s for s in speed_order if s in available] + [s for s in available if s not in speed_order]
        elif strategy == SearchStrategy.QUALITY_OPTIMIZED:
            # Prefer higher quality sources
            quality_order = ["perplexity", "omnisearch", "tavily", "brave"]
            available = [s for s in quality_order if s in available] + [s for s in available if s not in quality_order]
        
        return available[:max_sources]
    
    async def _execute_strategy(self,
                              search_context: SearchContext,
                              strategy: SearchStrategy,
                              selected_sources: List[str]) -> Dict[str, Any]:
        """Execute search strategy across selected sources."""
        strategy_config = self.strategy_configs.get(strategy, {})
        parallel = strategy_config.get("parallel", True)
        timeout = strategy_config.get("timeout", 30)
        
        source_results = {}
        
        if parallel:
            # Execute searches in parallel
            tasks = []
            for source_name in selected_sources:
                server = self.available_servers[source_name]
                if server:
                    task = asyncio.create_task(
                        self._search_source_with_timeout(
                            source_name, server, search_context.query, timeout
                        )
                    )
                    tasks.append((source_name, task))
            
            # Wait for all tasks to complete
            for source_name, task in tasks:
                try:
                    result = await task
                    source_results[source_name] = result
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Source {source_name} failed: {e}")
                    source_results[source_name] = None
        
        else:
            # Execute searches sequentially
            for source_name in selected_sources:
                server = self.available_servers[source_name]
                if server:
                    try:
                        result = await self._search_source_with_timeout(
                            source_name, server, search_context.query, timeout
                        )
                        source_results[source_name] = result
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Source {source_name} failed: {e}")
                        source_results[source_name] = None
        
        return source_results
    
    async def _search_source_with_timeout(self,
                                        source_name: str,
                                        server: MCPServer,
                                        query: str,
                                        timeout: int) -> Optional[Any]:
        """Search a specific source with timeout protection."""
        try:
            if source_name == "perplexity" and isinstance(server, PerplexityServer):
                return await asyncio.wait_for(server.search(query), timeout=timeout)
            elif source_name == "tavily" and isinstance(server, TavilyServer):
                return await asyncio.wait_for(server.search(query), timeout=timeout)
            elif source_name == "brave" and isinstance(server, BraveServer):
                return await asyncio.wait_for(server.search(query), timeout=timeout)
            elif source_name == "omnisearch" and isinstance(server, OmnisearchServer):
                return await asyncio.wait_for(server.search(query), timeout=timeout)
            else:
                return None
        except asyncio.TimeoutError:
            if self.logger:
                self.logger.warning(f"Source {source_name} search timed out")
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Source {source_name} search error: {e}")
            return None
    
    async def _aggregate_results(self,
                               query: str,
                               strategy: SearchStrategy,
                               source_results: Dict[str, Any],
                               start_time: float) -> MultiSourceSearchResult:
        """Aggregate results from multiple sources."""
        total_execution_time = (time.time() - start_time) * 1000
        
        # Create multi-source result
        multi_result = MultiSourceSearchResult(
            query=query,
            strategy_used=strategy,
            total_execution_time_ms=total_execution_time,
        )
        
        # Assign individual results
        if "perplexity" in source_results:
            multi_result.perplexity_result = source_results["perplexity"]
        if "tavily" in source_results:
            multi_result.tavily_result = source_results["tavily"]
        if "brave" in source_results:
            multi_result.brave_result = source_results["brave"]
        if "omnisearch" in source_results:
            multi_result.omnisearch_result = source_results["omnisearch"]
        
        # Aggregate all results
        all_results = []
        
        # Extract results from each source
        if multi_result.perplexity_result:
            all_results.append({
                "source": "perplexity",
                "answer": multi_result.perplexity_result.answer,
                "citations": multi_result.perplexity_result.citations,
                "confidence": 0.9,  # Perplexity is generally high quality
            })
        
        if multi_result.tavily_result:
            for source in multi_result.tavily_result.sources:
                all_results.append({
                    "source": "tavily",
                    "title": source.title,
                    "content": source.content,
                    "url": source.url,
                    "score": source.score,
                    "confidence": source.score,
                })
        
        if multi_result.brave_result:
            for result in multi_result.brave_result.web_results:
                all_results.append({
                    "source": "brave",
                    "title": result.title,
                    "content": result.description,
                    "url": result.url,
                    "confidence": 0.7,  # Brave web results are generally reliable
                })
        
        if multi_result.omnisearch_result:
            for result in multi_result.omnisearch_result.results:
                all_results.append({
                    "source": "omnisearch",
                    "title": result.title,
                    "content": result.content,
                    "url": result.url,
                    "score": result.score,
                    "confidence": result.score,
                })
        
        multi_result.aggregated_results = all_results
        
        # Perform cross-validation if strategy requires it
        if strategy == SearchStrategy.HYBRID_VALIDATION:
            multi_result.cross_validation = self._cross_validate_results(all_results)
            multi_result.confidence_score = multi_result.cross_validation.get("confidence", 0.5)
            multi_result.consensus_answer = multi_result.cross_validation.get("consensus", "")
        else:
            # Simple confidence scoring
            if all_results:
                avg_confidence = sum(r.get("confidence", 0.5) for r in all_results) / len(all_results)
                multi_result.confidence_score = avg_confidence
        
        # Generate consensus answer
        if not multi_result.consensus_answer:
            multi_result.consensus_answer = self._generate_consensus_answer(all_results)
        
        # Add execution metadata
        multi_result.execution_metadata = {
            "sources_attempted": list(source_results.keys()),
            "sources_successful": [k for k, v in source_results.items() if v is not None],
            "total_results_found": len(all_results),
            "strategy_config": self.strategy_configs.get(strategy, {}),
        }
        
        return multi_result
    
    def _cross_validate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-validate results across sources for consistency."""
        if len(results) < 2:
            return {"confidence": 0.5, "consensus": "", "validation_notes": "Insufficient sources for validation"}
        
        # Simple cross-validation based on content similarity
        content_similarity_scores = []
        
        for i, result1 in enumerate(results):
            for j, result2 in enumerate(results[i+1:], i+1):
                content1 = result1.get("content", result1.get("answer", ""))
                content2 = result2.get("content", result2.get("answer", ""))
                
                similarity = self._calculate_content_similarity(content1, content2)
                content_similarity_scores.append(similarity)
        
        # Calculate validation metrics
        avg_similarity = sum(content_similarity_scores) / len(content_similarity_scores) if content_similarity_scores else 0
        
        # Determine consensus
        if avg_similarity > 0.7:
            validation_status = "high_consensus"
            confidence = 0.9
        elif avg_similarity > 0.5:
            validation_status = "moderate_consensus"
            confidence = 0.7
        else:
            validation_status = "low_consensus"
            confidence = 0.4
        
        return {
            "confidence": confidence,
            "consensus": validation_status,
            "average_similarity": avg_similarity,
            "validation_notes": f"Cross-validated {len(results)} sources",
            "similarity_scores": content_similarity_scores,
        }
    
    def _generate_consensus_answer(self, results: List[Dict[str, Any]]) -> str:
        """Generate consensus answer from multiple results."""
        if not results:
            return ""
        
        # Prefer answers from AI sources (Perplexity)
        ai_answers = [r.get("answer", "") for r in results if r.get("source") == "perplexity" and r.get("answer")]
        if ai_answers:
            return ai_answers[0]  # Take the first AI answer
        
        # Fall back to highest scoring content
        scored_results = [r for r in results if r.get("score", 0) > 0]
        if scored_results:
            best_result = max(scored_results, key=lambda r: r.get("score", 0))
            return best_result.get("content", best_result.get("title", ""))[:500]
        
        # Fall back to first available content
        for result in results:
            content = result.get("content", result.get("title", ""))
            if content:
                return content[:500]
        
        return "No consensus answer available"
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _determine_fact_check_status(self, verification_result: MultiSourceSearchResult) -> str:
        """Determine fact-check status based on verification results."""
        confidence = verification_result.confidence_score
        
        if confidence >= 0.8:
            return "verified"
        elif confidence >= 0.6:
            return "likely_accurate"
        elif confidence >= 0.4:
            return "uncertain"
        else:
            return "likely_inaccurate"
    
    async def _store_orchestrated_results(self,
                                        search_context: SearchContext,
                                        result: MultiSourceSearchResult) -> None:
        """Store orchestrated search results in memory."""
        try:
            memory_text = f"""
Orchestrated Search Result:
Query: {search_context.query}
Strategy: {result.strategy_used.value}
Sources Used: {', '.join(result.sources_used)}

Consensus Answer:
{result.consensus_answer}

Confidence Score: {result.confidence_score:.2f}
Total Results: {result.total_results}
Execution Time: {result.total_execution_time_ms:.2f}ms

Cross-Validation: {result.cross_validation.get('consensus', 'N/A')}
            """.strip()
            
            if hasattr(self.memory_service, 'store'):
                await self.memory_service.store(
                    text=memory_text,
                    metadata={
                        "type": "orchestrated_search",
                        "query": search_context.query,
                        "strategy": result.strategy_used.value,
                        "sources_used": result.sources_used,
                        "confidence_score": result.confidence_score,
                        "total_results": result.total_results,
                    }
                )
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing orchestrated results in memory: {e}")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get status of all connected MCP servers."""
        status = {
            "available_servers": {},
            "routing_preferences": {qt.value: prefs for qt, prefs in self.routing_preferences.items()},
            "strategy_configs": {st.value: config for st, config in self.strategy_configs.items()},
        }
        
        for name, server in self.available_servers.items():
            if server:
                try:
                    server_status = asyncio.create_task(server.health_check())
                    status["available_servers"][name] = "available"
                except:
                    status["available_servers"][name] = "unavailable"
            else:
                status["available_servers"][name] = "not_configured"
        
        return status