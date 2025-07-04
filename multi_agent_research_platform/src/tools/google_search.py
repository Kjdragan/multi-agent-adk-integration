"""
Google Search Built-in Tool Integration

Provides comprehensive integration with ADK's built-in Google Search capabilities,
including result processing, relevance scoring, and integration with platform services.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from google.adk.tools.tool_context import ToolContext

from .base import (
    BaseTool, 
    BuiltInToolMixin, 
    ToolResult, 
    ToolType, 
    ToolExecutionStatus,
    ToolAuthConfig,
)
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


@dataclass
class SearchResult:
    """Individual search result from Google Search."""
    title: str
    url: str
    snippet: str
    position: int = 0
    relevance_score: float = 0.0
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract domain from URL."""
        if self.url:
            try:
                parsed = urlparse(self.url)
                self.domain = parsed.netloc
            except:
                self.domain = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "relevance_score": self.relevance_score,
            "domain": self.domain,
            "metadata": self.metadata,
        }


@dataclass
class SearchQuery:
    """Configuration for Google Search query."""
    query: str
    max_results: int = 10
    safe_search: bool = True
    country: Optional[str] = None
    language: Optional[str] = None
    date_restrict: Optional[str] = None  # e.g., "d1" for past day, "w1" for past week
    site_restrict: Optional[str] = None  # Restrict to specific site
    file_type: Optional[str] = None  # e.g., "pdf", "doc"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API call."""
        params = {
            "q": self.query,
            "num": self.max_results,
            "safe": "active" if self.safe_search else "off",
        }
        
        if self.country:
            params["cr"] = f"country{self.country.upper()}"
        if self.language:
            params["lr"] = f"lang_{self.language}"
        if self.date_restrict:
            params["tbs"] = f"qdr:{self.date_restrict}"
        if self.site_restrict:
            params["q"] += f" site:{self.site_restrict}"
        if self.file_type:
            params["q"] += f" filetype:{self.file_type}"
        
        return params


class GoogleSearchTool(BaseTool, BuiltInToolMixin):
    """
    Google Search built-in tool integration.
    
    Provides comprehensive Google Search capabilities with result processing,
    relevance scoring, and integration with platform memory services.
    """
    
    def __init__(self,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 auth_config: Optional[ToolAuthConfig] = None):
        
        super().__init__(
            tool_type=ToolType.GOOGLE_SEARCH,
            tool_name="google_search",
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            auth_config=auth_config,
        )
        
        # Search-specific configuration
        self.default_max_results = 10
        self.relevance_threshold = 0.3
        self.domain_reputation_scores = {
            # High-reputation domains get relevance boost
            "wikipedia.org": 0.9,
            "github.com": 0.8,
            "stackoverflow.com": 0.85,
            "arxiv.org": 0.95,
            "scholar.google.com": 0.9,
            "nature.com": 0.95,
            "ieee.org": 0.9,
        }
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get Google Search tool configuration."""
        return {
            "tool_type": "google_search",
            "built_in": True,
            "requires_auth": self.auth_config is not None,
            "default_max_results": self.default_max_results,
            "relevance_threshold": self.relevance_threshold,
            "supported_parameters": [
                "query", "max_results", "safe_search", "country", 
                "language", "date_restrict", "site_restrict", "file_type"
            ],
        }
    
    def execute_tool(self, context: ToolContext, **kwargs) -> ToolResult:
        """Execute Google Search with comprehensive result processing."""
        start_time = time.time()
        
        # Parse search parameters
        search_query = self._parse_search_parameters(**kwargs)
        
        if not search_query.query.strip():
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error="Empty search query provided",
            )
        
        try:
            # Get enhanced context for full capabilities
            enhanced_context = self.get_enhanced_context(context)
            enhanced_context.start_execution()
            
            # Execute search using ADK built-in tool
            builtin_search = self.get_builtin_tool_instance(context, "google_search")
            if not builtin_search:
                return ToolResult(
                    tool_type=self.tool_type,
                    status=ToolExecutionStatus.FAILED,
                    error="Google Search built-in tool not available",
                )
            
            # Log search execution
            self.log_tool_usage("search", search_query.to_dict())
            
            # Execute search
            search_params = search_query.to_dict()
            raw_results = builtin_search.search(**search_params)
            
            # Process and enhance results
            processed_results = self._process_search_results(
                raw_results, 
                search_query,
                enhanced_context
            )
            
            # Store results in memory if memory service available
            if self.memory_service and processed_results:
                self._store_search_results_in_memory(
                    search_query.query,
                    processed_results,
                    enhanced_context
                )
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.COMPLETED,
                data={
                    "query": search_query.query,
                    "results": [result.to_dict() for result in processed_results],
                    "result_count": len(processed_results),
                    "search_parameters": search_query.to_dict(),
                },
                metadata={
                    "query_length": len(search_query.query),
                    "max_results_requested": search_query.max_results,
                    "actual_results_returned": len(processed_results),
                    "average_relevance_score": self._calculate_average_relevance(processed_results),
                    "unique_domains": len(set(r.domain for r in processed_results)),
                },
                execution_time_ms=execution_time,
            )
            
            # Complete execution tracking
            enhanced_context.complete_execution(result)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Google Search execution failed: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg, 
                                query=search_query.query,
                                execution_time_ms=execution_time)
            
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error=error_msg,
                execution_time_ms=execution_time,
            )
    
    def search(self, 
              query: str,
              context: ToolContext,
              max_results: int = 10,
              **search_options) -> List[SearchResult]:
        """
        Convenient search method with direct result access.
        
        Args:
            query: Search query string
            context: Tool execution context
            max_results: Maximum number of results to return
            **search_options: Additional search options
            
        Returns:
            List of processed search results
        """
        result = self.execute_with_context(
            context,
            query=query,
            max_results=max_results,
            **search_options
        )
        
        if result.success and "results" in result.tool_result.data:
            return [
                SearchResult(**result_data) 
                for result_data in result.tool_result.data["results"]
            ]
        
        return []
    
    def search_with_memory_integration(self,
                                     query: str,
                                     context: ToolContext,
                                     memory_queries: Optional[List[str]] = None,
                                     max_results: int = 10,
                                     **search_options) -> Dict[str, Any]:
        """
        Search with automatic memory integration and cross-referencing.
        
        Args:
            query: Search query string
            context: Tool execution context  
            memory_queries: Related memory queries to run
            max_results: Maximum search results
            **search_options: Additional search options
            
        Returns:
            Dictionary with search results and memory correlations
        """
        # Default memory queries based on search query
        if memory_queries is None:
            memory_queries = [
                query,
                f"related to {query}",
                f"similar to {query}",
            ]
        
        result = self.execute_with_context(
            context,
            memory_queries=memory_queries,
            query=query,
            max_results=max_results,
            **search_options
        )
        
        return {
            "search_results": result.tool_result.data if result.success else None,
            "memory_results": result.memory_info.get("search_results", {}),
            "correlations": self._find_search_memory_correlations(
                result.tool_result.data if result.success else {},
                result.memory_info.get("search_results", {})
            ),
            "overall_success": result.success,
            "execution_info": result.to_dict(),
        }
    
    def _parse_search_parameters(self, **kwargs) -> SearchQuery:
        """Parse and validate search parameters."""
        return SearchQuery(
            query=kwargs.get("query", ""),
            max_results=min(kwargs.get("max_results", self.default_max_results), 50),
            safe_search=kwargs.get("safe_search", True),
            country=kwargs.get("country"),
            language=kwargs.get("language"),
            date_restrict=kwargs.get("date_restrict"),
            site_restrict=kwargs.get("site_restrict"),
            file_type=kwargs.get("file_type"),
        )
    
    def _process_search_results(self, 
                              raw_results: Any, 
                              search_query: SearchQuery,
                              context) -> List[SearchResult]:
        """Process and enhance raw search results."""
        if not raw_results:
            return []
        
        processed_results = []
        
        try:
            # Handle different possible result formats from ADK
            if hasattr(raw_results, 'items'):
                items = raw_results.items
            elif isinstance(raw_results, list):
                items = raw_results
            elif isinstance(raw_results, dict) and 'items' in raw_results:
                items = raw_results['items']
            else:
                if self.logger:
                    self.logger.warning(f"Unexpected search result format: {type(raw_results)}")
                return []
            
            for i, item in enumerate(items[:search_query.max_results]):
                try:
                    # Extract basic information
                    title = item.get('title', 'No title')
                    url = item.get('link', item.get('url', ''))
                    snippet = item.get('snippet', item.get('description', ''))
                    
                    # Create search result
                    result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        position=i + 1,
                    )
                    
                    # Calculate relevance score
                    result.relevance_score = self._calculate_relevance_score(
                        result, search_query
                    )
                    
                    # Add metadata
                    result.metadata = {
                        "raw_data_keys": list(item.keys()),
                        "has_image": 'image' in item or 'thumbnail' in item,
                        "content_type": item.get('mime', 'text/html'),
                    }
                    
                    processed_results.append(result)
                    
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error processing search result {i}: {e}")
                    continue
            
            # Sort by relevance score (highest first)
            processed_results.sort(key=lambda r: r.relevance_score, reverse=True)
            
            # Filter by relevance threshold
            processed_results = [
                r for r in processed_results 
                if r.relevance_score >= self.relevance_threshold
            ]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing search results: {e}")
        
        return processed_results
    
    def _calculate_relevance_score(self, 
                                 result: SearchResult, 
                                 search_query: SearchQuery) -> float:
        """Calculate relevance score for search result."""
        score = 0.0
        query_terms = search_query.query.lower().split()
        
        # Title matching (40% weight)
        title_lower = result.title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        title_score = (title_matches / len(query_terms)) * 0.4
        score += title_score
        
        # Snippet matching (30% weight)
        snippet_lower = result.snippet.lower()
        snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
        snippet_score = (snippet_matches / len(query_terms)) * 0.3
        score += snippet_score
        
        # Domain reputation (20% weight)
        domain_score = self.domain_reputation_scores.get(result.domain, 0.5) * 0.2
        score += domain_score
        
        # Position bonus (10% weight) - earlier results get slight boost
        position_score = max(0, (11 - result.position) / 10) * 0.1
        score += position_score
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_average_relevance(self, results: List[SearchResult]) -> float:
        """Calculate average relevance score across results."""
        if not results:
            return 0.0
        return sum(r.relevance_score for r in results) / len(results)
    
    def _store_search_results_in_memory(self,
                                      query: str,
                                      results: List[SearchResult],
                                      context) -> None:
        """Store search results in memory for future reference."""
        try:
            # Create memory entries for high-relevance results
            high_relevance_results = [
                r for r in results 
                if r.relevance_score > 0.7
            ]
            
            for result in high_relevance_results[:5]:  # Store top 5
                memory_text = f"Search result for '{query}': {result.title}\n{result.snippet}\nURL: {result.url}"
                
                # This would integrate with the actual memory service
                if hasattr(context, 'store_memory'):
                    context.store_memory(
                        text=memory_text,
                        metadata={
                            "type": "search_result",
                            "query": query,
                            "url": result.url,
                            "domain": result.domain,
                            "relevance_score": result.relevance_score,
                        }
                    )
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing search results in memory: {e}")
    
    def _find_search_memory_correlations(self,
                                       search_data: Dict[str, Any],
                                       memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find correlations between search results and memory."""
        correlations = {
            "related_memory_items": 0,
            "overlapping_domains": [],
            "common_terms": [],
        }
        
        try:
            if not search_data.get("results") or not memory_data:
                return correlations
            
            search_domains = set()
            search_terms = set()
            
            # Extract domains and terms from search results
            for result in search_data["results"]:
                if "domain" in result:
                    search_domains.add(result["domain"])
                
                # Extract terms from title and snippet
                title_terms = result.get("title", "").lower().split()
                snippet_terms = result.get("snippet", "").lower().split()
                search_terms.update(title_terms + snippet_terms)
            
            # Find correlations with memory results
            memory_domains = set()
            memory_terms = set()
            
            for query, query_result in memory_data.items():
                for memory_item in query_result.get("results", []):
                    # Extract terms and domains from memory items
                    text = memory_item.get("text", "").lower()
                    memory_terms.update(text.split())
                    
                    # Extract domain from metadata if available
                    metadata = memory_item.get("metadata", {})
                    if "domain" in metadata:
                        memory_domains.add(metadata["domain"])
            
            # Calculate correlations
            correlations["overlapping_domains"] = list(search_domains & memory_domains)
            correlations["common_terms"] = list(search_terms & memory_terms)[:10]  # Top 10
            correlations["related_memory_items"] = len([
                item for query_result in memory_data.values()
                for item in query_result.get("results", [])
                if item.get("score", 0) > 0.7
            ])
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating search-memory correlations: {e}")
        
        return correlations