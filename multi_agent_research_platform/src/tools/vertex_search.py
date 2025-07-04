"""
Vertex AI Search Built-in Tool Integration

Provides comprehensive integration with ADK's built-in Vertex AI Search capabilities,
including vector embeddings, semantic search, enterprise search, and result processing.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

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


class SearchType(str, Enum):
    """Types of Vertex AI searches."""
    SEMANTIC = "semantic"          # Vector-based semantic search
    KEYWORD = "keyword"           # Traditional keyword search
    HYBRID = "hybrid"             # Combination of semantic and keyword
    ENTERPRISE = "enterprise"     # Enterprise search with access controls


class ContentType(str, Enum):
    """Types of content that can be searched."""
    DOCUMENTS = "documents"
    WEBSITES = "websites"
    STRUCTURED_DATA = "structured_data"
    MEDIA = "media"
    ALL = "all"


@dataclass
class VertexSearchConfig:
    """Configuration for Vertex AI Search."""
    search_type: SearchType = SearchType.SEMANTIC
    content_type: ContentType = ContentType.ALL
    max_results: int = 10
    search_engine_id: Optional[str] = None
    data_store_id: Optional[str] = None
    serving_config: Optional[str] = None
    filter_expression: Optional[str] = None
    order_by: Optional[str] = None
    facet_specs: Optional[List[Dict[str, Any]]] = None
    boost_specs: Optional[List[Dict[str, Any]]] = None
    similarity_threshold: float = 0.7
    include_snippets: bool = True
    snippet_length: int = 200
    user_labels: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API call."""
        config = {
            "search_type": self.search_type.value,
            "content_type": self.content_type.value,
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            "include_snippets": self.include_snippets,
            "snippet_length": self.snippet_length,
        }
        
        # Add optional parameters
        for attr in ["search_engine_id", "data_store_id", "serving_config", 
                    "filter_expression", "order_by", "facet_specs", "boost_specs", "user_labels"]:
            value = getattr(self, attr)
            if value is not None:
                config[attr] = value
        
        return config


@dataclass
class VertexSearchResult:
    """Individual search result from Vertex AI Search."""
    document_id: str
    title: str
    content: str
    snippet: str = ""
    similarity_score: float = 0.0
    rank: int = 0
    url: Optional[str] = None
    content_type: str = "document"
    facets: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    
    @property
    def relevance_category(self) -> str:
        """Categorize relevance based on similarity score."""
        if self.similarity_score >= 0.9:
            return "highly_relevant"
        elif self.similarity_score >= 0.7:
            return "relevant"
        elif self.similarity_score >= 0.5:
            return "somewhat_relevant"
        else:
            return "low_relevance"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "snippet": self.snippet,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "url": self.url,
            "content_type": self.content_type,
            "facets": self.facets,
            "metadata": self.metadata,
            "relevance_category": self.relevance_category,
            "has_embeddings": self.embeddings is not None,
        }


class VertexSearchTool(BaseTool, BuiltInToolMixin):
    """
    Vertex AI Search built-in tool integration.
    
    Provides comprehensive Vertex AI Search capabilities including semantic search,
    vector embeddings, enterprise search, and integration with platform services.
    """
    
    def __init__(self,
                 default_config: Optional[VertexSearchConfig] = None,
                 project_id: Optional[str] = None,
                 location: Optional[str] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 auth_config: Optional[ToolAuthConfig] = None):
        
        super().__init__(
            tool_type=ToolType.VERTEX_SEARCH,
            tool_name="vertex_search",
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            auth_config=auth_config,
        )
        
        # Vertex AI configuration
        self.project_id = project_id
        self.location = location or "global"
        self.default_config = default_config or VertexSearchConfig()
        
        # Search optimization settings
        self.relevance_threshold = 0.5
        self.max_snippet_length = 300
        self.result_diversification = True
        
        # Search enhancement features
        self.auto_query_expansion = True
        self.spell_correction = True
        self.semantic_clustering = True
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get Vertex AI Search tool configuration."""
        return {
            "tool_type": "vertex_search",
            "built_in": True,
            "requires_auth": True,  # Vertex AI always requires authentication
            "project_id": self.project_id,
            "location": self.location,
            "supported_search_types": [stype.value for stype in SearchType],
            "supported_content_types": [ctype.value for ctype in ContentType],
            "default_config": self.default_config.to_dict(),
            "features": {
                "semantic_search": True,
                "vector_embeddings": True,
                "enterprise_search": True,
                "faceted_search": True,
                "boost_specifications": True,
                "auto_query_expansion": self.auto_query_expansion,
                "spell_correction": self.spell_correction,
                "semantic_clustering": self.semantic_clustering,
            },
        }
    
    def execute_tool(self, context: ToolContext, **kwargs) -> ToolResult:
        """Execute Vertex AI Search with comprehensive result processing."""
        start_time = time.time()
        
        # Parse search parameters
        query = kwargs.get("query", "")
        config = self._parse_search_config(**kwargs)
        
        if not query.strip():
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error="Empty search query provided",
            )
        
        try:
            # Get enhanced context for full capabilities
            enhanced_context = self.get_enhanced_context(context)
            enhanced_context.start_execution()
            
            # Get ADK built-in Vertex AI Search tool
            builtin_search = self.get_builtin_tool_instance(context, "vertex_ai_search")
            if not builtin_search:
                # Fallback to context method
                if not hasattr(context, 'vertex_search'):
                    return ToolResult(
                        tool_type=self.tool_type,
                        status=ToolExecutionStatus.FAILED,
                        error="Vertex AI Search capability not available",
                    )
                builtin_search = context
            
            # Log search execution
            self.log_tool_usage("vertex_search", {
                "query": query[:100],  # Truncate for logging
                "search_type": config.search_type.value,
                "content_type": config.content_type.value,
                "max_results": config.max_results,
            })
            
            # Execute search with different strategies based on search type
            if config.search_type == SearchType.SEMANTIC:
                search_results = self._execute_semantic_search(builtin_search, query, config)
            elif config.search_type == SearchType.KEYWORD:
                search_results = self._execute_keyword_search(builtin_search, query, config)
            elif config.search_type == SearchType.HYBRID:
                search_results = self._execute_hybrid_search(builtin_search, query, config)
            elif config.search_type == SearchType.ENTERPRISE:
                search_results = self._execute_enterprise_search(builtin_search, query, config)
            else:
                search_results = self._execute_semantic_search(builtin_search, query, config)
            
            # Post-process and enhance results
            enhanced_results = self._enhance_search_results(search_results, query, config, enhanced_context)
            
            # Store results in memory if available
            if self.memory_service and enhanced_results:
                self._store_search_results_in_memory(query, enhanced_results, enhanced_context)
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.COMPLETED,
                data={
                    "query": query,
                    "results": [result.to_dict() for result in enhanced_results],
                    "result_count": len(enhanced_results),
                    "search_config": config.to_dict(),
                    "search_metadata": self._generate_search_metadata(enhanced_results, config),
                },
                metadata={
                    "query_length": len(query),
                    "search_type": config.search_type.value,
                    "content_type": config.content_type.value,
                    "max_results_requested": config.max_results,
                    "actual_results_returned": len(enhanced_results),
                    "average_similarity_score": self._calculate_average_similarity(enhanced_results),
                    "relevance_distribution": self._analyze_relevance_distribution(enhanced_results),
                },
                execution_time_ms=execution_time,
            )
            
            # Complete execution tracking
            enhanced_context.complete_execution(result)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Vertex AI Search execution failed: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg, 
                                query=query[:100],
                                search_type=config.search_type.value,
                                execution_time_ms=execution_time)
            
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error=error_msg,
                execution_time_ms=execution_time,
            )
    
    def semantic_search(self, 
                       query: str,
                       context: ToolContext,
                       max_results: int = 10,
                       **search_options) -> List[VertexSearchResult]:
        """
        Convenient semantic search method with direct result access.
        
        Args:
            query: Search query string
            context: Tool execution context
            max_results: Maximum number of results
            **search_options: Additional search options
            
        Returns:
            List of processed search results
        """
        result = self.execute_with_context(
            context,
            query=query,
            search_type=SearchType.SEMANTIC,
            max_results=max_results,
            **search_options
        )
        
        if result.success and "results" in result.tool_result.data:
            return [
                VertexSearchResult(**result_data) 
                for result_data in result.tool_result.data["results"]
            ]
        
        return []
    
    def enterprise_search_with_access_control(self,
                                            query: str,
                                            context: ToolContext,
                                            user_id: Optional[str] = None,
                                            access_groups: Optional[List[str]] = None,
                                            **search_options) -> Dict[str, Any]:
        """
        Enterprise search with access control and user context.
        
        Args:
            query: Search query string
            context: Tool execution context
            user_id: User ID for access control
            access_groups: User's access groups
            **search_options: Additional search options
            
        Returns:
            Dictionary with search results and access control information
        """
        # Add user context to search configuration
        user_labels = {"user_id": user_id} if user_id else {}
        if access_groups:
            user_labels["access_groups"] = ",".join(access_groups)
        
        result = self.execute_with_context(
            context,
            query=query,
            search_type=SearchType.ENTERPRISE,
            user_labels=user_labels,
            **search_options
        )
        
        return {
            "search_results": result.tool_result.data if result.success else None,
            "access_control": {
                "user_id": user_id,
                "access_groups": access_groups,
                "filtered_results": True,  # Enterprise search applies access control
            },
            "overall_success": result.success,
            "execution_info": result.to_dict(),
        }
    
    def _parse_search_config(self, **kwargs) -> VertexSearchConfig:
        """Parse and validate search configuration."""
        search_type = SearchType(kwargs.get("search_type", self.default_config.search_type.value))
        content_type = ContentType(kwargs.get("content_type", self.default_config.content_type.value))
        
        return VertexSearchConfig(
            search_type=search_type,
            content_type=content_type,
            max_results=min(kwargs.get("max_results", self.default_config.max_results), 100),
            search_engine_id=kwargs.get("search_engine_id", self.default_config.search_engine_id),
            data_store_id=kwargs.get("data_store_id", self.default_config.data_store_id),
            serving_config=kwargs.get("serving_config", self.default_config.serving_config),
            filter_expression=kwargs.get("filter_expression", self.default_config.filter_expression),
            order_by=kwargs.get("order_by", self.default_config.order_by),
            facet_specs=kwargs.get("facet_specs", self.default_config.facet_specs),
            boost_specs=kwargs.get("boost_specs", self.default_config.boost_specs),
            similarity_threshold=kwargs.get("similarity_threshold", self.default_config.similarity_threshold),
            include_snippets=kwargs.get("include_snippets", self.default_config.include_snippets),
            snippet_length=kwargs.get("snippet_length", self.default_config.snippet_length),
            user_labels=kwargs.get("user_labels", self.default_config.user_labels),
        )
    
    def _execute_semantic_search(self, search_tool: Any, query: str, config: VertexSearchConfig) -> List[VertexSearchResult]:
        """Execute semantic search using vector embeddings."""
        try:
            search_params = config.to_dict()
            search_params["query"] = query
            search_params["search_type"] = "semantic"
            
            if hasattr(search_tool, 'semantic_search'):
                raw_results = search_tool.semantic_search(**search_params)
            elif hasattr(search_tool, 'search'):
                raw_results = search_tool.search(**search_params)
            elif hasattr(search_tool, 'vertex_search'):
                raw_results = search_tool.vertex_search(**search_params)
            else:
                raise ValueError("No semantic search method available")
            
            return self._process_vertex_results(raw_results, config)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _execute_keyword_search(self, search_tool: Any, query: str, config: VertexSearchConfig) -> List[VertexSearchResult]:
        """Execute traditional keyword search."""
        try:
            search_params = config.to_dict()
            search_params["query"] = query
            search_params["search_type"] = "keyword"
            
            if hasattr(search_tool, 'keyword_search'):
                raw_results = search_tool.keyword_search(**search_params)
            elif hasattr(search_tool, 'search'):
                raw_results = search_tool.search(**search_params)
            else:
                raise ValueError("No keyword search method available")
            
            return self._process_vertex_results(raw_results, config)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Keyword search failed: {e}")
            return []
    
    def _execute_hybrid_search(self, search_tool: Any, query: str, config: VertexSearchConfig) -> List[VertexSearchResult]:
        """Execute hybrid search combining semantic and keyword approaches."""
        try:
            # Execute both semantic and keyword searches
            semantic_results = self._execute_semantic_search(search_tool, query, config)
            keyword_results = self._execute_keyword_search(search_tool, query, config)
            
            # Merge and deduplicate results
            merged_results = self._merge_search_results(semantic_results, keyword_results, config)
            
            return merged_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _execute_enterprise_search(self, search_tool: Any, query: str, config: VertexSearchConfig) -> List[VertexSearchResult]:
        """Execute enterprise search with access controls."""
        try:
            search_params = config.to_dict()
            search_params["query"] = query
            search_params["search_type"] = "enterprise"
            
            # Add enterprise-specific parameters
            if config.user_labels:
                search_params["user_labels"] = config.user_labels
            
            if hasattr(search_tool, 'enterprise_search'):
                raw_results = search_tool.enterprise_search(**search_params)
            elif hasattr(search_tool, 'search'):
                raw_results = search_tool.search(**search_params)
            else:
                raise ValueError("No enterprise search method available")
            
            return self._process_vertex_results(raw_results, config)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Enterprise search failed: {e}")
            return []
    
    def _process_vertex_results(self, raw_results: Any, config: VertexSearchConfig) -> List[VertexSearchResult]:
        """Process raw Vertex AI Search results into platform format."""
        if not raw_results:
            return []
        
        processed_results = []
        
        try:
            # Handle different result formats from Vertex AI
            if hasattr(raw_results, 'results'):
                results = raw_results.results
            elif isinstance(raw_results, list):
                results = raw_results
            elif isinstance(raw_results, dict) and 'results' in raw_results:
                results = raw_results['results']
            else:
                if self.logger:
                    self.logger.warning(f"Unexpected Vertex AI result format: {type(raw_results)}")
                return []
            
            for i, result_item in enumerate(results[:config.max_results]):
                try:
                    # Extract information based on Vertex AI result structure
                    document_id = result_item.get('id', f"doc_{i}")
                    title = result_item.get('title', result_item.get('name', 'Untitled'))
                    content = result_item.get('content', result_item.get('text', ''))
                    snippet = result_item.get('snippet', content[:config.snippet_length])
                    
                    # Extract similarity/relevance score
                    similarity_score = result_item.get('similarity_score', 
                                                     result_item.get('relevance_score', 
                                                                   result_item.get('score', 0.0)))
                    
                    # Create search result
                    vertex_result = VertexSearchResult(
                        document_id=document_id,
                        title=title,
                        content=content,
                        snippet=snippet,
                        similarity_score=similarity_score,
                        rank=i + 1,
                        url=result_item.get('url', result_item.get('uri')),
                        content_type=result_item.get('content_type', 'document'),
                        facets=result_item.get('facets', {}),
                        metadata=result_item.get('metadata', {}),
                        embeddings=result_item.get('embeddings'),
                    )
                    
                    processed_results.append(vertex_result)
                    
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error processing Vertex AI result {i}: {e}")
                    continue
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing Vertex AI results: {e}")
        
        return processed_results
    
    def _enhance_search_results(self, 
                              results: List[VertexSearchResult], 
                              query: str, 
                              config: VertexSearchConfig,
                              context) -> List[VertexSearchResult]:
        """Enhance search results with additional processing."""
        if not results:
            return results
        
        enhanced_results = []
        
        for result in results:
            try:
                # Filter by similarity threshold
                if result.similarity_score < config.similarity_threshold:
                    continue
                
                # Enhance snippet if needed
                if config.include_snippets and not result.snippet:
                    result.snippet = result.content[:config.snippet_length]
                
                # Add query-specific relevance information
                result.metadata["query_terms_in_title"] = self._count_query_terms_in_text(query, result.title)
                result.metadata["query_terms_in_content"] = self._count_query_terms_in_text(query, result.content)
                
                enhanced_results.append(result)
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error enhancing result {result.document_id}: {e}")
                enhanced_results.append(result)  # Include original result
        
        # Apply result diversification if enabled
        if self.result_diversification:
            enhanced_results = self._diversify_results(enhanced_results)
        
        return enhanced_results
    
    def _merge_search_results(self, 
                            semantic_results: List[VertexSearchResult],
                            keyword_results: List[VertexSearchResult],
                            config: VertexSearchConfig) -> List[VertexSearchResult]:
        """Merge semantic and keyword search results for hybrid search."""
        # Create a map of document IDs to results
        result_map = {}
        
        # Add semantic results with boosted scores
        for result in semantic_results:
            result.similarity_score *= 1.2  # Boost semantic results
            result.metadata["search_method"] = "semantic"
            result_map[result.document_id] = result
        
        # Add keyword results, merging with existing if present
        for result in keyword_results:
            if result.document_id in result_map:
                # Merge scores (take maximum)
                existing = result_map[result.document_id]
                existing.similarity_score = max(existing.similarity_score, result.similarity_score)
                existing.metadata["search_method"] = "hybrid"
            else:
                result.metadata["search_method"] = "keyword"
                result_map[result.document_id] = result
        
        # Convert back to list and sort by similarity score
        merged_results = list(result_map.values())
        merged_results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return merged_results[:config.max_results]
    
    def _diversify_results(self, results: List[VertexSearchResult]) -> List[VertexSearchResult]:
        """Apply result diversification to avoid too many similar results."""
        if len(results) <= 3:
            return results
        
        diversified = []
        seen_domains = set()
        content_similarity_threshold = 0.8
        
        for result in results:
            # Extract domain from URL if available
            domain = None
            if result.url:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(result.url).netloc
                except:
                    pass
            
            # Check domain diversity
            if domain and domain in seen_domains:
                # Skip if we already have multiple results from this domain
                domain_count = sum(1 for r in diversified if r.url and urlparse(r.url).netloc == domain)
                if domain_count >= 2:
                    continue
            
            # Check content similarity (simplified)
            is_similar = False
            for existing in diversified[-3:]:  # Check against recent results
                if self._calculate_content_similarity(result.content, existing.content) > content_similarity_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                diversified.append(result)
                if domain:
                    seen_domains.add(domain)
        
        return diversified
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity (placeholder implementation)."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _count_query_terms_in_text(self, query: str, text: str) -> int:
        """Count how many query terms appear in the text."""
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        return sum(1 for term in query_terms if term in text_lower)
    
    def _calculate_average_similarity(self, results: List[VertexSearchResult]) -> float:
        """Calculate average similarity score across results."""
        if not results:
            return 0.0
        return sum(r.similarity_score for r in results) / len(results)
    
    def _analyze_relevance_distribution(self, results: List[VertexSearchResult]) -> Dict[str, int]:
        """Analyze distribution of relevance categories."""
        distribution = {"highly_relevant": 0, "relevant": 0, "somewhat_relevant": 0, "low_relevance": 0}
        
        for result in results:
            distribution[result.relevance_category] += 1
        
        return distribution
    
    def _generate_search_metadata(self, results: List[VertexSearchResult], config: VertexSearchConfig) -> Dict[str, Any]:
        """Generate comprehensive metadata about search results."""
        if not results:
            return {}
        
        content_types = {}
        facet_summary = {}
        
        for result in results:
            # Count content types
            content_types[result.content_type] = content_types.get(result.content_type, 0) + 1
            
            # Summarize facets
            for facet_key, facet_value in result.facets.items():
                if facet_key not in facet_summary:
                    facet_summary[facet_key] = {}
                facet_summary[facet_key][str(facet_value)] = facet_summary[facet_key].get(str(facet_value), 0) + 1
        
        return {
            "content_type_distribution": content_types,
            "facet_summary": facet_summary,
            "search_configuration": config.to_dict(),
            "result_quality": {
                "average_similarity": self._calculate_average_similarity(results),
                "relevance_distribution": self._analyze_relevance_distribution(results),
                "has_high_quality_results": any(r.similarity_score > 0.8 for r in results),
            },
        }
    
    def _store_search_results_in_memory(self,
                                      query: str,
                                      results: List[VertexSearchResult],
                                      context) -> None:
        """Store high-quality search results in memory for future reference."""
        try:
            # Store only high-relevance results
            high_quality_results = [
                r for r in results 
                if r.similarity_score > 0.8 and r.relevance_category in ["highly_relevant", "relevant"]
            ]
            
            for result in high_quality_results[:3]:  # Store top 3
                memory_text = f"""
Vertex AI Search Result for '{query}':
Title: {result.title}
Content: {result.content[:500]}
Similarity Score: {result.similarity_score:.3f}
Document ID: {result.document_id}
URL: {result.url or 'N/A'}
""".strip()
                
                if hasattr(context, 'store_memory'):
                    context.store_memory(
                        text=memory_text,
                        metadata={
                            "type": "vertex_search_result",
                            "query": query,
                            "document_id": result.document_id,
                            "similarity_score": result.similarity_score,
                            "relevance_category": result.relevance_category,
                            "content_type": result.content_type,
                        }
                    )
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing Vertex AI search results in memory: {e}")