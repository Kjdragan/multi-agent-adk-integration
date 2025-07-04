"""
Tavily MCP Server Integration

Provides comprehensive integration with Tavily's web search and research API,
including real-time data access, content extraction, and AI-optimized search results.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from ..base import (
    HTTPMCPServer,
    MCPServerConfig,
    MCPServerResult,
    MCPServerStatus,
    MCPAuthConfig,
    MCPOperationType,
)
from ...platform_logging import RunLogger
from ...services import SessionService, MemoryService, ArtifactService


class TavilySearchType(str, Enum):
    """Types of Tavily searches."""
    SEARCH = "search"               # General web search
    NEWS = "news"                  # News-focused search
    ACADEMIC = "academic"          # Academic and research content
    SOCIAL = "social"              # Social media content
    IMAGES = "images"              # Image search
    VIDEOS = "videos"              # Video search


class TavilyContentType(str, Enum):
    """Content types for Tavily extraction."""
    TEXT = "text"                  # Text content only
    FULL = "full"                  # Full page content
    SUMMARY = "summary"            # AI-generated summary
    MARKDOWN = "markdown"          # Markdown formatted content


@dataclass
class TavilyConfig:
    """Configuration for Tavily searches."""
    search_type: TavilySearchType = TavilySearchType.SEARCH
    max_results: int = 10
    include_answer: bool = True
    include_raw_content: bool = False
    include_images: bool = False
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    content_type: TavilyContentType = TavilyContentType.TEXT
    search_depth: str = "basic"  # "basic" or "advanced"
    topic: Optional[str] = None
    days: Optional[int] = None  # Number of days back to search
    max_tokens: int = 4000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "include_images": self.include_images,
            "max_tokens": self.max_tokens,
        }
        
        # Add optional parameters
        if self.include_domains:
            config["include_domains"] = self.include_domains
        if self.exclude_domains:
            config["exclude_domains"] = self.exclude_domains
        if self.topic:
            config["topic"] = self.topic
        if self.days:
            config["days"] = self.days
        
        return config


@dataclass
class TavilySource:
    """Source information from Tavily search."""
    url: str
    title: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None
    domain: str = ""
    favicon: Optional[str] = None
    raw_content: Optional[str] = None
    
    def __post_init__(self):
        """Extract domain from URL."""
        if self.url:
            try:
                from urllib.parse import urlparse
                self.domain = urlparse(self.url).netloc
            except:
                self.domain = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "score": self.score,
            "published_date": self.published_date,
            "domain": self.domain,
            "favicon": self.favicon,
            "raw_content": self.raw_content,
        }


@dataclass
class TavilySearchResult:
    """Result from Tavily search."""
    query: str
    answer: str = ""
    sources: List[TavilySource] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    follow_up_questions: List[str] = field(default_factory=list)
    search_type: TavilySearchType = TavilySearchType.SEARCH
    
    @property
    def source_count(self) -> int:
        """Number of sources in the result."""
        return len(self.sources)
    
    @property
    def unique_domains(self) -> List[str]:
        """List of unique domains in sources."""
        return list(set(source.domain for source in self.sources))
    
    @property
    def average_score(self) -> float:
        """Average relevance score of sources."""
        if not self.sources:
            return 0.0
        return sum(source.score for source in self.sources) / len(self.sources)
    
    @property
    def top_source(self) -> Optional[TavilySource]:
        """Source with highest relevance score."""
        if not self.sources:
            return None
        return max(self.sources, key=lambda s: s.score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "images": self.images,
            "search_metadata": self.search_metadata,
            "execution_time_ms": self.execution_time_ms,
            "follow_up_questions": self.follow_up_questions,
            "search_type": self.search_type.value,
            "source_count": self.source_count,
            "unique_domains": self.unique_domains,
            "average_score": self.average_score,
            "top_source": self.top_source.to_dict() if self.top_source else None,
        }


class TavilyServer(HTTPMCPServer):
    """
    Tavily MCP server integration.
    
    Provides comprehensive Tavily web search and research capabilities including
    real-time data access, content extraction, and AI-optimized search results.
    """
    
    def __init__(self,
                 api_key: str,
                 default_config: Optional[TavilyConfig] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        # Create MCP server configuration
        auth_config = MCPAuthConfig(
            auth_type="api_key",
            api_key=api_key,
            headers={"Content-Type": "application/json"},
        )
        
        server_config = MCPServerConfig(
            server_name="tavily",
            server_type="web_search",
            base_url="https://api.tavily.com",
            auth_config=auth_config,
            rate_limit_requests_per_minute=100,  # Tavily rate limits
            rate_limit_requests_per_hour=1000,
            timeout_seconds=30,
            enable_caching=True,
            cache_ttl_seconds=300,  # 5 minutes for web search
        )
        
        super().__init__(
            config=server_config,
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        # Tavily-specific configuration
        self.default_config = default_config or TavilyConfig()
        
        # Search optimization settings
        self.content_filtering = True
        self.duplicate_removal = True
        self.relevance_scoring = True
    
    async def _connect_server(self) -> bool:
        """Test connection to Tavily API."""
        try:
            # Test with a simple search
            test_response = await self.make_http_request(
                method="POST",
                endpoint="/search",
                json_data={
                    "api_key": self.config.auth_config.api_key,
                    "query": "test",
                    "max_results": 1,
                }
            )
            
            return "results" in test_response or "answer" in test_response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tavily connection test failed: {e}")
            return False
    
    async def _authenticate(self) -> bool:
        """Tavily uses API key in request body, so connection test serves as auth."""
        return True
    
    async def _execute_operation(self, 
                               operation_type: MCPOperationType, 
                               parameters: Dict[str, Any]) -> Any:
        """Execute Tavily-specific operations."""
        if operation_type == MCPOperationType.SEARCH:
            return await self._search(parameters)
        elif operation_type == MCPOperationType.RETRIEVE:
            return await self._extract_content(parameters)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Tavily server capabilities."""
        return {
            "server_type": "web_search",
            "real_time_data": True,
            "content_extraction": True,
            "ai_optimized": True,
            "supported_operations": [
                MCPOperationType.SEARCH.value,
                MCPOperationType.RETRIEVE.value,
            ],
            "search_types": [search_type.value for search_type in TavilySearchType],
            "content_types": [content_type.value for content_type in TavilyContentType],
            "features": {
                "web_search": True,
                "news_search": True,
                "academic_search": True,
                "image_search": True,
                "video_search": True,
                "content_extraction": True,
                "real_time_results": True,
                "ai_generated_answers": True,
                "follow_up_questions": True,
                "domain_filtering": True,
                "content_filtering": self.content_filtering,
                "duplicate_removal": self.duplicate_removal,
                "relevance_scoring": self.relevance_scoring,
            },
        }
    
    async def search(self, 
                    query: str,
                    search_config: Optional[TavilyConfig] = None) -> TavilySearchResult:
        """
        Perform web search with Tavily.
        
        Args:
            query: Search query
            search_config: Search configuration
            
        Returns:
            Comprehensive search result with sources and AI-generated answer
        """
        parameters = {
            "query": query,
            "config": search_config or self.default_config,
        }
        
        result = await self.execute_operation(
            MCPOperationType.SEARCH,
            parameters
        )
        
        if result.success and result.data:
            return TavilySearchResult(**result.data)
        else:
            # Return empty result with error information
            return TavilySearchResult(
                query=query,
                answer=f"Search failed: {result.error}",
                search_type=search_config.search_type if search_config else self.default_config.search_type,
            )
    
    async def news_search(self,
                         query: str,
                         days_back: int = 7,
                         max_results: int = 10) -> TavilySearchResult:
        """
        Search for recent news articles.
        
        Args:
            query: News search query
            days_back: Number of days back to search
            max_results: Maximum number of results
            
        Returns:
            News search results
        """
        news_config = TavilyConfig(
            search_type=TavilySearchType.NEWS,
            max_results=max_results,
            days=days_back,
            include_answer=True,
        )
        
        return await self.search(query, news_config)
    
    async def academic_search(self,
                            query: str,
                            max_results: int = 10) -> TavilySearchResult:
        """
        Search for academic and research content.
        
        Args:
            query: Academic search query
            max_results: Maximum number of results
            
        Returns:
            Academic search results
        """
        academic_config = TavilyConfig(
            search_type=TavilySearchType.ACADEMIC,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
            include_domains=["arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov"],
        )
        
        return await self.search(query, academic_config)
    
    async def extract_content(self,
                            urls: List[str],
                            content_type: TavilyContentType = TavilyContentType.TEXT) -> Dict[str, Any]:
        """
        Extract content from URLs using Tavily.
        
        Args:
            urls: List of URLs to extract content from
            content_type: Type of content to extract
            
        Returns:
            Dictionary mapping URLs to extracted content
        """
        parameters = {
            "urls": urls,
            "content_type": content_type,
        }
        
        result = await self.execute_operation(
            MCPOperationType.RETRIEVE,
            parameters
        )
        
        if result.success and result.data:
            return result.data
        else:
            return {"error": result.error, "urls": urls}
    
    async def comprehensive_research(self,
                                   topic: str,
                                   research_depth: str = "advanced",
                                   include_news: bool = True,
                                   include_academic: bool = True) -> Dict[str, TavilySearchResult]:
        """
        Perform comprehensive research on a topic using multiple search types.
        
        Args:
            topic: Research topic
            research_depth: Search depth ("basic" or "advanced")
            include_news: Whether to include news search
            include_academic: Whether to include academic search
            
        Returns:
            Dictionary mapping search type to results
        """
        research_results = {}
        
        # General web search
        general_config = TavilyConfig(
            search_type=TavilySearchType.SEARCH,
            search_depth=research_depth,
            max_results=15,
            include_answer=True,
        )
        
        general_result = await self.search(topic, general_config)
        research_results["general"] = general_result
        
        # News search
        if include_news:
            news_result = await self.news_search(topic, days_back=30, max_results=10)
            research_results["news"] = news_result
        
        # Academic search
        if include_academic:
            academic_result = await self.academic_search(topic, max_results=10)
            research_results["academic"] = academic_result
        
        # Store comprehensive results in memory
        for search_type, result in research_results.items():
            await self.store_results_in_memory(
                MCPOperationType.SEARCH,
                f"{topic} ({search_type})",
                result.to_dict()
            )
        
        return research_results
    
    async def _search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Internal search implementation."""
        start_time = time.time()
        
        query = parameters["query"]
        config = parameters.get("config", self.default_config)
        
        # Prepare API request
        api_request = {
            "api_key": self.config.auth_config.api_key,
            "query": query,
        }
        
        # Add configuration parameters
        api_request.update(config.to_dict())
        
        try:
            # Choose endpoint based on search type
            if config.search_type == TavilySearchType.NEWS:
                endpoint = "/news"
            elif config.search_type == TavilySearchType.IMAGES:
                endpoint = "/images"
            elif config.search_type == TavilySearchType.VIDEOS:
                endpoint = "/videos"
            else:
                endpoint = "/search"
            
            # Make API call
            response = await self.make_http_request(
                method="POST",
                endpoint=endpoint,
                json_data=api_request
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Process response
            search_result = self._process_search_response(
                response, query, config, execution_time
            )
            
            # Log successful search
            self.log_operation("search", parameters, None)
            
            return search_result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tavily search failed: {e}")
            raise
    
    async def _extract_content(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Internal content extraction implementation."""
        start_time = time.time()
        
        urls = parameters["urls"]
        content_type = parameters.get("content_type", TavilyContentType.TEXT)
        
        # Prepare API request
        api_request = {
            "api_key": self.config.auth_config.api_key,
            "urls": urls,
            "format": content_type.value,
        }
        
        try:
            response = await self.make_http_request(
                method="POST",
                endpoint="/extract",
                json_data=api_request
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Process extraction response
            extraction_result = self._process_extraction_response(
                response, urls, content_type, execution_time
            )
            
            self.log_operation("extract_content", parameters, None)
            
            return extraction_result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tavily content extraction failed: {e}")
            raise
    
    def _process_search_response(self, 
                               response: Dict[str, Any],
                               query: str,
                               config: TavilyConfig,
                               execution_time: float) -> Dict[str, Any]:
        """Process Tavily search API response."""
        try:
            # Extract answer
            answer = response.get("answer", "")
            
            # Extract sources
            sources = []
            if "results" in response:
                for source_data in response["results"]:
                    source = TavilySource(
                        url=source_data.get("url", ""),
                        title=source_data.get("title", ""),
                        content=source_data.get("content", ""),
                        score=source_data.get("score", 0.0),
                        published_date=source_data.get("published_date"),
                        favicon=source_data.get("favicon"),
                        raw_content=source_data.get("raw_content"),
                    )
                    sources.append(source)
            
            # Apply filtering and scoring if enabled
            if self.relevance_scoring:
                sources = self._apply_relevance_scoring(sources, query)
            
            if self.duplicate_removal:
                sources = self._remove_duplicates(sources)
            
            if self.content_filtering:
                sources = self._filter_content(sources)
            
            # Sort by score (highest first)
            sources.sort(key=lambda s: s.score, reverse=True)
            
            return {
                "query": query,
                "answer": answer,
                "sources": [source.to_dict() for source in sources],
                "images": response.get("images", []),
                "search_metadata": {
                    "search_id": response.get("search_id", ""),
                    "response_time": response.get("response_time", 0),
                    "total_results": len(sources),
                },
                "execution_time_ms": execution_time,
                "follow_up_questions": response.get("follow_up_questions", []),
                "search_type": config.search_type,
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing Tavily search response: {e}")
            
            return {
                "query": query,
                "answer": f"Error processing response: {str(e)}",
                "sources": [],
                "images": [],
                "search_metadata": {"error": str(e)},
                "execution_time_ms": execution_time,
                "follow_up_questions": [],
                "search_type": config.search_type,
            }
    
    def _process_extraction_response(self,
                                   response: Dict[str, Any],
                                   urls: List[str],
                                   content_type: TavilyContentType,
                                   execution_time: float) -> Dict[str, Any]:
        """Process Tavily content extraction API response."""
        try:
            extracted_content = {}
            
            if "results" in response:
                for result in response["results"]:
                    url = result.get("url", "")
                    content = result.get("content", "")
                    
                    extracted_content[url] = {
                        "content": content,
                        "title": result.get("title", ""),
                        "status": "success" if content else "failed",
                        "content_type": content_type.value,
                        "extraction_time": result.get("extraction_time", 0),
                    }
            
            return {
                "urls": urls,
                "extracted_content": extracted_content,
                "execution_time_ms": execution_time,
                "total_extracted": len(extracted_content),
                "success_count": sum(1 for item in extracted_content.values() if item["status"] == "success"),
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing Tavily extraction response: {e}")
            
            return {
                "urls": urls,
                "extracted_content": {},
                "error": str(e),
                "execution_time_ms": execution_time,
            }
    
    def _apply_relevance_scoring(self, sources: List[TavilySource], query: str) -> List[TavilySource]:
        """Apply additional relevance scoring to sources."""
        query_terms = query.lower().split()
        
        for source in sources:
            # Boost score based on query term matches in title and content
            title_matches = sum(1 for term in query_terms if term in source.title.lower())
            content_matches = sum(1 for term in query_terms if term in source.content.lower())
            
            # Calculate boost factor
            title_boost = (title_matches / len(query_terms)) * 0.3
            content_boost = (content_matches / len(query_terms)) * 0.2
            
            # Apply boost to existing score
            source.score = min(source.score + title_boost + content_boost, 1.0)
        
        return sources
    
    def _remove_duplicates(self, sources: List[TavilySource]) -> List[TavilySource]:
        """Remove duplicate sources based on URL and content similarity."""
        unique_sources = []
        seen_urls = set()
        
        for source in sources:
            # Skip exact URL duplicates
            if source.url in seen_urls:
                continue
            
            # Check content similarity with existing sources
            is_duplicate = False
            for existing in unique_sources:
                if self._calculate_content_similarity(source.content, existing.content) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sources.append(source)
                seen_urls.add(source.url)
        
        return unique_sources
    
    def _filter_content(self, sources: List[TavilySource]) -> List[TavilySource]:
        """Filter sources based on content quality."""
        filtered_sources = []
        
        for source in sources:
            # Filter based on content length and quality
            if len(source.content.strip()) < 50:  # Too short
                continue
            
            # Filter based on title quality
            if not source.title or len(source.title.strip()) < 10:  # No/poor title
                continue
            
            # Filter out common low-quality patterns
            low_quality_patterns = [
                "404", "not found", "error", "access denied",
                "please enable javascript", "loading...",
            ]
            
            if any(pattern in source.content.lower() for pattern in low_quality_patterns):
                continue
            
            filtered_sources.append(source)
        
        return filtered_sources
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity between two texts."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0