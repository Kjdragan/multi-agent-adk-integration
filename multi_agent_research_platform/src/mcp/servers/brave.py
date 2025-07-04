"""
Brave Search MCP Server Integration

Provides comprehensive integration with Brave Search API's privacy-focused search capabilities,
including independent web indexing, real-time results, and privacy-preserving search.
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


class BraveSearchType(str, Enum):
    """Types of Brave searches."""
    WEB = "web"                    # General web search
    NEWS = "news"                  # News search
    IMAGES = "images"              # Image search
    VIDEOS = "videos"              # Video search
    PLACES = "places"              # Local places search


class BraveSafeSearch(str, Enum):
    """Safe search levels for Brave."""
    OFF = "off"                    # No filtering
    MODERATE = "moderate"          # Moderate filtering
    STRICT = "strict"              # Strict filtering


class BraveFreshness(str, Enum):
    """Freshness filters for Brave search."""
    ALL = "all"                    # All time
    DAY = "pd"                     # Past day
    WEEK = "pw"                    # Past week
    MONTH = "pm"                   # Past month
    YEAR = "py"                    # Past year


@dataclass
class BraveConfig:
    """Configuration for Brave searches."""
    search_type: BraveSearchType = BraveSearchType.WEB
    count: int = 10                # Number of results (max 20)
    offset: int = 0                # Offset for pagination
    safe_search: BraveSafeSearch = BraveSafeSearch.MODERATE
    freshness: BraveFreshness = BraveFreshness.ALL
    country: str = "US"            # Country code
    search_lang: str = "en"        # Search language
    ui_lang: str = "en-US"         # UI language
    spellcheck: bool = True        # Enable spell checking
    result_filter: Optional[str] = None  # Filter results
    goggles_id: Optional[str] = None     # Brave Goggles ID
    units: str = "metric"          # Units for measurements
    extra_snippets: bool = False   # Include extra snippets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "count": min(self.count, 20),  # Brave max is 20
            "offset": self.offset,
            "safesearch": self.safe_search.value,
            "freshness": self.freshness.value,
            "country": self.country,
            "search_lang": self.search_lang,
            "ui_lang": self.ui_lang,
            "spellcheck": self.spellcheck,
            "units": self.units,
            "extra_snippets": self.extra_snippets,
        }
        
        # Add optional parameters
        if self.result_filter:
            config["result_filter"] = self.result_filter
        if self.goggles_id:
            config["goggles_id"] = self.goggles_id
        
        return config


@dataclass
class BraveWebResult:
    """Individual web result from Brave search."""
    title: str
    url: str
    description: str
    displayed_url: str = ""
    ranking: int = 0
    page_age: Optional[str] = None
    profile: Dict[str, Any] = field(default_factory=dict)
    language: str = ""
    family_friendly: bool = True
    type: str = "search_result"
    subtype: str = ""
    meta_url: Dict[str, str] = field(default_factory=dict)
    thumbnail: Optional[Dict[str, str]] = None
    age: Optional[str] = None
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(self.url).netloc
        except:
            return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "displayed_url": self.displayed_url,
            "ranking": self.ranking,
            "page_age": self.page_age,
            "profile": self.profile,
            "language": self.language,
            "family_friendly": self.family_friendly,
            "type": self.type,
            "subtype": self.subtype,
            "meta_url": self.meta_url,
            "thumbnail": self.thumbnail,
            "age": self.age,
            "domain": self.domain,
        }


@dataclass
class BraveNewsResult:
    """News result from Brave search."""
    title: str
    url: str
    description: str
    age: str
    page_age: str = ""
    breaking: bool = False
    family_friendly: bool = True
    meta_url: Dict[str, str] = field(default_factory=dict)
    thumbnail: Optional[Dict[str, str]] = None
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(self.url).netloc
        except:
            return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "age": self.age,
            "page_age": self.page_age,
            "breaking": self.breaking,
            "family_friendly": self.family_friendly,
            "meta_url": self.meta_url,
            "thumbnail": self.thumbnail,
            "domain": self.domain,
        }


@dataclass
class BraveSearchResult:
    """Result from Brave search."""
    query: str
    search_type: BraveSearchType
    web_results: List[BraveWebResult] = field(default_factory=list)
    news_results: List[BraveNewsResult] = field(default_factory=list)
    image_results: List[Dict[str, Any]] = field(default_factory=list)
    video_results: List[Dict[str, Any]] = field(default_factory=list)
    place_results: List[Dict[str, Any]] = field(default_factory=list)
    infobox: Optional[Dict[str, Any]] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    locations: List[Dict[str, Any]] = field(default_factory=list)
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    
    @property
    def total_results(self) -> int:
        """Total number of results across all types."""
        return (len(self.web_results) + 
                len(self.news_results) + 
                len(self.image_results) + 
                len(self.video_results) + 
                len(self.place_results))
    
    @property
    def unique_domains(self) -> List[str]:
        """List of unique domains in web and news results."""
        domains = set()
        
        for result in self.web_results:
            domains.add(result.domain)
        
        for result in self.news_results:
            domains.add(result.domain)
        
        return list(domains)
    
    @property
    def has_breaking_news(self) -> bool:
        """Check if any news results are marked as breaking."""
        return any(result.breaking for result in self.news_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "search_type": self.search_type.value,
            "web_results": [result.to_dict() for result in self.web_results],
            "news_results": [result.to_dict() for result in self.news_results],
            "image_results": self.image_results,
            "video_results": self.video_results,
            "place_results": self.place_results,
            "infobox": self.infobox,
            "knowledge_graph": self.knowledge_graph,
            "locations": self.locations,
            "search_metadata": self.search_metadata,
            "execution_time_ms": self.execution_time_ms,
            "total_results": self.total_results,
            "unique_domains": self.unique_domains,
            "has_breaking_news": self.has_breaking_news,
        }


class BraveServer(HTTPMCPServer):
    """
    Brave Search MCP server integration.
    
    Provides comprehensive Brave Search capabilities including privacy-focused web search,
    independent indexing, real-time results, and multi-media search.
    """
    
    def __init__(self,
                 api_key: str,
                 default_config: Optional[BraveConfig] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        # Create MCP server configuration
        auth_config = MCPAuthConfig(
            auth_type="api_key",
            api_key=api_key,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        
        server_config = MCPServerConfig(
            server_name="brave",
            server_type="web_search",
            base_url="https://api.search.brave.com/res/v1",
            auth_config=auth_config,
            rate_limit_requests_per_minute=60,   # Brave rate limits
            rate_limit_requests_per_hour=2000,
            timeout_seconds=15,
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
        
        # Brave-specific configuration
        self.default_config = default_config or BraveConfig()
        
        # Search optimization settings
        self.privacy_focused = True
        self.independent_indexing = True
        self.real_time_results = True
    
    async def _connect_server(self) -> bool:
        """Test connection to Brave Search API."""
        try:
            # Test with a simple web search
            test_response = await self.make_http_request(
                method="GET",
                endpoint="/web/search",
                params={
                    "q": "test",
                    "count": 1,
                }
            )
            
            return "web" in test_response or "mixed" in test_response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Brave connection test failed: {e}")
            return False
    
    async def _authenticate(self) -> bool:
        """Brave uses API key in headers, so connection test serves as auth."""
        return True
    
    async def _execute_operation(self, 
                               operation_type: MCPOperationType, 
                               parameters: Dict[str, Any]) -> Any:
        """Execute Brave-specific operations."""
        if operation_type == MCPOperationType.SEARCH:
            return await self._search(parameters)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Brave server capabilities."""
        return {
            "server_type": "web_search",
            "privacy_focused": self.privacy_focused,
            "independent_indexing": self.independent_indexing,
            "real_time_results": self.real_time_results,
            "supported_operations": [
                MCPOperationType.SEARCH.value,
            ],
            "search_types": [search_type.value for search_type in BraveSearchType],
            "safe_search_levels": [level.value for level in BraveSafeSearch],
            "freshness_filters": [fresh.value for fresh in BraveFreshness],
            "features": {
                "web_search": True,
                "news_search": True,
                "image_search": True,
                "video_search": True,
                "local_search": True,
                "privacy_protection": True,
                "no_tracking": True,
                "independent_index": True,
                "spell_checking": True,
                "goggles_support": True,
                "infobox": True,
                "knowledge_graph": True,
                "pagination": True,
            },
        }
    
    async def search(self, 
                    query: str,
                    search_config: Optional[BraveConfig] = None) -> BraveSearchResult:
        """
        Perform privacy-focused web search with Brave.
        
        Args:
            query: Search query
            search_config: Search configuration
            
        Returns:
            Comprehensive search result with privacy protection
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
            return BraveSearchResult(**result.data)
        else:
            # Return empty result with error information
            return BraveSearchResult(
                query=query,
                search_type=search_config.search_type if search_config else self.default_config.search_type,
                search_metadata={"error": result.error},
            )
    
    async def web_search(self,
                        query: str,
                        count: int = 10,
                        safe_search: BraveSafeSearch = BraveSafeSearch.MODERATE) -> BraveSearchResult:
        """
        Perform web search with Brave.
        
        Args:
            query: Search query
            count: Number of results (max 20)
            safe_search: Safe search level
            
        Returns:
            Web search results
        """
        web_config = BraveConfig(
            search_type=BraveSearchType.WEB,
            count=count,
            safe_search=safe_search,
        )
        
        return await self.search(query, web_config)
    
    async def news_search(self,
                         query: str,
                         count: int = 10,
                         freshness: BraveFreshness = BraveFreshness.ALL) -> BraveSearchResult:
        """
        Search for news articles with Brave.
        
        Args:
            query: News search query
            count: Number of results
            freshness: Freshness filter
            
        Returns:
            News search results
        """
        news_config = BraveConfig(
            search_type=BraveSearchType.NEWS,
            count=count,
            freshness=freshness,
        )
        
        return await self.search(query, news_config)
    
    async def multi_type_search(self,
                               query: str,
                               include_web: bool = True,
                               include_news: bool = True,
                               include_images: bool = False,
                               include_videos: bool = False) -> Dict[str, BraveSearchResult]:
        """
        Perform multi-type search across different content types.
        
        Args:
            query: Search query
            include_web: Include web results
            include_news: Include news results
            include_images: Include image results
            include_videos: Include video results
            
        Returns:
            Dictionary mapping search type to results
        """
        search_results = {}
        
        if include_web:
            web_result = await self.web_search(query, count=15)
            search_results["web"] = web_result
        
        if include_news:
            news_result = await self.news_search(query, count=10)
            search_results["news"] = news_result
        
        if include_images:
            image_config = BraveConfig(search_type=BraveSearchType.IMAGES, count=20)
            image_result = await self.search(query, image_config)
            search_results["images"] = image_result
        
        if include_videos:
            video_config = BraveConfig(search_type=BraveSearchType.VIDEOS, count=15)
            video_result = await self.search(query, video_config)
            search_results["videos"] = video_result
        
        # Store comprehensive results in memory
        for search_type, result in search_results.items():
            await self.store_results_in_memory(
                MCPOperationType.SEARCH,
                f"{query} ({search_type})",
                result.to_dict()
            )
        
        return search_results
    
    async def _search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Internal search implementation."""
        start_time = time.time()
        
        query = parameters["query"]
        config = parameters.get("config", self.default_config)
        
        # Prepare API parameters
        api_params = {
            "q": query,
        }
        
        # Add configuration parameters
        api_params.update(config.to_dict())
        
        try:
            # Choose endpoint based on search type
            endpoint_map = {
                BraveSearchType.WEB: "/web/search",
                BraveSearchType.NEWS: "/news/search",
                BraveSearchType.IMAGES: "/images/search",
                BraveSearchType.VIDEOS: "/videos/search",
                BraveSearchType.PLACES: "/local/search",
            }
            
            endpoint = endpoint_map.get(config.search_type, "/web/search")
            
            # Make API call
            response = await self.make_http_request(
                method="GET",
                endpoint=endpoint,
                params=api_params
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
                self.logger.error(f"Brave search failed: {e}")
            raise
    
    def _process_search_response(self, 
                               response: Dict[str, Any],
                               query: str,
                               config: BraveConfig,
                               execution_time: float) -> Dict[str, Any]:
        """Process Brave search API response."""
        try:
            search_result = {
                "query": query,
                "search_type": config.search_type,
                "web_results": [],
                "news_results": [],
                "image_results": [],
                "video_results": [],
                "place_results": [],
                "infobox": None,
                "knowledge_graph": None,
                "locations": [],
                "search_metadata": {},
                "execution_time_ms": execution_time,
            }
            
            # Process web results
            if "web" in response and "results" in response["web"]:
                for result_data in response["web"]["results"]:
                    web_result = BraveWebResult(
                        title=result_data.get("title", ""),
                        url=result_data.get("url", ""),
                        description=result_data.get("description", ""),
                        displayed_url=result_data.get("displayed_url", ""),
                        ranking=result_data.get("ranking", 0),
                        page_age=result_data.get("page_age"),
                        profile=result_data.get("profile", {}),
                        language=result_data.get("language", ""),
                        family_friendly=result_data.get("family_friendly", True),
                        type=result_data.get("type", "search_result"),
                        subtype=result_data.get("subtype", ""),
                        meta_url=result_data.get("meta_url", {}),
                        thumbnail=result_data.get("thumbnail"),
                        age=result_data.get("age"),
                    )
                    search_result["web_results"].append(web_result.to_dict())
            
            # Process news results
            if "news" in response and "results" in response["news"]:
                for result_data in response["news"]["results"]:
                    news_result = BraveNewsResult(
                        title=result_data.get("title", ""),
                        url=result_data.get("url", ""),
                        description=result_data.get("description", ""),
                        age=result_data.get("age", ""),
                        page_age=result_data.get("page_age", ""),
                        breaking=result_data.get("breaking", False),
                        family_friendly=result_data.get("family_friendly", True),
                        meta_url=result_data.get("meta_url", {}),
                        thumbnail=result_data.get("thumbnail"),
                    )
                    search_result["news_results"].append(news_result.to_dict())
            
            # Process other result types
            for result_type in ["images", "videos", "places"]:
                if result_type in response and "results" in response[result_type]:
                    search_result[f"{result_type}_results"] = response[result_type]["results"]
            
            # Process additional data
            if "infobox" in response:
                search_result["infobox"] = response["infobox"]
            
            if "knowledge_graph" in response:
                search_result["knowledge_graph"] = response["knowledge_graph"]
            
            if "locations" in response:
                search_result["locations"] = response["locations"]
            
            # Add search metadata
            search_result["search_metadata"] = {
                "mixed_type": response.get("type", ""),
                "query_altered": response.get("query", {}).get("altered", False),
                "query_original": response.get("query", {}).get("original", query),
                "spell_corrected": response.get("query", {}).get("spellcheck_off", False),
                "safe_search": response.get("query", {}).get("safesearch", "moderate"),
                "is_navigational": response.get("query", {}).get("is_navigational", False),
                "is_geolocated": response.get("query", {}).get("is_geolocated", False),
                "local_decision": response.get("query", {}).get("local_decision", ""),
            }
            
            return search_result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing Brave search response: {e}")
            
            return {
                "query": query,
                "search_type": config.search_type,
                "web_results": [],
                "news_results": [],
                "image_results": [],
                "video_results": [],
                "place_results": [],
                "infobox": None,
                "knowledge_graph": None,
                "locations": [],
                "search_metadata": {"error": str(e)},
                "execution_time_ms": execution_time,
            }