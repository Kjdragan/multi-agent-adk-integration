"""
Perplexity AI MCP Server Integration

Provides comprehensive integration with Perplexity AI's advanced search and research capabilities,
including AI-powered analysis, real-time data access, and contextual understanding.
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


class PerplexityModel(str, Enum):
    """Available Perplexity AI models."""
    SONAR_SMALL_CHAT = "sonar-small-chat"
    SONAR_SMALL_ONLINE = "sonar-small-online"
    SONAR_MEDIUM_CHAT = "sonar-medium-chat"
    SONAR_MEDIUM_ONLINE = "sonar-medium-online"
    SONAR_PRO = "sonar-pro"


class PerplexitySearchType(str, Enum):
    """Types of Perplexity searches."""
    GENERAL = "general"          # General web search
    ACADEMIC = "academic"        # Academic and research focused
    NEWS = "news"               # Recent news and events
    TECHNICAL = "technical"     # Technical and programming focused
    FINANCIAL = "financial"     # Financial and market data


@dataclass
class PerplexityConfig:
    """Configuration for Perplexity AI searches."""
    model: PerplexityModel = PerplexityModel.SONAR_MEDIUM_ONLINE
    search_type: PerplexitySearchType = PerplexitySearchType.GENERAL
    max_tokens: int = 2000
    temperature: float = 0.3
    top_p: float = 0.9
    return_citations: bool = True
    return_images: bool = False
    search_domain_filter: Optional[List[str]] = None
    search_recency_filter: Optional[str] = None  # "day", "week", "month", "year"
    frequency_penalty: float = 1.0
    presence_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "model": self.model.value,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "return_citations": self.return_citations,
            "return_images": self.return_images,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        
        # Add optional filters
        if self.search_domain_filter:
            config["search_domain_filter"] = self.search_domain_filter
        if self.search_recency_filter:
            config["search_recency_filter"] = self.search_recency_filter
        
        return config


@dataclass
class PerplexityCitation:
    """Citation from Perplexity search result."""
    number: int
    url: str
    title: str
    snippet: str = ""
    domain: str = ""
    publication_date: Optional[str] = None
    
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
            "number": self.number,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "domain": self.domain,
            "publication_date": self.publication_date,
        }


@dataclass
class PerplexitySearchResult:
    """Result from Perplexity AI search."""
    query: str
    answer: str
    model_used: str
    search_type: PerplexitySearchType
    citations: List[PerplexityCitation] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    related_questions: List[str] = field(default_factory=list)
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    
    @property
    def citation_count(self) -> int:
        """Number of citations in the result."""
        return len(self.citations)
    
    @property
    def unique_domains(self) -> List[str]:
        """List of unique domains cited."""
        return list(set(citation.domain for citation in self.citations))
    
    @property
    def answer_length(self) -> int:
        """Length of the answer in characters."""
        return len(self.answer)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "answer": self.answer,
            "model_used": self.model_used,
            "search_type": self.search_type.value,
            "citations": [citation.to_dict() for citation in self.citations],
            "images": self.images,
            "related_questions": self.related_questions,
            "search_metadata": self.search_metadata,
            "execution_time_ms": self.execution_time_ms,
            "token_usage": self.token_usage,
            "citation_count": self.citation_count,
            "unique_domains": self.unique_domains,
            "answer_length": self.answer_length,
        }


class PerplexityServer(HTTPMCPServer):
    """
    Perplexity AI MCP server integration.
    
    Provides comprehensive Perplexity AI capabilities including AI-powered search,
    contextual analysis, real-time data access, and citation management.
    """
    
    def __init__(self,
                 api_key: str,
                 default_config: Optional[PerplexityConfig] = None,
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
            server_name="perplexity",
            server_type="ai_search",
            base_url="https://api.perplexity.ai",
            auth_config=auth_config,
            rate_limit_requests_per_minute=20,  # Perplexity rate limits
            rate_limit_requests_per_hour=500,
            timeout_seconds=60,  # AI operations can take longer
            enable_caching=True,
            cache_ttl_seconds=600,  # 10 minutes for AI results
        )
        
        super().__init__(
            config=server_config,
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        # Perplexity-specific configuration
        self.default_config = default_config or PerplexityConfig()
        
        # Search optimization settings
        self.query_enhancement = True
        self.context_awareness = True
        self.citation_validation = True
    
    async def _connect_server(self) -> bool:
        """Test connection to Perplexity API."""
        try:
            # Test with a simple query
            test_response = await self.make_http_request(
                method="POST",
                endpoint="/chat/completions",
                json_data={
                    "model": "sonar-small-chat",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10,
                }
            )
            
            return "choices" in test_response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Perplexity connection test failed: {e}")
            return False
    
    async def _authenticate(self) -> bool:
        """Perplexity uses API key in headers, so connection test serves as auth."""
        return True
    
    async def _execute_operation(self, 
                               operation_type: MCPOperationType, 
                               parameters: Dict[str, Any]) -> Any:
        """Execute Perplexity-specific operations."""
        if operation_type == MCPOperationType.SEARCH:
            return await self._search(parameters)
        elif operation_type == MCPOperationType.ANALYZE:
            return await self._analyze(parameters)
        elif operation_type == MCPOperationType.SUMMARIZE:
            return await self._summarize(parameters)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get Perplexity server capabilities."""
        return {
            "server_type": "ai_search",
            "ai_powered": True,
            "real_time_data": True,
            "citations": True,
            "supported_operations": [
                MCPOperationType.SEARCH.value,
                MCPOperationType.ANALYZE.value,
                MCPOperationType.SUMMARIZE.value,
            ],
            "models": [model.value for model in PerplexityModel],
            "search_types": [search_type.value for search_type in PerplexitySearchType],
            "features": {
                "query_enhancement": self.query_enhancement,
                "context_awareness": self.context_awareness,
                "citation_validation": self.citation_validation,
                "real_time_search": True,
                "academic_search": True,
                "news_search": True,
                "technical_search": True,
            },
        }
    
    async def search(self, 
                    query: str,
                    search_config: Optional[PerplexityConfig] = None,
                    context: Optional[str] = None) -> PerplexitySearchResult:
        """
        Perform AI-powered search with Perplexity.
        
        Args:
            query: Search query
            search_config: Search configuration
            context: Additional context for the query
            
        Returns:
            Comprehensive search result with AI analysis
        """
        parameters = {
            "query": query,
            "config": search_config or self.default_config,
            "context": context,
        }
        
        result = await self.execute_operation(
            MCPOperationType.SEARCH,
            parameters
        )
        
        if result.success and result.data:
            return PerplexitySearchResult(**result.data)
        else:
            # Return empty result with error information
            return PerplexitySearchResult(
                query=query,
                answer=f"Search failed: {result.error}",
                model_used="error",
                search_type=search_config.search_type if search_config else self.default_config.search_type,
            )
    
    async def analyze_with_context(self,
                                 content: str,
                                 analysis_prompt: str,
                                 model: Optional[PerplexityModel] = None) -> PerplexitySearchResult:
        """
        Analyze content with AI-powered contextual understanding.
        
        Args:
            content: Content to analyze
            analysis_prompt: Analysis instructions
            model: Model to use for analysis
            
        Returns:
            Analysis result with insights and citations
        """
        parameters = {
            "content": content,
            "prompt": analysis_prompt,
            "model": model or self.default_config.model,
        }
        
        result = await self.execute_operation(
            MCPOperationType.ANALYZE,
            parameters
        )
        
        if result.success and result.data:
            return PerplexitySearchResult(**result.data)
        else:
            return PerplexitySearchResult(
                query=analysis_prompt,
                answer=f"Analysis failed: {result.error}",
                model_used="error",
                search_type=PerplexitySearchType.GENERAL,
            )
    
    async def multi_perspective_search(self,
                                     topic: str,
                                     perspectives: List[str],
                                     search_config: Optional[PerplexityConfig] = None) -> Dict[str, PerplexitySearchResult]:
        """
        Search a topic from multiple perspectives using Perplexity AI.
        
        Args:
            topic: Main topic to research
            perspectives: List of different perspectives or angles
            search_config: Search configuration
            
        Returns:
            Dictionary mapping perspective to search result
        """
        results = {}
        
        for perspective in perspectives:
            enhanced_query = f"{topic} from {perspective} perspective"
            
            try:
                search_result = await self.search(
                    query=enhanced_query,
                    search_config=search_config,
                )
                results[perspective] = search_result
                
                # Store in memory for cross-perspective analysis
                await self.store_results_in_memory(
                    MCPOperationType.SEARCH,
                    enhanced_query,
                    search_result.to_dict()
                )
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Multi-perspective search failed for {perspective}: {e}")
                
                results[perspective] = PerplexitySearchResult(
                    query=enhanced_query,
                    answer=f"Search failed: {str(e)}",
                    model_used="error",
                    search_type=search_config.search_type if search_config else self.default_config.search_type,
                )
        
        return results
    
    async def _search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Internal search implementation."""
        start_time = time.time()
        
        query = parameters["query"]
        config = parameters.get("config", self.default_config)
        context = parameters.get("context")
        
        # Enhance query if enabled
        if self.query_enhancement:
            enhanced_query = self._enhance_query(query, config.search_type)
        else:
            enhanced_query = query
        
        # Prepare messages for Perplexity API
        messages = []
        
        if context and self.context_awareness:
            messages.append({
                "role": "system",
                "content": f"Context: {context}. Please provide a comprehensive answer with citations."
            })
        
        messages.append({
            "role": "user",
            "content": enhanced_query
        })
        
        # Prepare API request
        api_config = config.to_dict()
        api_config["messages"] = messages
        
        try:
            # Make API call
            response = await self.make_http_request(
                method="POST",
                endpoint="/chat/completions",
                json_data=api_config
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
                self.logger.error(f"Perplexity search failed: {e}")
            raise
    
    async def _analyze(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Internal analysis implementation."""
        start_time = time.time()
        
        content = parameters["content"]
        prompt = parameters["prompt"] 
        model = parameters.get("model", self.default_config.model)
        
        # Prepare analysis messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert analyst. Provide detailed analysis with supporting evidence and citations where applicable."
            },
            {
                "role": "user", 
                "content": f"Content to analyze:\n{content}\n\nAnalysis request: {prompt}"
            }
        ]
        
        api_config = {
            "model": model.value,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.3,
            "return_citations": True,
        }
        
        try:
            response = await self.make_http_request(
                method="POST",
                endpoint="/chat/completions",
                json_data=api_config
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Process analysis response
            analysis_result = self._process_analysis_response(
                response, prompt, model, execution_time
            )
            
            self.log_operation("analyze", parameters, None)
            
            return analysis_result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Perplexity analysis failed: {e}")
            raise
    
    async def _summarize(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Internal summarization implementation."""
        start_time = time.time()
        
        content = parameters["content"]
        summary_length = parameters.get("length", "medium")
        focus_areas = parameters.get("focus_areas", [])
        
        # Prepare summarization prompt
        prompt = f"Provide a {summary_length} summary of the following content"
        if focus_areas:
            prompt += f", focusing on: {', '.join(focus_areas)}"
        prompt += ".\n\nContent:\n" + content
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at creating concise, accurate summaries that capture key insights and maintain important details."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        api_config = {
            "model": self.default_config.model.value,
            "messages": messages,
            "max_tokens": 1000 if summary_length == "short" else 2000,
            "temperature": 0.2,
        }
        
        try:
            response = await self.make_http_request(
                method="POST",
                endpoint="/chat/completions", 
                json_data=api_config
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Process summary response
            summary_result = self._process_summary_response(
                response, prompt, execution_time
            )
            
            self.log_operation("summarize", parameters, None)
            
            return summary_result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Perplexity summarization failed: {e}")
            raise
    
    def _enhance_query(self, query: str, search_type: PerplexitySearchType) -> str:
        """Enhance query based on search type."""
        enhancements = {
            PerplexitySearchType.ACADEMIC: f"academic research literature {query}",
            PerplexitySearchType.NEWS: f"recent news developments {query}",
            PerplexitySearchType.TECHNICAL: f"technical documentation implementation {query}",
            PerplexitySearchType.FINANCIAL: f"financial analysis market data {query}",
            PerplexitySearchType.GENERAL: query,
        }
        
        return enhancements.get(search_type, query)
    
    def _process_search_response(self, 
                               response: Dict[str, Any],
                               query: str,
                               config: PerplexityConfig,
                               execution_time: float) -> Dict[str, Any]:
        """Process Perplexity search API response."""
        try:
            choice = response["choices"][0]
            message = choice["message"]
            
            # Extract answer
            answer = message["content"]
            
            # Extract citations if available
            citations = []
            if config.return_citations and "citations" in choice:
                for i, citation_data in enumerate(choice["citations"]):
                    citation = PerplexityCitation(
                        number=i + 1,
                        url=citation_data.get("url", ""),
                        title=citation_data.get("title", ""),
                        snippet=citation_data.get("snippet", ""),
                        publication_date=citation_data.get("date"),
                    )
                    citations.append(citation)
            
            # Extract usage information
            token_usage = response.get("usage", {})
            
            return {
                "query": query,
                "answer": answer,
                "model_used": config.model.value,
                "search_type": config.search_type,
                "citations": [c.to_dict() for c in citations],
                "images": choice.get("images", []),
                "related_questions": choice.get("related_questions", []),
                "search_metadata": {
                    "finish_reason": choice.get("finish_reason", ""),
                    "response_id": response.get("id", ""),
                },
                "execution_time_ms": execution_time,
                "token_usage": token_usage,
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing Perplexity search response: {e}")
            
            return {
                "query": query,
                "answer": f"Error processing response: {str(e)}",
                "model_used": config.model.value,
                "search_type": config.search_type,
                "citations": [],
                "images": [],
                "related_questions": [],
                "search_metadata": {"error": str(e)},
                "execution_time_ms": execution_time,
                "token_usage": {},
            }
    
    def _process_analysis_response(self,
                                 response: Dict[str, Any],
                                 prompt: str,
                                 model: PerplexityModel,
                                 execution_time: float) -> Dict[str, Any]:
        """Process Perplexity analysis API response."""
        try:
            choice = response["choices"][0]
            message = choice["message"]
            
            return {
                "query": prompt,
                "answer": message["content"],
                "model_used": model.value,
                "search_type": PerplexitySearchType.GENERAL,
                "citations": [],
                "images": [],
                "related_questions": [],
                "search_metadata": {
                    "operation": "analysis",
                    "finish_reason": choice.get("finish_reason", ""),
                },
                "execution_time_ms": execution_time,
                "token_usage": response.get("usage", {}),
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing Perplexity analysis response: {e}")
            raise
    
    def _process_summary_response(self,
                                response: Dict[str, Any],
                                prompt: str,
                                execution_time: float) -> Dict[str, Any]:
        """Process Perplexity summary API response."""
        try:
            choice = response["choices"][0]
            message = choice["message"]
            
            return {
                "query": prompt,
                "answer": message["content"],
                "model_used": self.default_config.model.value,
                "search_type": PerplexitySearchType.GENERAL,
                "citations": [],
                "images": [],
                "related_questions": [],
                "search_metadata": {
                    "operation": "summarization",
                    "finish_reason": choice.get("finish_reason", ""),
                },
                "execution_time_ms": execution_time,
                "token_usage": response.get("usage", {}),
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing Perplexity summary response: {e}")
            raise