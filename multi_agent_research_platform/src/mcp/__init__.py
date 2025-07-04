"""
Model Context Protocol (MCP) Server Integration for Multi-Agent Research Platform

This module provides comprehensive MCP server integration for external services:
- Perplexity AI: Advanced AI-powered search and research
- Tavily: Web search and research API with real-time data
- Brave Search: Privacy-focused search with independent indexing
- Omnisearch: Universal search across multiple data sources

All MCP servers integrate with the platform's context management patterns, logging system,
service architecture, and ADK built-in tools for sophisticated multi-source workflows.
"""

from .base import (
    MCPServer,
    MCPServerConfig,
    MCPServerResult,
    MCPServerStatus,
    MCPAuthConfig,
)
from .servers.perplexity import (
    PerplexityServer,
    PerplexityConfig,
    PerplexitySearchResult,
)
from .servers.tavily import (
    TavilyServer,
    TavilyConfig,
    TavilySearchResult,
)
from .servers.brave import (
    BraveServer,
    BraveConfig,
    BraveSearchResult,
)
from .servers.omnisearch import (
    OmnisearchServer,
    OmnisearchConfig,
    OmnisearchResult,
)
from .orchestrator import (
    MCPOrchestrator,
    MultiSourceSearchResult,
    SearchStrategy,
)
from .factory import (
    MCPServerFactory,
    create_mcp_server,
    create_search_suite,
)

__all__ = [
    # Base classes
    "MCPServer",
    "MCPServerConfig",
    "MCPServerResult", 
    "MCPServerStatus",
    "MCPAuthConfig",
    
    # Perplexity
    "PerplexityServer",
    "PerplexityConfig",
    "PerplexitySearchResult",
    
    # Tavily
    "TavilyServer", 
    "TavilyConfig",
    "TavilySearchResult",
    
    # Brave
    "BraveServer",
    "BraveConfig", 
    "BraveSearchResult",
    
    # Omnisearch
    "OmnisearchServer",
    "OmnisearchConfig",
    "OmnisearchResult",
    
    # Orchestration
    "MCPOrchestrator",
    "MultiSourceSearchResult",
    "SearchStrategy",
    
    # Factory
    "MCPServerFactory",
    "create_mcp_server",
    "create_search_suite",
]