"""
ADK Built-in Tools Integration for Multi-Agent Research Platform

This module provides comprehensive integration with Google ADK's built-in tools:
- Google Search: Built-in search capabilities with result processing
- Code Execution: Built-in code execution environment with safety measures
- Vertex AI Search: Enterprise search with embeddings and indexing
- BigQuery: Database query capabilities with result processing

All tools integrate with the platform's context management patterns, logging system,
and service architecture for comprehensive monitoring and debugging.
"""

from .base import (
    BaseTool,
    ToolExecutionResult,
    ToolAuthConfig,
    ToolResult,
)
from .google_search import (
    GoogleSearchTool,
    SearchResult,
    SearchQuery,
)
from .code_execution import (
    CodeExecutionTool,
    CodeExecutionResult,
    CodeExecutionConfig,
)
from .vertex_search import (
    VertexSearchTool,
    VertexSearchResult,
    VertexSearchConfig,
)
from .bigquery import (
    BigQueryTool,
    BigQueryResult,
    BigQueryConfig,
)
from .factory import (
    ToolFactory,
    get_tool,
    create_tool_suite,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolExecutionResult", 
    "ToolAuthConfig",
    "ToolResult",
    
    # Google Search
    "GoogleSearchTool",
    "SearchResult",
    "SearchQuery",
    
    # Code Execution
    "CodeExecutionTool",
    "CodeExecutionResult",
    "CodeExecutionConfig",
    
    # Vertex AI Search
    "VertexSearchTool",
    "VertexSearchResult", 
    "VertexSearchConfig",
    
    # BigQuery
    "BigQueryTool",
    "BigQueryResult",
    "BigQueryConfig",
    
    # Factory
    "ToolFactory",
    "get_tool",
    "create_tool_suite",
]