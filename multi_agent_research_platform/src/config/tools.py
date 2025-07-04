"""
Tool configuration models for ADK built-in tools and MCP servers.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, SecretStr

from .base import BaseConfig


class ToolType(str, Enum):
    """Types of tools supported by the platform."""
    ADK_BUILTIN = "adk_builtin"
    MCP_SERVER = "mcp_server"
    FUNCTION_TOOL = "function_tool"
    AGENT_TOOL = "agent_tool"
    CUSTOM = "custom"


class ADKBuiltInToolType(str, Enum):
    """Types of ADK built-in tools."""
    GOOGLE_SEARCH = "google_search"
    CODE_EXECUTION = "code_execution"
    VERTEX_AI_SEARCH = "vertex_ai_search"
    BIGQUERY_LIST_DATASETS = "bigquery_list_datasets"
    BIGQUERY_GET_DATASET_INFO = "bigquery_get_dataset_info"
    BIGQUERY_LIST_TABLES = "bigquery_list_tables"
    BIGQUERY_GET_TABLE_INFO = "bigquery_get_table_info"
    BIGQUERY_EXECUTE_SQL = "bigquery_execute_sql"


class MCPTransportType(str, Enum):
    """MCP server transport types."""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


class AuthenticationType(str, Enum):
    """Authentication types for external services."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    NONE = "none"


class ToolConfig(BaseModel):
    """Base configuration for all tools."""
    model_config = ConfigDict(extra="ignore")
    
    # Basic identification
    name: str = Field(description="Unique tool name")
    tool_type: ToolType = Field(description="Type of tool")
    description: str = Field(description="Tool description")
    
    # Availability
    enabled: bool = Field(default=True, description="Whether tool is enabled")
    
    # Execution settings
    timeout_seconds: int = Field(default=60, description="Tool execution timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Performance
    is_long_running: bool = Field(default=False, description="Whether tool is long-running")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Usage restrictions
    max_concurrent_calls: int = Field(default=5, description="Maximum concurrent calls")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Tool name must be alphanumeric with underscores/hyphens')
        return v


class ADKBuiltInToolsConfig(BaseConfig):
    """Configuration for ADK built-in tools."""
    
    # Google Search tool
    google_search_enabled: bool = Field(default=True, description="Enable Google Search")
    google_search_model_requirement: str = Field(
        default="gemini-2.0",
        description="Required model version for Google Search"
    )
    
    # Code execution tool
    code_execution_enabled: bool = Field(default=True, description="Enable code execution")
    code_execution_timeout: int = Field(default=30, description="Code execution timeout")
    code_execution_memory_limit_mb: int = Field(
        default=512,
        description="Memory limit for code execution"
    )
    code_execution_allowed_packages: List[str] = Field(
        default_factory=lambda: [
            "numpy", "pandas", "matplotlib", "plotly", "scipy", "requests"
        ],
        description="Allowed packages for code execution"
    )
    
    # Vertex AI Search tool
    vertex_ai_search_enabled: bool = Field(default=True, description="Enable Vertex AI Search")
    vertex_ai_search_data_store_id: Optional[str] = Field(
        default=None,
        description="Vertex AI Search data store ID"
    )
    vertex_ai_search_location: str = Field(
        default="global",
        description="Vertex AI Search location"
    )
    
    # BigQuery tools
    bigquery_enabled: bool = Field(default=True, description="Enable BigQuery tools")
    bigquery_project_id: Optional[str] = Field(
        default=None,
        description="BigQuery project ID"
    )
    bigquery_default_dataset: Optional[str] = Field(
        default=None,
        description="Default BigQuery dataset"
    )
    bigquery_query_timeout: int = Field(
        default=60,
        description="BigQuery query timeout in seconds"
    )
    bigquery_max_results: int = Field(
        default=1000,
        description="Maximum results per query"
    )
    
    def get_enabled_tools(self) -> List[ADKBuiltInToolType]:
        """Get list of enabled ADK built-in tools."""
        enabled = []
        
        if self.google_search_enabled:
            enabled.append(ADKBuiltInToolType.GOOGLE_SEARCH)
        
        if self.code_execution_enabled:
            enabled.append(ADKBuiltInToolType.CODE_EXECUTION)
        
        if self.vertex_ai_search_enabled:
            enabled.append(ADKBuiltInToolType.VERTEX_AI_SEARCH)
        
        if self.bigquery_enabled:
            enabled.extend([
                ADKBuiltInToolType.BIGQUERY_LIST_DATASETS,
                ADKBuiltInToolType.BIGQUERY_GET_DATASET_INFO,
                ADKBuiltInToolType.BIGQUERY_LIST_TABLES,
                ADKBuiltInToolType.BIGQUERY_GET_TABLE_INFO,
                ADKBuiltInToolType.BIGQUERY_EXECUTE_SQL,
            ])
        
        return enabled


class AuthenticationConfig(BaseModel):
    """Authentication configuration for external services."""
    model_config = ConfigDict(extra="ignore")
    
    auth_type: AuthenticationType = Field(description="Authentication type")
    
    # API Key authentication
    api_key: Optional[SecretStr] = Field(default=None, description="API key")
    api_key_header: str = Field(default="Authorization", description="API key header name")
    api_key_prefix: str = Field(default="Bearer", description="API key prefix")
    
    # OAuth2 authentication
    client_id: Optional[str] = Field(default=None, description="OAuth2 client ID")
    client_secret: Optional[SecretStr] = Field(default=None, description="OAuth2 client secret")
    auth_url: Optional[str] = Field(default=None, description="OAuth2 authorization URL")
    token_url: Optional[str] = Field(default=None, description="OAuth2 token URL")
    scopes: List[str] = Field(default_factory=list, description="OAuth2 scopes")
    
    # Basic authentication
    username: Optional[str] = Field(default=None, description="Basic auth username")
    password: Optional[SecretStr] = Field(default=None, description="Basic auth password")
    
    @field_validator('auth_type')
    @classmethod
    def validate_auth_requirements(cls, v, info):
        """Validate that required fields are present based on auth type."""
        values = info.data if info else {}
        
        if v == AuthenticationType.API_KEY and not values.get('api_key'):
            raise ValueError("API key is required for API_KEY authentication")
        
        if v == AuthenticationType.OAUTH2:
            required = ['client_id', 'client_secret', 'auth_url', 'token_url']
            missing = [field for field in required if not values.get(field)]
            if missing:
                raise ValueError(f"OAuth2 authentication missing: {', '.join(missing)}")
        
        if v == AuthenticationType.BASIC_AUTH:
            if not values.get('username') or not values.get('password'):
                raise ValueError("Username and password required for basic authentication")
        
        return v


class MCPServerConfig(BaseModel):
    """Configuration for MCP (Model Context Protocol) servers."""
    model_config = ConfigDict(extra="ignore")
    
    # Server identification
    name: str = Field(description="MCP server name")
    description: str = Field(description="Server description")
    
    # Connection settings
    transport: MCPTransportType = Field(description="Transport protocol")
    endpoint: Optional[str] = Field(default=None, description="Server endpoint URL")
    command: Optional[List[str]] = Field(default=None, description="Command to start server")
    
    # Authentication
    authentication: Optional[AuthenticationConfig] = Field(
        default=None,
        description="Authentication configuration"
    )
    
    # Connection management
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    read_timeout: int = Field(default=60, description="Read timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum connection retries")
    
    # Health monitoring
    health_check_interval: int = Field(
        default=60,
        description="Health check interval in seconds"
    )
    health_check_endpoint: Optional[str] = Field(
        default=None,
        description="Health check endpoint"
    )
    
    # Performance
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent requests"
    )
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Features
    supports_notifications: bool = Field(
        default=False,
        description="Whether server supports notifications"
    )
    supports_completion: bool = Field(
        default=False,
        description="Whether server supports completion"
    )
    
    @field_validator('transport')
    @classmethod
    def validate_transport_config(cls, v, info):
        """Validate transport-specific configuration."""
        values = info.data if info else {}
        
        if v == MCPTransportType.SSE and not values.get('endpoint'):
            raise ValueError("SSE transport requires endpoint URL")
        
        if v == MCPTransportType.STDIO and not values.get('command'):
            raise ValueError("STDIO transport requires command")
        
        return v


class MCPToolConfig(ToolConfig):
    """Configuration for tools provided by MCP servers."""
    
    tool_type: ToolType = Field(default=ToolType.MCP_SERVER, description="Tool type")
    
    # MCP server reference
    server_name: str = Field(description="Name of MCP server providing this tool")
    
    # Tool-specific settings
    function_name: str = Field(description="Function name on the MCP server")
    parameters_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for tool parameters"
    )
    
    # Result handling
    parse_result: bool = Field(default=True, description="Parse tool result as JSON")
    result_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Expected result schema"
    )
    
    # Error handling
    ignore_errors: bool = Field(default=False, description="Ignore tool execution errors")
    default_error_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Default response on error"
    )


class ToolRegistry(BaseConfig):
    """Registry of all tools available in the system."""
    
    # Built-in tools configuration
    adk_builtin: ADKBuiltInToolsConfig = Field(
        default_factory=ADKBuiltInToolsConfig,
        description="ADK built-in tools configuration"
    )
    
    # MCP servers
    mcp_servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict,
        description="MCP server configurations"
    )
    
    # MCP tools
    mcp_tools: Dict[str, MCPToolConfig] = Field(
        default_factory=dict,
        description="MCP tool configurations"
    )
    
    # Function tools
    function_tools: Dict[str, ToolConfig] = Field(
        default_factory=dict,
        description="Custom function tool configurations"
    )
    
    # Global settings
    global_timeout: int = Field(default=60, description="Global tool timeout")
    enable_tool_caching: bool = Field(default=True, description="Enable tool result caching")
    max_tools_per_agent: int = Field(default=20, description="Maximum tools per agent")
    
    def add_mcp_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server to the registry."""
        self.mcp_servers[config.name] = config
    
    def add_mcp_tool(self, config: MCPToolConfig) -> None:
        """Add an MCP tool to the registry."""
        if config.server_name not in self.mcp_servers:
            raise ValueError(f"MCP server '{config.server_name}' not found")
        
        self.mcp_tools[config.name] = config
    
    def get_tools_for_agent(self, agent_name: str, tool_names: List[str]) -> List[ToolConfig]:
        """Get tool configurations for a specific agent."""
        tools = []
        
        for tool_name in tool_names:
            # Check MCP tools first
            if tool_name in self.mcp_tools:
                tools.append(self.mcp_tools[tool_name])
            # Check function tools
            elif tool_name in self.function_tools:
                tools.append(self.function_tools[tool_name])
            # Check if it's an ADK built-in tool
            elif tool_name in [t.value for t in ADKBuiltInToolType]:
                # Create a basic config for built-in tools
                tools.append(ToolConfig(
                    name=tool_name,
                    tool_type=ToolType.ADK_BUILTIN,
                    description=f"ADK built-in tool: {tool_name}"
                ))
        
        return tools
    
    def validate_tool_dependencies(self) -> None:
        """Validate that all tool dependencies are satisfied."""
        # Check MCP tool server references
        for tool_name, tool_config in self.mcp_tools.items():
            if tool_config.server_name not in self.mcp_servers:
                raise ValueError(
                    f"MCP tool '{tool_name}' references unknown server '{tool_config.server_name}'"
                )
    
    def create_default_mcp_servers(self) -> None:
        """Create default MCP server configurations."""
        
        # Perplexity MCP Server
        self.add_mcp_server(MCPServerConfig(
            name="perplexity",
            description="Perplexity AI search and conversation",
            transport=MCPTransportType.SSE,
            endpoint="https://api.perplexity.ai/mcp",
            authentication=AuthenticationConfig(
                auth_type=AuthenticationType.API_KEY,
                api_key_header="Authorization",
                api_key_prefix="Bearer"
            ),
            supports_completion=True,
            max_concurrent_requests=5
        ))
        
        # Tavily MCP Server
        self.add_mcp_server(MCPServerConfig(
            name="tavily",
            description="Tavily factual search with citations",
            transport=MCPTransportType.SSE,
            endpoint="https://api.tavily.com/mcp",
            authentication=AuthenticationConfig(
                auth_type=AuthenticationType.API_KEY,
                api_key_header="X-API-Key"
            ),
            max_concurrent_requests=10
        ))
        
        # Brave Search MCP Server
        self.add_mcp_server(MCPServerConfig(
            name="brave_search",
            description="Brave Search privacy-focused search",
            transport=MCPTransportType.SSE,
            endpoint="https://api.search.brave.com/mcp",
            authentication=AuthenticationConfig(
                auth_type=AuthenticationType.API_KEY,
                api_key_header="X-Subscription-Token"
            ),
            max_concurrent_requests=15
        ))
        
        # Omnisearch MCP Server
        self.add_mcp_server(MCPServerConfig(
            name="omnisearch",
            description="Unified multi-provider search",
            transport=MCPTransportType.SSE,
            endpoint="https://api.omnisearch.ai/mcp",
            authentication=AuthenticationConfig(
                auth_type=AuthenticationType.API_KEY,
                api_key_header="Authorization",
                api_key_prefix="Bearer"
            ),
            supports_notifications=True,
            max_concurrent_requests=8
        ))
    
    def create_default_mcp_tools(self) -> None:
        """Create default MCP tool configurations."""
        
        # Perplexity tools
        self.add_mcp_tool(MCPToolConfig(
            name="perplexity_search",
            server_name="perplexity",
            function_name="search",
            description="Conversational AI-powered search with Perplexity",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "focus": {"type": "string", "enum": ["academic", "news", "general"]}
                },
                "required": ["query"]
            }
        ))
        
        # Tavily tools
        self.add_mcp_tool(MCPToolConfig(
            name="tavily_search",
            server_name="tavily",
            function_name="search",
            description="Factual search with citations using Tavily",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "include_citations": {"type": "boolean", "default": True}
                },
                "required": ["query"]
            }
        ))
        
        # Brave Search tools
        self.add_mcp_tool(MCPToolConfig(
            name="brave_search",
            server_name="brave_search",
            function_name="web_search",
            description="Privacy-focused web search using Brave",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "country": {"type": "string", "default": "US"},
                    "search_lang": {"type": "string", "default": "en"}
                },
                "required": ["query"]
            }
        ))
        
        # Omnisearch tools
        self.add_mcp_tool(MCPToolConfig(
            name="omnisearch_unified",
            server_name="omnisearch",
            function_name="unified_search",
            description="Unified search across multiple providers",
            parameters_schema={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search providers to use"
                    }
                },
                "required": ["query"]
            }
        ))