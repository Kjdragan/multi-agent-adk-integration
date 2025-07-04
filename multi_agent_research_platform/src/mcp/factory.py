"""
MCP Server Factory for Easy Creation and Management

Provides factory methods and utilities for creating and managing MCP servers
and orchestrator with proper configuration and service integration.
"""

from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum

from .base import MCPServer, MCPServerConfig, MCPAuthConfig
from .servers.perplexity import PerplexityServer, PerplexityConfig
from .servers.tavily import TavilyServer, TavilyConfig
from .servers.brave import BraveServer, BraveConfig
from .servers.omnisearch import OmnisearchServer, OmnisearchConfig
from .orchestrator import MCPOrchestrator, SearchStrategy
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..config.tools import ToolRegistry


class MCPServerType(str, Enum):
    """Types of MCP servers available."""
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    BRAVE = "brave"
    OMNISEARCH = "omnisearch"


class MCPSuite(str, Enum):
    """Predefined MCP server suites for different use cases."""
    RESEARCH = "research"           # Perplexity + Tavily
    WEB_SEARCH = "web_search"      # Brave + Tavily
    AI_POWERED = "ai_powered"      # Perplexity + Omnisearch
    COMPREHENSIVE = "comprehensive" # All servers


class MCPServerFactory:
    """
    Factory for creating and managing MCP servers and orchestrator.
    
    Provides centralized configuration, authentication, and service integration
    for all MCP servers with convenient creation patterns.
    """
    
    def __init__(self,
                 config: Optional[ToolRegistry] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        self.config = config
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Server instances cache
        self._server_instances: Dict[MCPServerType, MCPServer] = {}
        self._orchestrator_instance: Optional[MCPOrchestrator] = None
        
        # Default API configurations
        self._api_configs = {
            MCPServerType.PERPLEXITY: self._get_perplexity_config(),
            MCPServerType.TAVILY: self._get_tavily_config(),
            MCPServerType.BRAVE: self._get_brave_config(),
            MCPServerType.OMNISEARCH: self._get_omnisearch_config(),
        }
    
    def create_server(self, 
                     server_type: MCPServerType,
                     api_key: Optional[str] = None,
                     custom_config: Optional[Dict[str, Any]] = None) -> Optional[MCPServer]:
        """
        Create a specific MCP server with configuration.
        
        Args:
            server_type: Type of MCP server to create
            api_key: API key for the service
            custom_config: Custom configuration for the server
            
        Returns:
            Configured MCP server instance or None if creation fails
        """
        # Get API key from config if not provided
        if not api_key:
            api_key = self._get_api_key_from_config(server_type)
        
        if not api_key:
            if self.logger:
                self.logger.warning(f"No API key available for {server_type.value}")
            return None
        
        # Common server arguments
        server_args = {
            "logger": self.logger,
            "session_service": self.session_service,
            "memory_service": self.memory_service,
            "artifact_service": self.artifact_service,
        }
        
        # Add custom configuration
        if custom_config:
            server_args.update(custom_config)
        
        try:
            # Create server based on type
            if server_type == MCPServerType.PERPLEXITY:
                return self._create_perplexity_server(api_key, **server_args)
            elif server_type == MCPServerType.TAVILY:
                return self._create_tavily_server(api_key, **server_args)
            elif server_type == MCPServerType.BRAVE:
                return self._create_brave_server(api_key, **server_args)
            elif server_type == MCPServerType.OMNISEARCH:
                return self._create_omnisearch_server(api_key, **server_args)
            else:
                raise ValueError(f"Unsupported server type: {server_type}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create {server_type.value} server: {e}")
            return None
    
    def get_server(self, 
                  server_type: MCPServerType,
                  create_if_missing: bool = True,
                  **kwargs) -> Optional[MCPServer]:
        """
        Get a server instance, creating it if necessary.
        
        Args:
            server_type: Type of server to get
            create_if_missing: Whether to create the server if not cached
            **kwargs: Additional arguments for server creation
            
        Returns:
            Server instance or None if not available
        """
        # Check cache first
        if server_type in self._server_instances:
            return self._server_instances[server_type]
        
        # Create if requested
        if create_if_missing:
            server = self.create_server(server_type, **kwargs)
            if server:
                self._server_instances[server_type] = server
            return server
        
        return None
    
    def create_suite(self, 
                    suite_type: MCPSuite,
                    api_keys: Optional[Dict[str, str]] = None,
                    custom_configs: Optional[Dict[MCPServerType, Dict[str, Any]]] = None) -> Dict[MCPServerType, MCPServer]:
        """
        Create a predefined suite of MCP servers.
        
        Args:
            suite_type: Type of server suite to create
            api_keys: Dictionary of API keys for each service
            custom_configs: Custom configurations for specific servers
            
        Returns:
            Dictionary of server type to server instance
        """
        suite_servers = {}
        api_keys = api_keys or {}
        custom_configs = custom_configs or {}
        
        # Define server suites
        suite_definitions = {
            MCPSuite.RESEARCH: [
                MCPServerType.PERPLEXITY,
                MCPServerType.TAVILY,
            ],
            MCPSuite.WEB_SEARCH: [
                MCPServerType.BRAVE,
                MCPServerType.TAVILY,
            ],
            MCPSuite.AI_POWERED: [
                MCPServerType.PERPLEXITY,
                MCPServerType.OMNISEARCH,
            ],
            MCPSuite.COMPREHENSIVE: [
                MCPServerType.PERPLEXITY,
                MCPServerType.TAVILY,
                MCPServerType.BRAVE,
                MCPServerType.OMNISEARCH,
            ],
        }
        
        server_types = suite_definitions.get(suite_type, [])
        
        for server_type in server_types:
            try:
                api_key = api_keys.get(server_type.value)
                custom_config = custom_configs.get(server_type, {})
                
                server = self.create_server(server_type, api_key, custom_config)
                if server:
                    suite_servers[server_type] = server
                    
                    if self.logger:
                        self.logger.info(f"Created {server_type.value} for {suite_type.value} suite")
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to create {server_type.value} for suite: {e}")
        
        return suite_servers
    
    def create_orchestrator(self,
                          servers: Optional[Dict[MCPServerType, MCPServer]] = None,
                          auto_create_servers: bool = True) -> Optional[MCPOrchestrator]:
        """
        Create MCP orchestrator with server integration.
        
        Args:
            servers: Dictionary of servers to include
            auto_create_servers: Whether to auto-create available servers
            
        Returns:
            Configured MCP orchestrator
        """
        if self._orchestrator_instance:
            return self._orchestrator_instance
        
        # Use provided servers or get from cache
        if servers is None:
            servers = self._server_instances.copy()
        
        # Auto-create servers if requested and none provided
        if auto_create_servers and not servers:
            servers = self.create_suite(MCPSuite.COMPREHENSIVE)
        
        try:
            orchestrator = MCPOrchestrator(
                perplexity_server=servers.get(MCPServerType.PERPLEXITY),
                tavily_server=servers.get(MCPServerType.TAVILY),
                brave_server=servers.get(MCPServerType.BRAVE),
                omnisearch_server=servers.get(MCPServerType.OMNISEARCH),
                logger=self.logger,
                session_service=self.session_service,
                memory_service=self.memory_service,
                artifact_service=self.artifact_service,
            )
            
            self._orchestrator_instance = orchestrator
            return orchestrator
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create MCP orchestrator: {e}")
            return None
    
    def get_server_capabilities(self) -> Dict[MCPServerType, Dict[str, Any]]:
        """
        Get capabilities information for all server types.
        
        Returns:
            Dictionary mapping server types to their capabilities
        """
        capabilities = {}
        
        for server_type in MCPServerType:
            try:
                # Create temporary server instance to get capabilities
                temp_server = self.create_server(server_type)
                if temp_server:
                    capabilities[server_type] = temp_server.get_capabilities()
                else:
                    capabilities[server_type] = {
                        "available": False,
                        "error": "Could not create server instance",
                    }
            except Exception as e:
                capabilities[server_type] = {
                    "available": False,
                    "error": str(e),
                }
                if self.logger:
                    self.logger.warning(f"Could not get capabilities for {server_type.value}: {e}")
        
        return capabilities
    
    def validate_server_requirements(self) -> Dict[MCPServerType, Dict[str, Any]]:
        """
        Validate requirements for all server types (API keys, etc.).
        
        Returns:
            Dictionary with validation results for each server type
        """
        validation_results = {}
        
        for server_type in MCPServerType:
            result = {
                "valid": False,
                "issues": [],
                "api_key_configured": False,
                "config_valid": False,
            }
            
            try:
                # Check API key availability
                api_key = self._get_api_key_from_config(server_type)
                if api_key:
                    result["api_key_configured"] = True
                else:
                    result["issues"].append(f"No API key configured for {server_type.value}")
                
                # Check server-specific configuration
                server_config = self._api_configs.get(server_type, {})
                if server_config:
                    result["config_valid"] = True
                else:
                    result["issues"].append(f"No configuration found for {server_type.value}")
                
                # Overall validation
                result["valid"] = len(result["issues"]) == 0
                
            except Exception as e:
                result["issues"].append(f"Validation error: {str(e)}")
            
            validation_results[server_type] = result
        
        return validation_results
    
    def _get_api_key_from_config(self, server_type: MCPServerType) -> Optional[str]:
        """Get API key from configuration for a specific server type."""
        if not self.config:
            return None
        
        # Define config key mappings
        config_keys = {
            MCPServerType.PERPLEXITY: "perplexity_api_key",
            MCPServerType.TAVILY: "tavily_api_key",
            MCPServerType.BRAVE: "brave_api_key",
            MCPServerType.OMNISEARCH: "omnisearch_api_key",
        }
        
        config_key = config_keys.get(server_type)
        if config_key:
            return getattr(self.config, config_key, None)
        
        return None
    
    def _create_perplexity_server(self, api_key: str, **kwargs) -> PerplexityServer:
        """Create Perplexity server with default configuration."""
        default_config = PerplexityConfig()
        
        # Extract custom config if provided
        custom_config = kwargs.pop('default_config', None)
        if custom_config:
            if isinstance(custom_config, dict):
                default_config = PerplexityConfig(**custom_config)
            elif isinstance(custom_config, PerplexityConfig):
                default_config = custom_config
        
        server_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['logger', 'session_service', 'memory_service', 'artifact_service']}
        
        return PerplexityServer(
            api_key=api_key,
            default_config=default_config,
            **server_kwargs
        )
    
    def _create_tavily_server(self, api_key: str, **kwargs) -> TavilyServer:
        """Create Tavily server with default configuration."""
        default_config = TavilyConfig()
        
        # Extract custom config if provided
        custom_config = kwargs.pop('default_config', None)
        if custom_config:
            if isinstance(custom_config, dict):
                default_config = TavilyConfig(**custom_config)
            elif isinstance(custom_config, TavilyConfig):
                default_config = custom_config
        
        server_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['logger', 'session_service', 'memory_service', 'artifact_service']}
        
        return TavilyServer(
            api_key=api_key,
            default_config=default_config,
            **server_kwargs
        )
    
    def _create_brave_server(self, api_key: str, **kwargs) -> BraveServer:
        """Create Brave server with default configuration."""
        default_config = BraveConfig()
        
        # Extract custom config if provided
        custom_config = kwargs.pop('default_config', None)
        if custom_config:
            if isinstance(custom_config, dict):
                default_config = BraveConfig(**custom_config)
            elif isinstance(custom_config, BraveConfig):
                default_config = custom_config
        
        server_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['logger', 'session_service', 'memory_service', 'artifact_service']}
        
        return BraveServer(
            api_key=api_key,
            default_config=default_config,
            **server_kwargs
        )
    
    def _create_omnisearch_server(self, api_key: str, **kwargs) -> OmnisearchServer:
        """Create Omnisearch server with default configuration."""
        # Omnisearch requires multiple API configurations
        source_apis = kwargs.pop('source_apis', {})
        
        default_config = OmnisearchConfig()
        
        # Extract custom config if provided
        custom_config = kwargs.pop('default_config', None)
        if custom_config:
            if isinstance(custom_config, dict):
                default_config = OmnisearchConfig(**custom_config)
            elif isinstance(custom_config, OmnisearchConfig):
                default_config = custom_config
        
        server_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['logger', 'session_service', 'memory_service', 'artifact_service']}
        
        return OmnisearchServer(
            source_apis=source_apis,
            default_config=default_config,
            **server_kwargs
        )
    
    def _get_perplexity_config(self) -> Dict[str, Any]:
        """Get Perplexity-specific configuration."""
        return {
            "model": "sonar-medium-online",
            "max_tokens": 2000,
            "temperature": 0.3,
        }
    
    def _get_tavily_config(self) -> Dict[str, Any]:
        """Get Tavily-specific configuration."""
        return {
            "search_depth": "advanced",
            "max_results": 10,
            "include_answer": True,
        }
    
    def _get_brave_config(self) -> Dict[str, Any]:
        """Get Brave-specific configuration."""
        return {
            "count": 10,
            "safe_search": "moderate",
            "freshness": "all",
        }
    
    def _get_omnisearch_config(self) -> Dict[str, Any]:
        """Get Omnisearch-specific configuration."""
        return {
            "max_results_per_source": 10,
            "total_max_results": 50,
            "parallel_execution": True,
        }


# Convenience functions for easy server access

def create_mcp_server(server_type: MCPServerType, 
                     api_key: str,
                     config: Optional[ToolRegistry] = None,
                     logger: Optional[RunLogger] = None,
                     **kwargs) -> Optional[MCPServer]:
    """
    Convenience function to create an MCP server quickly.
    
    Args:
        server_type: Type of server to create
        api_key: API key for the service
        config: Tools configuration
        logger: Logger instance
        **kwargs: Additional arguments
        
    Returns:
        Server instance or None
    """
    factory = MCPServerFactory(config=config, logger=logger)
    return factory.create_server(server_type, api_key, **kwargs)


def create_search_suite(suite_type: MCPSuite,
                       api_keys: Dict[str, str],
                       config: Optional[ToolRegistry] = None,
                       logger: Optional[RunLogger] = None,
                       **kwargs) -> Dict[MCPServerType, MCPServer]:
    """
    Convenience function to create an MCP server suite quickly.
    
    Args:
        suite_type: Type of server suite to create
        api_keys: Dictionary of API keys
        config: Tools configuration
        logger: Logger instance
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of servers in the suite
    """
    factory = MCPServerFactory(config=config, logger=logger)
    return factory.create_suite(suite_type, api_keys, **kwargs)


def create_orchestrator(api_keys: Dict[str, str],
                       config: Optional[ToolRegistry] = None,
                       logger: Optional[RunLogger] = None) -> Optional[MCPOrchestrator]:
    """
    Convenience function to create an MCP orchestrator quickly.
    
    Args:
        api_keys: Dictionary of API keys for services
        config: Tools configuration
        logger: Logger instance
        
    Returns:
        Orchestrator instance or None
    """
    factory = MCPServerFactory(config=config, logger=logger)
    
    # Create comprehensive suite first
    servers = factory.create_suite(MCPSuite.COMPREHENSIVE, api_keys)
    
    return factory.create_orchestrator(servers)


def get_research_mcp_servers(api_keys: Dict[str, str],
                           config: Optional[ToolRegistry] = None,
                           logger: Optional[RunLogger] = None) -> Dict[str, MCPServer]:
    """
    Get MCP servers optimized for research tasks.
    
    Returns:
        Dictionary with 'perplexity' and 'tavily' servers
    """
    servers = create_search_suite(MCPSuite.RESEARCH, api_keys, config, logger)
    return {
        "perplexity": servers.get(MCPServerType.PERPLEXITY),
        "tavily": servers.get(MCPServerType.TAVILY),
    }


def get_web_search_mcp_servers(api_keys: Dict[str, str],
                             config: Optional[ToolRegistry] = None,
                             logger: Optional[RunLogger] = None) -> Dict[str, MCPServer]:
    """
    Get MCP servers optimized for web search tasks.
    
    Returns:
        Dictionary with 'brave' and 'tavily' servers
    """
    servers = create_search_suite(MCPSuite.WEB_SEARCH, api_keys, config, logger)
    return {
        "brave": servers.get(MCPServerType.BRAVE),
        "tavily": servers.get(MCPServerType.TAVILY),
    }


def get_ai_powered_mcp_servers(api_keys: Dict[str, str],
                             config: Optional[ToolRegistry] = None,
                             logger: Optional[RunLogger] = None) -> Dict[str, MCPServer]:
    """
    Get MCP servers optimized for AI-powered search tasks.
    
    Returns:
        Dictionary with 'perplexity' and 'omnisearch' servers
    """
    servers = create_search_suite(MCPSuite.AI_POWERED, api_keys, config, logger)
    return {
        "perplexity": servers.get(MCPServerType.PERPLEXITY),
        "omnisearch": servers.get(MCPServerType.OMNISEARCH),
    }


def get_all_mcp_servers(api_keys: Dict[str, str],
                       config: Optional[ToolRegistry] = None,
                       logger: Optional[RunLogger] = None) -> Dict[str, MCPServer]:
    """
    Get all available MCP servers.
    
    Returns:
        Dictionary with all MCP servers
    """
    servers = create_search_suite(MCPSuite.COMPREHENSIVE, api_keys, config, logger)
    return {
        "perplexity": servers.get(MCPServerType.PERPLEXITY),
        "tavily": servers.get(MCPServerType.TAVILY),
        "brave": servers.get(MCPServerType.BRAVE),
        "omnisearch": servers.get(MCPServerType.OMNISEARCH),
    }