"""
Tool Factory for ADK Built-in Tools

Provides factory methods and utilities for creating and managing ADK built-in tools
with proper configuration, authentication, and service integration.
"""

from typing import Any, Dict, List, Optional, Type, Union
from enum import Enum

from .base import BaseTool, ToolType, ToolAuthConfig
from .google_search import GoogleSearchTool, SearchQuery
from .code_execution import CodeExecutionTool, CodeExecutionConfig, ExecutionSafety, CodeLanguage
from .vertex_search import VertexSearchTool, VertexSearchConfig, SearchType as VertexSearchType
from .bigquery import BigQueryTool, BigQueryConfig
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..config.tools import ToolsConfig


class ToolSuite(str, Enum):
    """Predefined tool suites for different use cases."""
    RESEARCH = "research"           # Google Search + Vertex AI Search
    DEVELOPMENT = "development"     # Code Execution + BigQuery
    ANALYTICS = "analytics"         # BigQuery + Vertex AI Search  
    COMPREHENSIVE = "comprehensive" # All tools enabled


class ToolFactory:
    """
    Factory for creating and managing ADK built-in tools.
    
    Provides centralized configuration, authentication, and service integration
    for all built-in tools with convenient creation patterns.
    """
    
    def __init__(self,
                 config: Optional[ToolsConfig] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        self.config = config or ToolsConfig()
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Tool instances cache
        self._tool_instances: Dict[ToolType, BaseTool] = {}
        
        # Default configurations
        self._default_auth_configs = {
            ToolType.GOOGLE_SEARCH: ToolAuthConfig(
                auth_type="google_search_api",
                project_id=getattr(config, 'google_cloud_project', None) if config else None,
            ),
            ToolType.CODE_EXECUTION: None,  # Code execution typically doesn't need auth
            ToolType.VERTEX_SEARCH: ToolAuthConfig(
                auth_type="vertex_ai",
                project_id=getattr(config, 'google_cloud_project', None) if config else None,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            ),
            ToolType.BIGQUERY: ToolAuthConfig(
                auth_type="bigquery",
                project_id=getattr(config, 'google_cloud_project', None) if config else None,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            ),
        }
    
    def create_tool(self, 
                   tool_type: ToolType,
                   custom_config: Optional[Dict[str, Any]] = None,
                   auth_config: Optional[ToolAuthConfig] = None) -> BaseTool:
        """
        Create a specific ADK built-in tool with configuration.
        
        Args:
            tool_type: Type of tool to create
            custom_config: Custom configuration for the tool
            auth_config: Custom authentication configuration
            
        Returns:
            Configured tool instance
        """
        # Use provided auth config or default
        if auth_config is None:
            auth_config = self._default_auth_configs.get(tool_type)
        
        # Common tool arguments
        tool_args = {
            "logger": self.logger,
            "session_service": self.session_service,
            "memory_service": self.memory_service,
            "artifact_service": self.artifact_service,
            "auth_config": auth_config,
        }
        
        # Add custom configuration
        if custom_config:
            tool_args.update(custom_config)
        
        # Create tool based on type
        if tool_type == ToolType.GOOGLE_SEARCH:
            return self._create_google_search_tool(**tool_args)
        elif tool_type == ToolType.CODE_EXECUTION:
            return self._create_code_execution_tool(**tool_args)
        elif tool_type == ToolType.VERTEX_SEARCH:
            return self._create_vertex_search_tool(**tool_args)
        elif tool_type == ToolType.BIGQUERY:
            return self._create_bigquery_tool(**tool_args)
        else:
            raise ValueError(f"Unsupported tool type: {tool_type}")
    
    def get_tool(self, 
                tool_type: ToolType,
                create_if_missing: bool = True,
                **kwargs) -> Optional[BaseTool]:
        """
        Get a tool instance, creating it if necessary.
        
        Args:
            tool_type: Type of tool to get
            create_if_missing: Whether to create the tool if not cached
            **kwargs: Additional arguments for tool creation
            
        Returns:
            Tool instance or None if not available
        """
        # Check cache first
        if tool_type in self._tool_instances:
            return self._tool_instances[tool_type]
        
        # Create if requested
        if create_if_missing:
            try:
                tool = self.create_tool(tool_type, **kwargs)
                self._tool_instances[tool_type] = tool
                return tool
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to create tool {tool_type}: {e}")
                return None
        
        return None
    
    def create_tool_suite(self, 
                         suite_type: ToolSuite,
                         custom_configs: Optional[Dict[ToolType, Dict[str, Any]]] = None) -> Dict[ToolType, BaseTool]:
        """
        Create a predefined suite of tools for specific use cases.
        
        Args:
            suite_type: Type of tool suite to create
            custom_configs: Custom configurations for specific tools
            
        Returns:
            Dictionary of tool type to tool instance
        """
        suite_tools = {}
        custom_configs = custom_configs or {}
        
        # Define tool suites
        suite_definitions = {
            ToolSuite.RESEARCH: [
                ToolType.GOOGLE_SEARCH,
                ToolType.VERTEX_SEARCH,
            ],
            ToolSuite.DEVELOPMENT: [
                ToolType.CODE_EXECUTION,
                ToolType.BIGQUERY,
            ],
            ToolSuite.ANALYTICS: [
                ToolType.BIGQUERY,
                ToolType.VERTEX_SEARCH,
            ],
            ToolSuite.COMPREHENSIVE: [
                ToolType.GOOGLE_SEARCH,
                ToolType.CODE_EXECUTION,
                ToolType.VERTEX_SEARCH,
                ToolType.BIGQUERY,
            ],
        }
        
        tool_types = suite_definitions.get(suite_type, [])
        
        for tool_type in tool_types:
            try:
                custom_config = custom_configs.get(tool_type, {})
                tool = self.create_tool(tool_type, custom_config)
                suite_tools[tool_type] = tool
                
                if self.logger:
                    self.logger.info(f"Created {tool_type.value} for {suite_type.value} suite")
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to create {tool_type.value} for suite: {e}")
        
        return suite_tools
    
    def get_tool_capabilities(self) -> Dict[ToolType, Dict[str, Any]]:
        """
        Get capabilities information for all available tools.
        
        Returns:
            Dictionary mapping tool types to their capabilities
        """
        capabilities = {}
        
        for tool_type in ToolType:
            try:
                # Create temporary tool instance to get config
                temp_tool = self.create_tool(tool_type)
                capabilities[tool_type] = temp_tool.get_tool_config()
            except Exception as e:
                capabilities[tool_type] = {
                    "available": False,
                    "error": str(e),
                }
                if self.logger:
                    self.logger.warning(f"Could not get capabilities for {tool_type}: {e}")
        
        return capabilities
    
    def validate_tool_requirements(self) -> Dict[ToolType, Dict[str, Any]]:
        """
        Validate requirements for all tools (authentication, configuration, etc.).
        
        Returns:
            Dictionary with validation results for each tool type
        """
        validation_results = {}
        
        for tool_type in ToolType:
            result = {
                "valid": False,
                "issues": [],
                "auth_required": True,
                "config_valid": False,
            }
            
            try:
                # Check authentication requirements
                auth_config = self._default_auth_configs.get(tool_type)
                if auth_config:
                    if not auth_config.project_id:
                        result["issues"].append("Missing project_id in authentication config")
                else:
                    result["auth_required"] = False
                
                # Check tool-specific configuration
                if tool_type == ToolType.GOOGLE_SEARCH:
                    # Google Search may need API key
                    if self.config and hasattr(self.config, 'google_search_api_key'):
                        result["config_valid"] = True
                    else:
                        result["issues"].append("Google Search API key not configured")
                
                elif tool_type == ToolType.VERTEX_SEARCH:
                    # Vertex AI needs project and location
                    if auth_config and auth_config.project_id:
                        result["config_valid"] = True
                    else:
                        result["issues"].append("Vertex AI project configuration missing")
                
                elif tool_type == ToolType.BIGQUERY:
                    # BigQuery needs project
                    if auth_config and auth_config.project_id:
                        result["config_valid"] = True
                    else:
                        result["issues"].append("BigQuery project configuration missing")
                
                elif tool_type == ToolType.CODE_EXECUTION:
                    # Code execution usually works without special config
                    result["config_valid"] = True
                
                # Overall validation
                result["valid"] = len(result["issues"]) == 0
                
            except Exception as e:
                result["issues"].append(f"Validation error: {str(e)}")
            
            validation_results[tool_type] = result
        
        return validation_results
    
    def _create_google_search_tool(self, **kwargs) -> GoogleSearchTool:
        """Create Google Search tool with default configuration."""
        # Remove tool-specific args that aren't for GoogleSearchTool
        tool_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['logger', 'session_service', 'memory_service', 'artifact_service', 'auth_config']}
        
        return GoogleSearchTool(**tool_kwargs)
    
    def _create_code_execution_tool(self, **kwargs) -> CodeExecutionTool:
        """Create Code Execution tool with default configuration."""
        # Set up default code execution configuration
        default_config = CodeExecutionConfig(
            language=CodeLanguage.PYTHON,
            safety_level=ExecutionSafety.MODERATE,
            timeout_seconds=30,
            max_output_length=10000,
        )
        
        # Extract custom config if provided
        custom_config = kwargs.pop('default_config', None)
        if custom_config:
            if isinstance(custom_config, dict):
                # Convert dict to CodeExecutionConfig
                default_config = CodeExecutionConfig(**custom_config)
            elif isinstance(custom_config, CodeExecutionConfig):
                default_config = custom_config
        
        tool_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['logger', 'session_service', 'memory_service', 'artifact_service', 'auth_config']}
        
        return CodeExecutionTool(default_config=default_config, **tool_kwargs)
    
    def _create_vertex_search_tool(self, **kwargs) -> VertexSearchTool:
        """Create Vertex AI Search tool with default configuration."""
        # Set up default configuration
        project_id = None
        location = "global"
        
        if self.config:
            project_id = getattr(self.config, 'google_cloud_project', None)
            location = getattr(self.config, 'google_cloud_location', 'global')
        
        # Override with provided values
        project_id = kwargs.pop('project_id', project_id)
        location = kwargs.pop('location', location)
        
        default_config = VertexSearchConfig(
            search_type=VertexSearchType.SEMANTIC,
            max_results=10,
            similarity_threshold=0.7,
        )
        
        # Extract custom config if provided
        custom_config = kwargs.pop('default_config', None)
        if custom_config:
            if isinstance(custom_config, dict):
                default_config = VertexSearchConfig(**custom_config)
            elif isinstance(custom_config, VertexSearchConfig):
                default_config = custom_config
        
        tool_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['logger', 'session_service', 'memory_service', 'artifact_service', 'auth_config']}
        
        return VertexSearchTool(
            default_config=default_config,
            project_id=project_id,
            location=location,
            **tool_kwargs
        )
    
    def _create_bigquery_tool(self, **kwargs) -> BigQueryTool:
        """Create BigQuery tool with default configuration."""
        # Set up default configuration
        project_id = None
        location = "US"
        
        if self.config:
            project_id = getattr(self.config, 'google_cloud_project', None)
            location = getattr(self.config, 'bigquery_location', 'US')
        
        # Override with provided values
        project_id = kwargs.pop('project_id', project_id)
        location = kwargs.pop('location', location)
        
        default_config = BigQueryConfig(
            project_id=project_id or "default-project",
            location=location,
            max_results=1000,
            timeout_seconds=300,
        )
        
        # Extract custom config if provided
        custom_config = kwargs.pop('default_config', None)
        if custom_config:
            if isinstance(custom_config, dict):
                default_config = BigQueryConfig(**custom_config)
            elif isinstance(custom_config, BigQueryConfig):
                default_config = custom_config
        
        tool_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['logger', 'session_service', 'memory_service', 'artifact_service', 'auth_config']}
        
        return BigQueryTool(
            default_config=default_config,
            project_id=project_id,
            location=location,
            **tool_kwargs
        )


# Convenience functions for easy tool access

def get_tool(tool_type: ToolType, 
            config: Optional[ToolsConfig] = None,
            logger: Optional[RunLogger] = None,
            **kwargs) -> Optional[BaseTool]:
    """
    Convenience function to get a tool instance quickly.
    
    Args:
        tool_type: Type of tool to get
        config: Tools configuration
        logger: Logger instance
        **kwargs: Additional arguments
        
    Returns:
        Tool instance or None
    """
    factory = ToolFactory(config=config, logger=logger)
    return factory.get_tool(tool_type, **kwargs)


def create_tool_suite(suite_type: ToolSuite,
                     config: Optional[ToolsConfig] = None,
                     logger: Optional[RunLogger] = None,
                     **kwargs) -> Dict[ToolType, BaseTool]:
    """
    Convenience function to create a tool suite quickly.
    
    Args:
        suite_type: Type of tool suite to create
        config: Tools configuration
        logger: Logger instance
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of tools in the suite
    """
    factory = ToolFactory(config=config, logger=logger)
    return factory.create_tool_suite(suite_type, **kwargs)


def get_research_tools(config: Optional[ToolsConfig] = None,
                      logger: Optional[RunLogger] = None) -> Dict[str, BaseTool]:
    """
    Get tools optimized for research tasks.
    
    Returns:
        Dictionary with 'search' and 'vertex_search' tools
    """
    tools = create_tool_suite(ToolSuite.RESEARCH, config, logger)
    return {
        "search": tools.get(ToolType.GOOGLE_SEARCH),
        "vertex_search": tools.get(ToolType.VERTEX_SEARCH),
    }


def get_development_tools(config: Optional[ToolsConfig] = None,
                         logger: Optional[RunLogger] = None) -> Dict[str, BaseTool]:
    """
    Get tools optimized for development tasks.
    
    Returns:
        Dictionary with 'code_execution' and 'bigquery' tools
    """
    tools = create_tool_suite(ToolSuite.DEVELOPMENT, config, logger)
    return {
        "code_execution": tools.get(ToolType.CODE_EXECUTION),
        "bigquery": tools.get(ToolType.BIGQUERY),
    }


def get_analytics_tools(config: Optional[ToolsConfig] = None,
                       logger: Optional[RunLogger] = None) -> Dict[str, BaseTool]:
    """
    Get tools optimized for analytics tasks.
    
    Returns:
        Dictionary with 'bigquery' and 'vertex_search' tools
    """
    tools = create_tool_suite(ToolSuite.ANALYTICS, config, logger)
    return {
        "bigquery": tools.get(ToolType.BIGQUERY),
        "vertex_search": tools.get(ToolType.VERTEX_SEARCH),
    }


def get_all_tools(config: Optional[ToolsConfig] = None,
                 logger: Optional[RunLogger] = None) -> Dict[str, BaseTool]:
    """
    Get all available ADK built-in tools.
    
    Returns:
        Dictionary with all tools
    """
    tools = create_tool_suite(ToolSuite.COMPREHENSIVE, config, logger)
    return {
        "search": tools.get(ToolType.GOOGLE_SEARCH),
        "code_execution": tools.get(ToolType.CODE_EXECUTION),
        "vertex_search": tools.get(ToolType.VERTEX_SEARCH),
        "bigquery": tools.get(ToolType.BIGQUERY),
    }