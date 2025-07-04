"""
Base classes for ADK built-in tool integration.

Provides common functionality for all built-in tools including context management,
authentication handling, result processing, and comprehensive logging.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type
from enum import Enum

from google.adk.tools.tool_context import ToolContext
from google.adk.agents.invocation_context import InvocationContext

from ..context import (
    ToolContextPattern,
    AuthenticationPattern,
    MemoryAccessPattern,
    PlatformToolContext,
)
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


class ToolExecutionStatus(str, Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolType(str, Enum):
    """Types of ADK built-in tools."""
    GOOGLE_SEARCH = "google_search"
    CODE_EXECUTION = "code_execution"
    VERTEX_SEARCH = "vertex_search"
    BIGQUERY = "bigquery"


@dataclass
class ToolAuthConfig:
    """Configuration for tool authentication."""
    auth_type: str
    credentials: Optional[Dict[str, Any]] = None
    scopes: Optional[List[str]] = None
    service_account: Optional[str] = None
    project_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ADK authentication."""
        return {
            "type": self.auth_type,
            "credentials": self.credentials,
            "scopes": self.scopes or [],
            "service_account": self.service_account,
            "project_id": self.project_id,
        }


@dataclass
class ToolResult:
    """Base result class for all tool executions."""
    tool_type: ToolType
    status: ToolExecutionStatus
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolExecutionStatus.COMPLETED and self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_type": self.tool_type.value,
            "status": self.status.value,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class ToolExecutionResult:
    """Comprehensive result of tool execution with context information."""
    tool_result: ToolResult
    context_info: Dict[str, Any] = field(default_factory=dict)
    authentication_info: Dict[str, Any] = field(default_factory=dict)
    memory_info: Dict[str, Any] = field(default_factory=dict)
    performance_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if overall execution was successful."""
        return self.tool_result.success
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to comprehensive dictionary representation."""
        return {
            "tool_result": self.tool_result.to_dict(),
            "context_info": self.context_info,
            "authentication_info": self.authentication_info,
            "memory_info": self.memory_info,
            "performance_info": self.performance_info,
            "overall_success": self.success,
        }


class BaseTool(ABC):
    """
    Base class for all ADK built-in tool integrations.
    
    Provides common functionality including context management, authentication,
    logging, and integration with platform services.
    """
    
    def __init__(self,
                 tool_type: ToolType,
                 tool_name: str,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 auth_config: Optional[ToolAuthConfig] = None):
        
        self.tool_type = tool_type
        self.tool_name = tool_name
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        self.auth_config = auth_config
        
        # Context patterns for different operations
        self.tool_pattern = ToolContextPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        self.auth_pattern = AuthenticationPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        self.memory_pattern = MemoryAccessPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
    
    @abstractmethod
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool-specific configuration."""
        pass
    
    @abstractmethod
    def execute_tool(self, context: ToolContext, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def execute_with_context(self, 
                           context: ToolContext,
                           memory_queries: Optional[List[str]] = None,
                           required_artifacts: Optional[List[str]] = None,
                           **tool_kwargs) -> ToolExecutionResult:
        """
        Execute tool with comprehensive context management.
        
        Args:
            context: ToolContext for execution
            memory_queries: Optional memory queries to run before execution
            required_artifacts: Optional list of required artifacts
            **tool_kwargs: Tool-specific parameters
            
        Returns:
            Comprehensive execution result with context information
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.info(
                f"Starting {self.tool_name} execution",
                tool_type=self.tool_type.value,
                has_memory_queries=bool(memory_queries),
                required_artifacts=required_artifacts,
            )
        
        result = ToolExecutionResult(
            tool_result=ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.PENDING
            )
        )
        
        try:
            # Setup tool context with comprehensive tracking
            tool_context_result = self.tool_pattern.execute(
                context,
                tool_name=self.tool_name,
                auth_config=self.auth_config.to_dict() if self.auth_config else None,
                memory_query=memory_queries[0] if memory_queries else None,
                required_artifacts=required_artifacts,
            )
            
            result.context_info = {
                "tool_context_success": tool_context_result["success"],
                "available_resources": tool_context_result.get("available_resources", {}),
                "errors": tool_context_result.get("errors", []),
            }
            
            # Handle authentication if needed
            if self.auth_config:
                auth_result = self.auth_pattern.execute(
                    context,
                    auth_configs=[self.auth_config.to_dict()],
                )
                
                result.authentication_info = {
                    "auth_success": auth_result["success"],
                    "authentication_results": auth_result.get("authentication_results", {}),
                    "service_status": auth_result.get("service_status", {}),
                }
                
                if not auth_result["success"]:
                    result.tool_result.status = ToolExecutionStatus.FAILED
                    result.tool_result.error = "Authentication failed"
                    return result
            
            # Run memory queries if requested
            if memory_queries:
                memory_result = self.memory_pattern.execute(
                    context,
                    search_queries=memory_queries,
                    result_limit=5,
                    similarity_threshold=0.7,
                )
                
                result.memory_info = {
                    "memory_success": memory_result["success"],
                    "search_results": memory_result.get("search_results", {}),
                    "total_results": memory_result.get("total_results", 0),
                }
            
            # Execute the actual tool
            result.tool_result.status = ToolExecutionStatus.RUNNING
            tool_result = self.execute_tool(context, **tool_kwargs)
            
            # Update with tool execution results
            result.tool_result = tool_result
            
            if self.logger:
                self.logger.info(
                    f"Completed {self.tool_name} execution",
                    tool_type=self.tool_type.value,
                    success=tool_result.success,
                    execution_time_ms=tool_result.execution_time_ms,
                )
            
        except Exception as e:
            result.tool_result.status = ToolExecutionStatus.FAILED
            result.tool_result.error = str(e)
            
            if self.logger:
                self.logger.error(
                    f"Failed {self.tool_name} execution: {e}",
                    tool_type=self.tool_type.value,
                    error=str(e),
                )
        
        finally:
            # Record performance information
            total_time = time.time() - start_time
            result.performance_info = {
                "total_execution_time_ms": total_time * 1000,
                "tool_execution_time_ms": result.tool_result.execution_time_ms,
                "context_overhead_ms": (total_time * 1000) - result.tool_result.execution_time_ms,
            }
        
        return result
    
    def validate_context(self, context: ToolContext) -> bool:
        """Validate that context has required capabilities."""
        if not isinstance(context, ToolContext):
            if self.logger:
                self.logger.error("Invalid context type for tool execution")
            return False
        
        # Check for required attributes
        required_attrs = ['function_call_id', 'invocation_id']
        for attr in required_attrs:
            if not hasattr(context, attr):
                if self.logger:
                    self.logger.warning(f"Context missing required attribute: {attr}")
        
        return True
    
    def get_enhanced_context(self, context: ToolContext) -> PlatformToolContext:
        """Get platform-enhanced context with full capabilities."""
        return PlatformToolContext(
            context,
            tool_name=self.tool_name,
            logger=self.logger,
            session_service=self.session_service,
            memory_service=self.memory_service,
            artifact_service=self.artifact_service,
        )
    
    def log_tool_usage(self, 
                      operation: str, 
                      parameters: Dict[str, Any],
                      result: Optional[ToolResult] = None) -> None:
        """Log tool usage for monitoring and debugging."""
        if not self.logger:
            return
        
        log_data = {
            "tool_name": self.tool_name,
            "tool_type": self.tool_type.value,
            "operation": operation,
            "parameter_keys": list(parameters.keys()),
            "parameter_count": len(parameters),
        }
        
        if result:
            log_data.update({
                "result_status": result.status.value,
                "result_success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "has_error": result.error is not None,
            })
        
        self.logger.info(f"Tool usage: {self.tool_name}.{operation}", **log_data)


class BuiltInToolMixin:
    """
    Mixin for ADK built-in tool specific functionality.
    
    Provides helper methods for working with ADK's built-in tools.
    """
    
    def get_builtin_tool_instance(self, context: ToolContext, tool_name: str) -> Any:
        """Get ADK built-in tool instance from context."""
        try:
            # ADK built-in tools are typically accessed through the context
            # This is a placeholder for the actual ADK API
            if hasattr(context, 'get_builtin_tool'):
                return context.get_builtin_tool(tool_name)
            elif hasattr(context, 'tools'):
                return getattr(context.tools, tool_name, None)
            else:
                return None
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error getting built-in tool {tool_name}: {e}")
            return None
    
    def handle_builtin_result(self, builtin_result: Any, tool_type: ToolType) -> ToolResult:
        """Convert ADK built-in tool result to platform result format."""
        try:
            # This would need to be customized based on actual ADK result formats
            if builtin_result is None:
                return ToolResult(
                    tool_type=tool_type,
                    status=ToolExecutionStatus.FAILED,
                    error="No result from built-in tool",
                )
            
            # Handle successful result
            return ToolResult(
                tool_type=tool_type,
                status=ToolExecutionStatus.COMPLETED,
                data=builtin_result,
                metadata={"builtin_result_type": type(builtin_result).__name__},
            )
            
        except Exception as e:
            return ToolResult(
                tool_type=tool_type,
                status=ToolExecutionStatus.FAILED,
                error=f"Error processing built-in result: {e}",
            )