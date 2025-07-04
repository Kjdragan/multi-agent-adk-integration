"""
Context usage patterns and best practices for different scenarios.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext

from .helpers import (
    managed_invocation_context,
    managed_callback_context,
    managed_tool_context,
    context_with_services
)
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


class ContextPatternType(str, Enum):
    """Types of context patterns."""
    AGENT_EXECUTION = "agent_execution"
    TOOL_EXECUTION = "tool_execution"
    CALLBACK_LIFECYCLE = "callback_lifecycle"
    STATE_MANAGEMENT = "state_management"
    ARTIFACT_MANAGEMENT = "artifact_management"
    MEMORY_ACCESS = "memory_access"
    AUTHENTICATION = "authentication"


class ContextPattern(ABC):
    """
    Abstract base class for context usage patterns.
    
    Provides reusable patterns for common context operations.
    """
    
    def __init__(self, 
                 pattern_type: ContextPatternType,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        self.pattern_type = pattern_type
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
    
    @abstractmethod
    def execute(self, context: Any, **kwargs) -> Any:
        """Execute the pattern with the given context."""
        pass
    
    def log_pattern_start(self, context: Any, **kwargs) -> None:
        """Log the start of pattern execution."""
        if self.logger:
            self.logger.debug(
                f"Starting context pattern: {self.pattern_type}",
                pattern_type=self.pattern_type.value,
                context_type=type(context).__name__,
                **kwargs
            )
    
    def log_pattern_complete(self, context: Any, result: Any, **kwargs) -> None:
        """Log the completion of pattern execution."""
        if self.logger:
            self.logger.debug(
                f"Completed context pattern: {self.pattern_type}",
                pattern_type=self.pattern_type.value,
                context_type=type(context).__name__,
                result_type=type(result).__name__,
                **kwargs
            )


class AgentContextPattern(ContextPattern):
    """
    Pattern for agent execution using InvocationContext.
    
    Handles the complete lifecycle of agent execution including health checks,
    cross-agent coordination, and error handling.
    """
    
    def __init__(self, **kwargs):
        super().__init__(ContextPatternType.AGENT_EXECUTION, **kwargs)
    
    def execute(self, context: InvocationContext, 
                shared_state_keys: Optional[List[str]] = None,
                health_check: bool = True,
                auto_terminate_on_error: bool = True) -> Dict[str, Any]:
        """
        Execute agent context pattern.
        
        Args:
            context: InvocationContext for agent execution
            shared_state_keys: Keys for cross-agent state sharing
            health_check: Whether to perform session health check
            auto_terminate_on_error: Whether to auto-terminate on errors
            
        Returns:
            Dictionary with execution results and metadata
        """
        self.log_pattern_start(context, 
                             shared_keys=shared_state_keys,
                             health_check=health_check)
        
        result = {
            "success": False,
            "health_status": None,
            "shared_state": {},
            "errors": []
        }
        
        try:
            with managed_invocation_context(
                context,
                logger=self.logger,
                session_service=self.session_service,
                memory_service=self.memory_service,
                artifact_service=self.artifact_service,
                auto_health_check=health_check
            ) as managed_ctx:
                
                # Record health status
                result["health_status"] = managed_ctx.check_session_health()
                
                # Get shared state if requested
                if shared_state_keys:
                    result["shared_state"] = managed_ctx.get_shared_state(shared_state_keys)
                
                # Mark as successful
                result["success"] = True
                
                # Return the managed context for further use
                result["managed_context"] = managed_ctx
                
        except Exception as e:
            result["errors"].append(str(e))
            if auto_terminate_on_error and self.logger:
                self.logger.error(f"Agent execution error: {e}")
        
        self.log_pattern_complete(context, result)
        return result


class ToolContextPattern(ContextPattern):
    """
    Pattern for tool execution using ToolContext.
    
    Handles authentication, memory access, artifact management, and execution tracking.
    """
    
    def __init__(self, **kwargs):
        super().__init__(ContextPatternType.TOOL_EXECUTION, **kwargs)
    
    def execute(self, context: ToolContext,
                tool_name: str,
                auth_config: Optional[Any] = None,
                memory_query: Optional[str] = None,
                required_artifacts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute tool context pattern.
        
        Args:
            context: ToolContext for tool execution
            tool_name: Name of the tool being executed
            auth_config: Authentication configuration if needed
            memory_query: Query for memory search
            required_artifacts: List of required artifact names
            
        Returns:
            Dictionary with execution results and resources
        """
        self.log_pattern_start(context,
                             tool_name=tool_name,
                             needs_auth=auth_config is not None,
                             memory_query=memory_query,
                             required_artifacts=required_artifacts)
        
        result = {
            "success": False,
            "authentication": {"required": False, "success": False},
            "memory_results": None,
            "artifacts": [],
            "available_resources": {},
            "errors": []
        }
        
        try:
            with managed_tool_context(
                context,
                tool_name=tool_name,
                logger=self.logger
            ) as managed_ctx:
                
                # Handle authentication if needed
                if auth_config:
                    result["authentication"]["required"] = True
                    auth_success = managed_ctx.handle_authentication(auth_config)
                    result["authentication"]["success"] = auth_success
                    
                    if not auth_success:
                        result["errors"].append("Authentication failed")
                        return result
                
                # Search memory if requested
                if memory_query:
                    memory_results = managed_ctx.search_memory(memory_query)
                    result["memory_results"] = memory_results
                
                # List available artifacts
                artifacts = managed_ctx.list_artifacts()
                result["artifacts"] = artifacts
                
                # Check for required artifacts
                if required_artifacts:
                    missing_artifacts = [
                        artifact for artifact in required_artifacts
                        if artifact not in artifacts
                    ]
                    if missing_artifacts:
                        result["errors"].append(f"Missing artifacts: {missing_artifacts}")
                
                # Prepare available resources
                result["available_resources"] = {
                    "state_access": hasattr(managed_ctx, 'safe_state_access'),
                    "memory_search": memory_query is not None,
                    "artifacts": len(artifacts),
                    "authentication": result["authentication"]["success"]
                }
                
                result["success"] = len(result["errors"]) == 0
                result["managed_context"] = managed_ctx
                
        except Exception as e:
            result["errors"].append(str(e))
            if self.logger:
                self.logger.error(f"Tool execution pattern error: {e}")
        
        self.log_pattern_complete(context, result, tool_name=tool_name)
        return result


class CallbackContextPattern(ContextPattern):
    """
    Pattern for callback lifecycle management using CallbackContext.
    
    Handles state tracking, artifact monitoring, and change detection.
    """
    
    def __init__(self, **kwargs):
        super().__init__(ContextPatternType.CALLBACK_LIFECYCLE, **kwargs)
    
    def execute(self, context: CallbackContext,
                callback_type: str,
                state_operations: Optional[List[Dict[str, Any]]] = None,
                artifact_operations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute callback context pattern.
        
        Args:
            context: CallbackContext for callback execution
            callback_type: Type of callback (e.g., "before_agent", "after_model")
            state_operations: List of state operations to perform
            artifact_operations: List of artifact operations to perform
            
        Returns:
            Dictionary with change tracking and operation results
        """
        self.log_pattern_start(context,
                             callback_type=callback_type,
                             state_ops=len(state_operations or []),
                             artifact_ops=len(artifact_operations or []))
        
        result = {
            "success": False,
            "changes": {},
            "operations": {"state": [], "artifacts": []},
            "errors": []
        }
        
        try:
            with managed_callback_context(
                context,
                callback_type=callback_type,
                logger=self.logger
            ) as managed_ctx:
                
                # Perform state operations
                if state_operations:
                    for op in state_operations:
                        try:
                            if op.get("action") == "update":
                                success = managed_ctx.safe_state_update(
                                    op["key"], op["value"]
                                )
                                result["operations"]["state"].append({
                                    "action": "update",
                                    "key": op["key"],
                                    "success": success
                                })
                        except Exception as e:
                            result["errors"].append(f"State operation error: {e}")
                
                # Perform artifact operations
                if artifact_operations:
                    for op in artifact_operations:
                        try:
                            if op.get("action") == "save":
                                success = managed_ctx.safe_artifact_save(
                                    op["filename"], op["content"]
                                )
                                result["operations"]["artifacts"].append({
                                    "action": "save",
                                    "filename": op["filename"],
                                    "success": success
                                })
                        except Exception as e:
                            result["errors"].append(f"Artifact operation error: {e}")
                
                # Get final change summary
                result["changes"] = managed_ctx.finalize_tracking()
                result["success"] = len(result["errors"]) == 0
                result["managed_context"] = managed_ctx
                
        except Exception as e:
            result["errors"].append(str(e))
            if self.logger:
                self.logger.error(f"Callback pattern error: {e}")
        
        self.log_pattern_complete(context, result, callback_type=callback_type)
        return result


class StateManagementPattern(ContextPattern):
    """
    Pattern for sophisticated state management across context types.
    """
    
    def __init__(self, **kwargs):
        super().__init__(ContextPatternType.STATE_MANAGEMENT, **kwargs)
    
    def execute(self, context: Any,
                state_schema: Dict[str, Any],
                validation_rules: Optional[Dict[str, Callable]] = None,
                auto_prefix: bool = True) -> Dict[str, Any]:
        """
        Execute state management pattern.
        
        Args:
            context: Any context with state access
            state_schema: Schema defining expected state structure
            validation_rules: Custom validation functions for state values
            auto_prefix: Whether to automatically apply state prefixes
            
        Returns:
            Dictionary with state management results
        """
        self.log_pattern_start(context,
                             schema_keys=list(state_schema.keys()),
                             has_validation=validation_rules is not None)
        
        result = {
            "success": False,
            "state_status": {},
            "validation_results": {},
            "applied_prefixes": {},
            "errors": []
        }
        
        try:
            # Enhance context with services
            enhanced_ctx = context_with_services(
                context,
                logger=self.logger,
                session_service=self.session_service,
                memory_service=self.memory_service,
                artifact_service=self.artifact_service
            )
            
            # Process each state key in schema
            for key, config in state_schema.items():
                try:
                    # Apply prefix if needed
                    actual_key = key
                    if auto_prefix and "prefix" in config:
                        actual_key = f"{config['prefix']}:{key}"
                        result["applied_prefixes"][key] = actual_key
                    
                    # Check if key exists
                    if hasattr(enhanced_ctx, 'safe_state_access'):
                        current_value = enhanced_ctx.safe_state_access(actual_key)
                    elif hasattr(enhanced_ctx, 'state'):
                        current_value = getattr(enhanced_ctx.state, actual_key, None)
                    else:
                        current_value = None
                    
                    # Apply default if needed
                    if current_value is None and "default" in config:
                        if hasattr(enhanced_ctx, 'safe_state_update'):
                            enhanced_ctx.safe_state_update(actual_key, config["default"])
                        elif hasattr(enhanced_ctx, 'state'):
                            enhanced_ctx.state[actual_key] = config["default"]
                        current_value = config["default"]
                    
                    result["state_status"][key] = {
                        "exists": current_value is not None,
                        "value": current_value,
                        "type": type(current_value).__name__
                    }
                    
                    # Validate if rules provided
                    if validation_rules and key in validation_rules:
                        try:
                            is_valid = validation_rules[key](current_value)
                            result["validation_results"][key] = {
                                "valid": is_valid,
                                "value": current_value
                            }
                        except Exception as e:
                            result["validation_results"][key] = {
                                "valid": False,
                                "error": str(e)
                            }
                    
                except Exception as e:
                    result["errors"].append(f"Error processing state key {key}: {e}")
            
            result["success"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(str(e))
            if self.logger:
                self.logger.error(f"State management pattern error: {e}")
        
        self.log_pattern_complete(context, result)
        return result


class ArtifactManagementPattern(ContextPattern):
    """
    Pattern for comprehensive artifact management.
    """
    
    def __init__(self, **kwargs):
        super().__init__(ContextPatternType.ARTIFACT_MANAGEMENT, **kwargs)
    
    def execute(self, context: Any,
                required_artifacts: Optional[List[str]] = None,
                artifact_operations: Optional[List[Dict[str, Any]]] = None,
                cleanup_old: bool = False) -> Dict[str, Any]:
        """
        Execute artifact management pattern.
        
        Args:
            context: Any context with artifact access
            required_artifacts: List of required artifact names
            artifact_operations: List of artifact operations to perform
            cleanup_old: Whether to clean up old artifacts
            
        Returns:
            Dictionary with artifact management results
        """
        self.log_pattern_start(context,
                             required_count=len(required_artifacts or []),
                             operations_count=len(artifact_operations or []))
        
        result = {
            "success": False,
            "available_artifacts": [],
            "required_status": {},
            "operation_results": [],
            "cleanup_results": {},
            "errors": []
        }
        
        try:
            # List available artifacts
            if hasattr(context, 'list_artifacts'):
                result["available_artifacts"] = context.list_artifacts() or []
            elif hasattr(context, 'enhanced_execution'):
                exec_info = context.enhanced_execution
                result["available_artifacts"] = exec_info.get("artifacts", [])
            
            # Check required artifacts
            if required_artifacts:
                for artifact in required_artifacts:
                    result["required_status"][artifact] = {
                        "available": artifact in result["available_artifacts"]
                    }
            
            # Perform artifact operations
            if artifact_operations:
                for op in artifact_operations:
                    op_result = {"operation": op.get("action", "unknown"), "success": False}
                    
                    try:
                        if op.get("action") == "load":
                            if hasattr(context, 'load_artifact'):
                                artifact_content = context.load_artifact(op["filename"])
                                op_result["success"] = artifact_content is not None
                                op_result["content_available"] = artifact_content is not None
                        
                        elif op.get("action") == "save":
                            if hasattr(context, 'save_artifact'):
                                version = context.save_artifact(op["filename"], op["content"])
                                op_result["success"] = version is not None
                                op_result["version"] = version
                        
                        elif op.get("action") == "delete":
                            # Note: ADK contexts typically don't have delete methods
                            # This would need to be handled at the service level
                            op_result["success"] = False
                            op_result["error"] = "Delete not supported in context"
                        
                    except Exception as e:
                        op_result["error"] = str(e)
                    
                    result["operation_results"].append(op_result)
            
            # Cleanup if requested
            if cleanup_old:
                # This would typically be handled at the service level
                result["cleanup_results"]["attempted"] = True
                result["cleanup_results"]["success"] = False
                result["cleanup_results"]["note"] = "Cleanup requires service-level access"
            
            result["success"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(str(e))
            if self.logger:
                self.logger.error(f"Artifact management pattern error: {e}")
        
        self.log_pattern_complete(context, result)
        return result


class MemoryAccessPattern(ContextPattern):
    """
    Pattern for memory access and management.
    """
    
    def __init__(self, **kwargs):
        super().__init__(ContextPatternType.MEMORY_ACCESS, **kwargs)
    
    def execute(self, context: Any,
                search_queries: List[str],
                result_limit: int = 5,
                similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Execute memory access pattern.
        
        Args:
            context: Any context with memory search capability
            search_queries: List of search queries to execute
            result_limit: Maximum results per query
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with memory search results
        """
        self.log_pattern_start(context,
                             query_count=len(search_queries),
                             result_limit=result_limit)
        
        result = {
            "success": False,
            "search_results": {},
            "total_results": 0,
            "query_performance": {},
            "errors": []
        }
        
        try:
            for query in search_queries:
                query_result = {
                    "query": query,
                    "results": [],
                    "result_count": 0,
                    "execution_time": 0
                }
                
                try:
                    import time
                    start_time = time.time()
                    
                    # Attempt memory search
                    if hasattr(context, 'search_memory'):
                        memory_results = context.search_memory(query)
                    elif hasattr(context, 'enhanced_execution'):
                        # Try enhanced context
                        exec_info = context.enhanced_execution
                        memory_search = exec_info.get("memory_search", {})
                        memory_results = memory_search.get("results") if memory_search else None
                    else:
                        memory_results = None
                    
                    execution_time = time.time() - start_time
                    query_result["execution_time"] = execution_time
                    
                    if memory_results and hasattr(memory_results, 'results'):
                        # Filter by similarity threshold and limit
                        filtered_results = [
                            result for result in memory_results.results
                            if getattr(result, 'score', 1.0) >= similarity_threshold
                        ][:result_limit]
                        
                        query_result["results"] = [
                            {
                                "text": getattr(result, 'text', ''),
                                "score": getattr(result, 'score', 0.0),
                                "metadata": getattr(result, 'metadata', {})
                            }
                            for result in filtered_results
                        ]
                        query_result["result_count"] = len(query_result["results"])
                    
                except Exception as e:
                    query_result["error"] = str(e)
                    result["errors"].append(f"Query '{query}' failed: {e}")
                
                result["search_results"][query] = query_result
                result["total_results"] += query_result["result_count"]
                result["query_performance"][query] = query_result["execution_time"]
            
            result["success"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(str(e))
            if self.logger:
                self.logger.error(f"Memory access pattern error: {e}")
        
        self.log_pattern_complete(context, result)
        return result


class AuthenticationPattern(ContextPattern):
    """
    Pattern for handling authentication flows.
    """
    
    def __init__(self, **kwargs):
        super().__init__(ContextPatternType.AUTHENTICATION, **kwargs)
    
    def execute(self, context: ToolContext,
                auth_configs: List[Dict[str, Any]],
                required_services: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute authentication pattern.
        
        Args:
            context: ToolContext with authentication capabilities
            auth_configs: List of authentication configurations
            required_services: List of required service names
            
        Returns:
            Dictionary with authentication results
        """
        self.log_pattern_start(context,
                             auth_config_count=len(auth_configs),
                             required_services=required_services)
        
        result = {
            "success": False,
            "authentication_results": {},
            "service_status": {},
            "errors": []
        }
        
        try:
            # Process each authentication configuration
            for i, auth_config in enumerate(auth_configs):
                auth_name = auth_config.get("name", f"auth_{i}")
                auth_result = {
                    "name": auth_name,
                    "success": False,
                    "method": auth_config.get("method", "unknown")
                }
                
                try:
                    # Handle authentication based on method
                    if hasattr(context, 'handle_authentication'):
                        # Use enhanced context method
                        success = context.handle_authentication(auth_config)
                        auth_result["success"] = success
                    elif hasattr(context, 'request_credential'):
                        # Use direct ADK method
                        context.request_credential(auth_config)
                        auth_result["success"] = True
                        auth_result["status"] = "requested"
                    
                except Exception as e:
                    auth_result["error"] = str(e)
                    result["errors"].append(f"Authentication {auth_name} failed: {e}")
                
                result["authentication_results"][auth_name] = auth_result
            
            # Check service availability if specified
            if required_services:
                for service in required_services:
                    service_available = False
                    
                    # Check if service is accessible (simplified check)
                    if hasattr(context, 'state'):
                        service_key = f"service:{service}:available"
                        service_available = getattr(context.state, service_key, False)
                    
                    result["service_status"][service] = {
                        "available": service_available,
                        "authenticated": any(
                            auth_result["success"] 
                            for auth_result in result["authentication_results"].values()
                        )
                    }
            
            # Overall success if no errors
            result["success"] = len(result["errors"]) == 0
            
        except Exception as e:
            result["errors"].append(str(e))
            if self.logger:
                self.logger.error(f"Authentication pattern error: {e}")
        
        self.log_pattern_complete(context, result)
        return result