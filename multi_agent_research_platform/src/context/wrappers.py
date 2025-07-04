"""
Platform-enhanced context wrappers for ADK context objects.
"""

from typing import Any, Dict, List, Optional, Union
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext

from .managers import (
    InvocationContextManager, 
    CallbackContextManager, 
    ToolContextManager, 
    ReadonlyContextManager
)
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


class PlatformContextWrapper:
    """
    Base wrapper for platform-enhanced context objects.
    
    Provides common functionality for all platform context wrappers.
    """
    
    def __init__(self, 
                 context: Any,
                 manager: Any,
                 logger: Optional[RunLogger] = None):
        self._context = context
        self._manager = manager
        self._logger = logger
        self._enhanced_operations: Dict[str, Any] = {}
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped context."""
        return getattr(self._context, name)
    
    def get_enhanced_operation(self, operation: str) -> Any:
        """Get result of an enhanced operation."""
        return self._enhanced_operations.get(operation)
    
    def set_enhanced_operation(self, operation: str, result: Any) -> None:
        """Store result of an enhanced operation."""
        self._enhanced_operations[operation] = result


class PlatformInvocationContext(PlatformContextWrapper):
    """
    Platform-enhanced InvocationContext.
    
    Adds lifecycle management, health monitoring, and cross-agent coordination
    to the standard ADK InvocationContext.
    """
    
    def __init__(self, 
                 context: InvocationContext,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        manager = InvocationContextManager(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
        
        super().__init__(context, manager, logger)
        self._lifecycle_managed = False
    
    def setup_lifecycle_management(self) -> 'PlatformInvocationContext':
        """Set up comprehensive lifecycle management."""
        if not self._lifecycle_managed:
            self._manager.manage_invocation_lifecycle(self._context)
            self._lifecycle_managed = True
        return self
    
    def terminate_early(self, reason: str) -> 'PlatformInvocationContext':
        """Terminate the invocation early with proper logging."""
        self._manager.handle_early_termination(self._context, reason)
        return self
    
    def check_session_health(self) -> bool:
        """Check and log session health status."""
        is_healthy = self._manager.monitor_session_health(self._context)
        self.set_enhanced_operation("session_health", is_healthy)
        return is_healthy
    
    def get_shared_state(self, keys: List[str]) -> Dict[str, Any]:
        """Get state shared between agents with tracking."""
        shared_state = self._manager.manage_cross_agent_state(self._context, keys)
        self.set_enhanced_operation("shared_state", shared_state)
        return shared_state
    
    def ensure_healthy_session(self) -> bool:
        """Ensure session is healthy, terminate if not."""
        if not self.check_session_health():
            self.terminate_early("unhealthy_session")
            return False
        return True
    
    @property
    def enhanced_session(self):
        """Access to enhanced session operations."""
        return {
            "health_status": self.get_enhanced_operation("session_health"),
            "shared_state": self.get_enhanced_operation("shared_state"),
            "lifecycle_managed": self._lifecycle_managed
        }


class PlatformCallbackContext(PlatformContextWrapper):
    """
    Platform-enhanced CallbackContext.
    
    Adds state change tracking, artifact monitoring, and performance measurement
    to the standard ADK CallbackContext.
    """
    
    def __init__(self, 
                 context: CallbackContext,
                 callback_type: str = "unknown",
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        manager = CallbackContextManager(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
        
        super().__init__(context, manager, logger)
        self.callback_type = callback_type
        self._tracking_started = False
        self._initial_state: Optional[Dict[str, Any]] = None
        self._initial_artifacts: Optional[List[str]] = None
    
    def start_tracking(self) -> 'PlatformCallbackContext':
        """Start tracking state and artifact changes."""
        if not self._tracking_started:
            self._initial_state = self._manager.manage_callback_state(
                self._context, self.callback_type
            )
            self._initial_artifacts = self._manager.manage_callback_artifacts(
                self._context, self.callback_type
            )
            self._tracking_started = True
        return self
    
    def finalize_tracking(self) -> Dict[str, Any]:
        """Finalize tracking and return change summary."""
        if not self._tracking_started:
            return {}
        
        changes = {}
        
        # Track state changes
        if self._initial_state is not None:
            state_changes = self._manager.track_state_changes(
                self._context, self._initial_state, self.callback_type
            )
            changes["state"] = state_changes
        
        # Track artifact changes
        if self._initial_artifacts is not None:
            artifact_changes = self._manager.track_artifact_changes(
                self._context, self._initial_artifacts, self.callback_type
            )
            changes["artifacts"] = artifact_changes
        
        self.set_enhanced_operation("changes", changes)
        return changes
    
    def safe_state_update(self, key: str, value: Any) -> bool:
        """Safely update state with validation and logging."""
        try:
            if not self._tracking_started:
                self.start_tracking()
            
            # Validate key
            if not isinstance(key, str) or not key:
                if self._logger:
                    self._logger.warning(f"Invalid state key: {key}")
                return False
            
            # Validate value (should be serializable)
            try:
                import json
                json.dumps(value)
            except (TypeError, ValueError):
                if self._logger:
                    self._logger.warning(f"Non-serializable value for key {key}")
                return False
            
            # Update state
            if hasattr(self._context, 'state'):
                old_value = getattr(self._context.state, key, None)
                self._context.state[key] = value
                
                if self._logger:
                    self._logger.debug(
                        f"State updated: {key}",
                        key=key,
                        old_value=old_value,
                        new_value=value,
                        callback_type=self.callback_type
                    )
                
                return True
            
            return False
            
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error updating state {key}: {e}")
            return False
    
    def safe_artifact_save(self, filename: str, content: Any) -> bool:
        """Safely save artifact with validation and logging."""
        try:
            if not self._tracking_started:
                self.start_tracking()
            
            if hasattr(self._context, 'save_artifact'):
                version = self._context.save_artifact(filename, content)
                
                if self._logger:
                    self._logger.debug(
                        f"Artifact saved: {filename}",
                        filename=filename,
                        version=version,
                        callback_type=self.callback_type
                    )
                
                return version is not None
            
            return False
            
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error saving artifact {filename}: {e}")
            return False
    
    @property
    def enhanced_tracking(self):
        """Access to enhanced tracking information."""
        return {
            "tracking_started": self._tracking_started,
            "callback_type": self.callback_type,
            "changes": self.get_enhanced_operation("changes"),
            "initial_state_keys": list(self._initial_state.keys()) if self._initial_state else [],
            "initial_artifact_count": len(self._initial_artifacts) if self._initial_artifacts else 0
        }


class PlatformToolContext(PlatformContextWrapper):
    """
    Platform-enhanced ToolContext.
    
    Adds execution management, authentication handling, memory access,
    and comprehensive logging to the standard ADK ToolContext.
    """
    
    def __init__(self, 
                 context: ToolContext,
                 tool_name: str,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        manager = ToolContextManager(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
        
        super().__init__(context, manager, logger)
        self.tool_name = tool_name
        self._execution_info: Optional[Dict[str, Any]] = None
        self._execution_started = False
    
    def start_execution(self) -> 'PlatformToolContext':
        """Start tool execution with comprehensive tracking."""
        if not self._execution_started:
            self._execution_info = self._manager.manage_tool_execution(
                self._context, self.tool_name
            )
            self._execution_started = True
        return self
    
    def handle_authentication(self, auth_config: Any) -> bool:
        """Handle tool authentication with comprehensive logging."""
        if not self._execution_started:
            self.start_execution()
        
        success = self._manager.handle_tool_authentication(self._context, auth_config)
        self.set_enhanced_operation("authentication", {
            "success": success,
            "config_type": type(auth_config).__name__
        })
        return success
    
    def search_memory(self, query: str) -> Optional[Any]:
        """Search memory with enhanced tracking."""
        if not self._execution_started:
            self.start_execution()
        
        results = self._manager.manage_tool_memory_access(self._context, query)
        self.set_enhanced_operation("memory_search", {
            "query": query,
            "result_count": len(results.results) if results else 0
        })
        return results
    
    def list_artifacts(self) -> List[str]:
        """List artifacts with enhanced tracking."""
        if not self._execution_started:
            self.start_execution()
        
        artifacts = self._manager.manage_tool_artifacts(self._context)
        self.set_enhanced_operation("artifacts", artifacts)
        return artifacts
    
    def complete_execution(self, result: Any, error: Optional[Exception] = None) -> Any:
        """Complete tool execution with comprehensive logging."""
        if self._execution_started and self._execution_info:
            self._manager.complete_tool_execution(
                self._context, self._execution_info, result, error
            )
        
        self.set_enhanced_operation("execution_result", {
            "result": result,
            "error": str(error) if error else None,
            "success": error is None
        })
        
        return result
    
    def safe_state_access(self, key: str, default: Any = None) -> Any:
        """Safely access state with fallback."""
        try:
            if hasattr(self._context, 'state'):
                return getattr(self._context.state, key, default)
            return default
        except Exception as e:
            if self._logger:
                self._logger.warning(f"Error accessing state key {key}: {e}")
            return default
    
    def safe_state_update(self, key: str, value: Any) -> bool:
        """Safely update state with validation."""
        try:
            if hasattr(self._context, 'state'):
                self._context.state[key] = value
                
                if self._logger:
                    self._logger.debug(
                        f"Tool state updated: {key}",
                        tool_name=self.tool_name,
                        key=key
                    )
                
                return True
            return False
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error updating tool state {key}: {e}")
            return False
    
    @property
    def enhanced_execution(self):
        """Access to enhanced execution information."""
        return {
            "tool_name": self.tool_name,
            "execution_started": self._execution_started,
            "execution_info": self._execution_info,
            "authentication": self.get_enhanced_operation("authentication"),
            "memory_search": self.get_enhanced_operation("memory_search"),
            "artifacts": self.get_enhanced_operation("artifacts"),
            "execution_result": self.get_enhanced_operation("execution_result")
        }


class PlatformReadonlyContext(PlatformContextWrapper):
    """
    Platform-enhanced ReadonlyContext.
    
    Adds safe access patterns and comprehensive logging to the standard ADK ReadonlyContext.
    """
    
    def __init__(self, 
                 context: ReadonlyContext,
                 logger: Optional[RunLogger] = None):
        
        manager = ReadonlyContextManager(logger=logger)
        super().__init__(context, manager, logger)
    
    def safe_get_state(self, keys: Union[str, List[str]], 
                      defaults: Union[Any, Dict[str, Any]] = None) -> Any:
        """Safely get state values with defaults."""
        if isinstance(keys, str):
            # Single key access
            result = self._manager.safe_state_access(self._context, [keys])
            return result.get(keys, defaults)
        else:
            # Multiple key access
            result = self._manager.safe_state_access(self._context, keys)
            if isinstance(defaults, dict):
                return {key: result.get(key, defaults.get(key)) for key in keys}
            else:
                return {key: result.get(key, defaults) for key in keys}
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get comprehensive context information."""
        summary = self._manager.get_context_summary(self._context)
        self.set_enhanced_operation("context_info", summary)
        return summary
    
    def safe_get_invocation_id(self) -> str:
        """Safely get invocation ID with fallback."""
        try:
            return getattr(self._context, 'invocation_id', 'unknown')
        except:
            return 'unknown'
    
    def safe_get_agent_name(self) -> str:
        """Safely get agent name with fallback."""
        try:
            return getattr(self._context, 'agent_name', 'unknown')
        except:
            return 'unknown'
    
    @property
    def enhanced_info(self):
        """Access to enhanced context information."""
        return {
            "context_info": self.get_enhanced_operation("context_info"),
            "invocation_id": self.safe_get_invocation_id(),
            "agent_name": self.safe_get_agent_name()
        }