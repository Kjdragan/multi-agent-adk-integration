"""
Context managers for different ADK context types.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext

from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService
from ..config.agents import AgentConfig


class ContextManager(ABC):
    """
    Abstract base class for context managers.
    
    Provides common functionality for all context types including logging,
    performance tracking, and service integration.
    """
    
    def __init__(self, 
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Performance tracking
        self._start_times: Dict[str, float] = {}
        self._operation_counts: Dict[str, int] = {}
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        self._start_times[operation] = time.time()
        if self.logger:
            self.logger.start_timing(f"context_{operation}")
    
    def end_timing(self, operation: str, **metadata) -> float:
        """End timing and return duration."""
        if operation in self._start_times:
            duration = time.time() - self._start_times[operation]
            del self._start_times[operation]
            
            # Update operation counts
            self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
            
            # Log performance
            if self.logger:
                self.logger.end_timing(
                    f"context_{operation}", 
                    component_type="context",
                    operation_count=self._operation_counts[operation],
                    **metadata
                )
            
            return duration
        return 0.0
    
    def log_context_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log a context operation."""
        if self.logger:
            self.logger.debug(
                f"Context operation: {operation}",
                operation=operation,
                context_type=self.__class__.__name__,
                **details
            )
    
    @abstractmethod
    def get_context_type(self) -> str:
        """Get the type of context this manager handles."""
        pass


class InvocationContextManager(ContextManager):
    """
    Manager for InvocationContext operations.
    
    Provides enhanced functionality for agent core implementations that need
    full session control and service access.
    """
    
    def get_context_type(self) -> str:
        return "invocation"
    
    def manage_invocation_lifecycle(self, context: InvocationContext) -> None:
        """Set up invocation lifecycle management."""
        self.start_timing("invocation_setup")
        
        # Log invocation start
        if self.logger:
            self.logger.info(
                f"Starting invocation {context.invocation_id}",
                agent_name=context.agent.name if context.agent else "unknown",
                session_id=context.session.id if context.session else "unknown"
            )
        
        self.log_context_operation("invocation_start", {
            "invocation_id": context.invocation_id,
            "agent_name": getattr(context.agent, 'name', 'unknown'),
            "session_id": getattr(context.session, 'id', 'unknown')
        })
        
        self.end_timing("invocation_setup")
    
    def handle_early_termination(self, context: InvocationContext, reason: str) -> None:
        """Handle early termination of invocation."""
        self.start_timing("early_termination")
        
        # Set termination flag
        context.end_invocation = True
        
        # Log termination
        if self.logger:
            self.logger.warning(
                f"Early termination of invocation {context.invocation_id}: {reason}",
                reason=reason,
                agent_name=getattr(context.agent, 'name', 'unknown')
            )
        
        self.log_context_operation("early_termination", {
            "invocation_id": context.invocation_id,
            "reason": reason
        })
        
        self.end_timing("early_termination", reason=reason)
    
    def monitor_session_health(self, context: InvocationContext) -> bool:
        """Monitor session health and return True if healthy."""
        self.start_timing("health_check")
        
        try:
            session = context.session
            if not session:
                return False
            
            # Check session validity
            is_healthy = True
            health_issues = []
            
            # Check session age
            if hasattr(session, 'last_update_time'):
                from datetime import datetime, timedelta
                age = datetime.utcnow() - session.last_update_time
                if age > timedelta(hours=24):
                    health_issues.append("session_too_old")
                    is_healthy = False
            
            # Check event count (potential memory issue)
            if len(session.events) > 1000:
                health_issues.append("too_many_events")
                is_healthy = False
            
            # Check state size
            state_size = len(str(session.state))
            if state_size > 1024 * 1024:  # 1MB
                health_issues.append("state_too_large")
                is_healthy = False
            
            # Log health status
            self.log_context_operation("health_check", {
                "is_healthy": is_healthy,
                "issues": health_issues,
                "event_count": len(session.events),
                "state_size": state_size
            })
            
            self.end_timing("health_check", 
                          is_healthy=is_healthy, 
                          issue_count=len(health_issues))
            
            return is_healthy
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during session health check: {e}")
            self.end_timing("health_check", error=str(e))
            return False
    
    def manage_cross_agent_state(self, context: InvocationContext, 
                                shared_keys: List[str]) -> Dict[str, Any]:
        """Manage state shared between agents."""
        self.start_timing("cross_agent_state")
        
        shared_state = {}
        
        for key in shared_keys:
            if key in context.session.state:
                shared_state[key] = context.session.state[key]
        
        self.log_context_operation("cross_agent_state", {
            "requested_keys": shared_keys,
            "found_keys": list(shared_state.keys()),
            "missing_keys": [k for k in shared_keys if k not in shared_state]
        })
        
        self.end_timing("cross_agent_state", key_count=len(shared_state))
        return shared_state


class CallbackContextManager(ContextManager):
    """
    Manager for CallbackContext operations.
    
    Provides enhanced functionality for agent lifecycle callbacks including
    state management and artifact operations.
    """
    
    def get_context_type(self) -> str:
        return "callback"
    
    def manage_callback_state(self, context: CallbackContext, 
                            operation: str) -> Dict[str, Any]:
        """Manage state changes during callback execution."""
        self.start_timing(f"callback_state_{operation}")
        
        # Capture initial state
        initial_state = dict(context.state) if hasattr(context, 'state') else {}
        
        self.log_context_operation(f"callback_{operation}_start", {
            "initial_state_keys": list(initial_state.keys()),
            "state_size": len(initial_state)
        })
        
        return initial_state
    
    def track_state_changes(self, context: CallbackContext, 
                          initial_state: Dict[str, Any],
                          operation: str) -> Dict[str, Any]:
        """Track and log state changes made during callback."""
        if not hasattr(context, 'state'):
            return {}
        
        current_state = dict(context.state)
        changes = {}
        
        # Find additions and modifications
        for key, value in current_state.items():
            if key not in initial_state:
                changes[key] = {"type": "added", "value": value}
            elif initial_state[key] != value:
                changes[key] = {
                    "type": "modified", 
                    "old_value": initial_state[key],
                    "new_value": value
                }
        
        # Find deletions
        for key in initial_state:
            if key not in current_state:
                changes[key] = {"type": "deleted", "old_value": initial_state[key]}
        
        # Log changes
        if changes:
            self.log_context_operation(f"callback_{operation}_changes", {
                "changes": {k: v["type"] for k, v in changes.items()},
                "change_count": len(changes)
            })
        
        self.end_timing(f"callback_state_{operation}", 
                       change_count=len(changes))
        
        return changes
    
    def manage_callback_artifacts(self, context: CallbackContext,
                                operation: str) -> List[str]:
        """Manage artifact operations during callbacks."""
        self.start_timing(f"callback_artifacts_{operation}")
        
        artifacts_before = []
        if hasattr(context, 'list_artifacts'):
            try:
                artifacts_before = context.list_artifacts() or []
            except:
                artifacts_before = []
        
        self.log_context_operation(f"callback_{operation}_artifacts_start", {
            "existing_artifacts": len(artifacts_before)
        })
        
        return artifacts_before
    
    def track_artifact_changes(self, context: CallbackContext,
                             initial_artifacts: List[str],
                             operation: str) -> Dict[str, str]:
        """Track artifact changes made during callback."""
        if not hasattr(context, 'list_artifacts'):
            return {}
        
        try:
            current_artifacts = context.list_artifacts() or []
        except:
            current_artifacts = []
        
        changes = {}
        
        # Find new artifacts
        for artifact in current_artifacts:
            if artifact not in initial_artifacts:
                changes[artifact] = "added"
        
        # Find removed artifacts
        for artifact in initial_artifacts:
            if artifact not in current_artifacts:
                changes[artifact] = "removed"
        
        # Log changes
        if changes:
            self.log_context_operation(f"callback_{operation}_artifact_changes", {
                "changes": changes,
                "change_count": len(changes)
            })
        
        self.end_timing(f"callback_artifacts_{operation}",
                       change_count=len(changes))
        
        return changes


class ToolContextManager(ContextManager):
    """
    Manager for ToolContext operations.
    
    Provides enhanced functionality for tool execution including authentication,
    memory access, and artifact management.
    """
    
    def get_context_type(self) -> str:
        return "tool"
    
    def manage_tool_execution(self, context: ToolContext, 
                            tool_name: str) -> Dict[str, Any]:
        """Set up tool execution with comprehensive tracking."""
        self.start_timing(f"tool_{tool_name}")
        
        execution_info = {
            "tool_name": tool_name,
            "function_call_id": getattr(context, 'function_call_id', 'unknown'),
            "agent_name": getattr(context, 'agent_name', 'unknown'),
            "invocation_id": getattr(context, 'invocation_id', 'unknown'),
            "start_time": time.time()
        }
        
        # Log tool execution start
        if self.logger:
            self.logger.info(
                f"Starting tool execution: {tool_name}",
                tool_name=tool_name,
                agent_name=execution_info["agent_name"]
            )
        
        self.log_context_operation("tool_start", execution_info)
        
        return execution_info
    
    def handle_tool_authentication(self, context: ToolContext,
                                 auth_config: Any) -> bool:
        """Handle tool authentication with tracking."""
        self.start_timing("tool_auth")
        
        try:
            # Check if authentication is needed
            if hasattr(context, 'request_credential'):
                # Request credentials
                context.request_credential(auth_config)
                
                self.log_context_operation("tool_auth_requested", {
                    "auth_type": type(auth_config).__name__,
                    "function_call_id": getattr(context, 'function_call_id', 'unknown')
                })
                
                self.end_timing("tool_auth", auth_status="requested")
                return True
            
            # Check for existing credentials
            if hasattr(context, 'get_auth_response'):
                auth_response = context.get_auth_response(auth_config)
                if auth_response:
                    self.log_context_operation("tool_auth_retrieved", {
                        "auth_type": type(auth_config).__name__
                    })
                    self.end_timing("tool_auth", auth_status="retrieved")
                    return True
            
            self.end_timing("tool_auth", auth_status="not_needed")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tool authentication error: {e}")
            
            self.end_timing("tool_auth", auth_status="error", error=str(e))
            return False
    
    def manage_tool_memory_access(self, context: ToolContext, 
                                query: str) -> Optional[Any]:
        """Manage memory access for tools."""
        self.start_timing("tool_memory")
        
        try:
            if hasattr(context, 'search_memory'):
                results = context.search_memory(query)
                
                self.log_context_operation("tool_memory_search", {
                    "query": query[:100],  # Truncate for logging
                    "result_count": len(results.results) if results else 0
                })
                
                self.end_timing("tool_memory", 
                              result_count=len(results.results) if results else 0)
                return results
            
            self.end_timing("tool_memory", status="not_available")
            return None
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tool memory access error: {e}")
            
            self.end_timing("tool_memory", status="error", error=str(e))
            return None
    
    def manage_tool_artifacts(self, context: ToolContext) -> List[str]:
        """Manage artifact access for tools."""
        self.start_timing("tool_artifacts")
        
        try:
            if hasattr(context, 'list_artifacts'):
                artifacts = context.list_artifacts() or []
                
                self.log_context_operation("tool_artifacts_list", {
                    "artifact_count": len(artifacts)
                })
                
                self.end_timing("tool_artifacts", artifact_count=len(artifacts))
                return artifacts
            
            self.end_timing("tool_artifacts", status="not_available")
            return []
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Tool artifact access error: {e}")
            
            self.end_timing("tool_artifacts", status="error", error=str(e))
            return []
    
    def complete_tool_execution(self, context: ToolContext,
                              execution_info: Dict[str, Any],
                              result: Any,
                              error: Optional[Exception] = None) -> None:
        """Complete tool execution with comprehensive logging."""
        tool_name = execution_info["tool_name"]
        duration = time.time() - execution_info["start_time"]
        
        # Log completion
        if error:
            if self.logger:
                self.logger.error(
                    f"Tool execution failed: {tool_name}",
                    tool_name=tool_name,
                    error=str(error),
                    duration_ms=duration * 1000
                )
            
            self.log_context_operation("tool_error", {
                **execution_info,
                "error": str(error),
                "duration_ms": duration * 1000
            })
        else:
            if self.logger:
                self.logger.info(
                    f"Tool execution completed: {tool_name}",
                    tool_name=tool_name,
                    duration_ms=duration * 1000
                )
            
            self.log_context_operation("tool_complete", {
                **execution_info,
                "result_type": type(result).__name__,
                "duration_ms": duration * 1000
            })
        
        self.end_timing(f"tool_{tool_name}",
                       success=error is None,
                       result_type=type(result).__name__ if result else None)


class ReadonlyContextManager(ContextManager):
    """
    Manager for ReadonlyContext operations.
    
    Provides safe read-only access to context information with logging.
    """
    
    def get_context_type(self) -> str:
        return "readonly"
    
    def safe_state_access(self, context: ReadonlyContext, 
                         keys: List[str]) -> Dict[str, Any]:
        """Safely access state with comprehensive logging."""
        self.start_timing("readonly_state_access")
        
        accessed_state = {}
        
        try:
            if hasattr(context, 'state'):
                for key in keys:
                    if hasattr(context.state, 'get'):
                        value = context.state.get(key)
                        if value is not None:
                            accessed_state[key] = value
                    elif hasattr(context.state, '__getitem__'):
                        try:
                            value = context.state[key]
                            accessed_state[key] = value
                        except (KeyError, AttributeError):
                            pass
            
            self.log_context_operation("readonly_state_access", {
                "requested_keys": keys,
                "found_keys": list(accessed_state.keys()),
                "access_count": len(accessed_state)
            })
            
            self.end_timing("readonly_state_access", 
                          access_count=len(accessed_state))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Readonly state access error: {e}")
            
            self.end_timing("readonly_state_access", 
                          error=str(e), access_count=0)
        
        return accessed_state
    
    def get_context_summary(self, context: ReadonlyContext) -> Dict[str, Any]:
        """Get a summary of context information."""
        self.start_timing("context_summary")
        
        summary = {
            "context_type": "readonly",
            "has_invocation_id": hasattr(context, 'invocation_id'),
            "has_agent_name": hasattr(context, 'agent_name'),
            "has_state": hasattr(context, 'state'),
        }
        
        # Add available information
        if hasattr(context, 'invocation_id'):
            summary["invocation_id"] = getattr(context, 'invocation_id', 'unknown')
        
        if hasattr(context, 'agent_name'):
            summary["agent_name"] = getattr(context, 'agent_name', 'unknown')
        
        self.log_context_operation("context_summary", summary)
        self.end_timing("context_summary")
        
        return summary