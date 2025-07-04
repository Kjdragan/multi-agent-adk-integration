"""
Helper functions and utilities for context management.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext

from .wrappers import (
    PlatformInvocationContext,
    PlatformCallbackContext,
    PlatformToolContext,
    PlatformReadonlyContext
)
from .managers import ContextManager
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService

ContextType = TypeVar('ContextType')


def context_with_logging(context: Any, logger: RunLogger) -> Any:
    """
    Enhance any context with logging capabilities.
    
    Args:
        context: ADK context object (InvocationContext, CallbackContext, etc.)
        logger: Platform logger instance
        
    Returns:
        Platform-enhanced context wrapper
    """
    if isinstance(context, InvocationContext):
        return PlatformInvocationContext(context, logger=logger)
    elif isinstance(context, CallbackContext):
        return PlatformCallbackContext(context, logger=logger)
    elif isinstance(context, ToolContext):
        # Extract tool name if available
        tool_name = getattr(context, 'function_name', 'unknown_tool')
        return PlatformToolContext(context, tool_name, logger=logger)
    elif isinstance(context, ReadonlyContext):
        return PlatformReadonlyContext(context, logger=logger)
    else:
        # Return original context if type not recognized
        return context


def context_with_services(context: Any,
                         logger: Optional[RunLogger] = None,
                         session_service: Optional[SessionService] = None,
                         memory_service: Optional[MemoryService] = None,
                         artifact_service: Optional[ArtifactService] = None) -> Any:
    """
    Enhance any context with full service integration.
    
    Args:
        context: ADK context object
        logger: Platform logger
        session_service: Session management service
        memory_service: Memory/knowledge service
        artifact_service: Artifact storage service
        
    Returns:
        Platform-enhanced context wrapper with full service access
    """
    if isinstance(context, InvocationContext):
        return PlatformInvocationContext(
            context,
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
    elif isinstance(context, CallbackContext):
        return PlatformCallbackContext(
            context,
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
    elif isinstance(context, ToolContext):
        tool_name = getattr(context, 'function_name', 'unknown_tool')
        return PlatformToolContext(
            context,
            tool_name,
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
    elif isinstance(context, ReadonlyContext):
        return PlatformReadonlyContext(context, logger=logger)
    else:
        return context


def context_with_performance_tracking(context: Any, 
                                    operation_name: str,
                                    logger: Optional[RunLogger] = None) -> Any:
    """
    Enhance context with automatic performance tracking.
    
    Args:
        context: ADK context object
        operation_name: Name of operation being tracked
        logger: Platform logger
        
    Returns:
        Enhanced context with performance tracking enabled
    """
    enhanced_context = context_with_logging(context, logger) if logger else context
    
    # Add performance tracking based on context type
    if hasattr(enhanced_context, 'start_tracking'):
        enhanced_context.start_tracking()
    elif hasattr(enhanced_context, 'start_execution'):
        enhanced_context.start_execution()
    elif hasattr(enhanced_context, 'setup_lifecycle_management'):
        enhanced_context.setup_lifecycle_management()
    
    return enhanced_context


@contextmanager
def managed_invocation_context(context: InvocationContext,
                             logger: Optional[RunLogger] = None,
                             session_service: Optional[SessionService] = None,
                             memory_service: Optional[MemoryService] = None,
                             artifact_service: Optional[ArtifactService] = None,
                             auto_health_check: bool = True):
    """
    Context manager for InvocationContext with automatic lifecycle management.
    
    Usage:
        with managed_invocation_context(ctx, logger) as managed_ctx:
            # Use managed_ctx with enhanced capabilities
            if managed_ctx.check_session_health():
                # Continue with processing
                pass
    """
    enhanced_ctx = PlatformInvocationContext(
        context,
        logger=logger,
        session_service=session_service,
        memory_service=memory_service,
        artifact_service=artifact_service
    )
    
    try:
        # Setup lifecycle management
        enhanced_ctx.setup_lifecycle_management()
        
        # Perform health check if requested
        if auto_health_check:
            if not enhanced_ctx.ensure_healthy_session():
                raise RuntimeError("Session health check failed")
        
        yield enhanced_ctx
        
    except Exception as e:
        # Handle errors gracefully
        if logger:
            logger.error(f"Error in managed invocation context: {e}")
        
        # Attempt early termination
        try:
            enhanced_ctx.terminate_early(f"exception: {str(e)}")
        except:
            pass  # Don't raise additional errors during cleanup
        
        raise
    
    finally:
        # Any cleanup operations would go here
        pass


@contextmanager
def managed_callback_context(context: CallbackContext,
                           callback_type: str,
                           logger: Optional[RunLogger] = None,
                           auto_tracking: bool = True):
    """
    Context manager for CallbackContext with automatic change tracking.
    
    Usage:
        with managed_callback_context(ctx, "before_agent", logger) as managed_ctx:
            managed_ctx.safe_state_update("key", "value")
            # Changes are automatically tracked
    """
    enhanced_ctx = PlatformCallbackContext(
        context,
        callback_type=callback_type,
        logger=logger
    )
    
    try:
        # Start tracking if requested
        if auto_tracking:
            enhanced_ctx.start_tracking()
        
        yield enhanced_ctx
        
    finally:
        # Finalize tracking
        if auto_tracking:
            changes = enhanced_ctx.finalize_tracking()
            if logger and changes:
                logger.debug(f"Callback {callback_type} changes: {changes}")


@contextmanager
def managed_tool_context(context: ToolContext,
                       tool_name: str,
                       logger: Optional[RunLogger] = None,
                       auto_execution_tracking: bool = True):
    """
    Context manager for ToolContext with automatic execution tracking.
    
    Usage:
        with managed_tool_context(ctx, "my_tool", logger) as managed_ctx:
            result = managed_ctx.search_memory("query")
            return managed_ctx.complete_execution(result)
    """
    enhanced_ctx = PlatformToolContext(
        context,
        tool_name=tool_name,
        logger=logger
    )
    
    try:
        # Start execution tracking if requested
        if auto_execution_tracking:
            enhanced_ctx.start_execution()
        
        yield enhanced_ctx
        
    except Exception as e:
        # Complete execution with error
        if auto_execution_tracking:
            enhanced_ctx.complete_execution(None, error=e)
        raise
    
    finally:
        # Any cleanup operations would go here
        pass


def create_context_manager(context_type: str,
                         logger: Optional[RunLogger] = None,
                         session_service: Optional[SessionService] = None,
                         memory_service: Optional[MemoryService] = None,
                         artifact_service: Optional[ArtifactService] = None) -> ContextManager:
    """
    Factory function to create appropriate context manager.
    
    Args:
        context_type: Type of context ("invocation", "callback", "tool", "readonly")
        logger: Platform logger
        session_service: Session service
        memory_service: Memory service
        artifact_service: Artifact service
        
    Returns:
        Appropriate context manager instance
    """
    from .managers import (
        InvocationContextManager,
        CallbackContextManager,
        ToolContextManager,
        ReadonlyContextManager
    )
    
    if context_type == "invocation":
        return InvocationContextManager(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
    elif context_type == "callback":
        return CallbackContextManager(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
    elif context_type == "tool":
        return ToolContextManager(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service
        )
    elif context_type == "readonly":
        return ReadonlyContextManager(logger=logger)
    else:
        raise ValueError(f"Unknown context type: {context_type}")


def validate_context_state(context: Any, 
                         required_keys: List[str],
                         logger: Optional[RunLogger] = None) -> Dict[str, bool]:
    """
    Validate that required state keys are present in context.
    
    Args:
        context: Any context object with state
        required_keys: List of required state keys
        logger: Optional logger for validation messages
        
    Returns:
        Dict mapping keys to their presence status
    """
    validation_results = {}
    
    try:
        if hasattr(context, 'state'):
            state = context.state
            
            for key in required_keys:
                if hasattr(state, 'get'):
                    present = state.get(key) is not None
                elif hasattr(state, '__contains__'):
                    present = key in state
                else:
                    present = hasattr(state, key)
                
                validation_results[key] = present
                
                if not present and logger:
                    logger.warning(f"Required state key missing: {key}")
        else:
            # No state available
            validation_results = {key: False for key in required_keys}
            
            if logger:
                logger.warning("Context has no state attribute")
    
    except Exception as e:
        # Error during validation
        validation_results = {key: False for key in required_keys}
        
        if logger:
            logger.error(f"Error validating context state: {e}")
    
    return validation_results


def extract_context_metadata(context: Any) -> Dict[str, Any]:
    """
    Extract metadata from any context object.
    
    Args:
        context: Any ADK context object
        
    Returns:
        Dictionary of metadata extracted from the context
    """
    metadata = {
        "context_type": type(context).__name__,
        "has_state": hasattr(context, 'state'),
        "has_invocation_id": hasattr(context, 'invocation_id'),
        "has_agent_name": hasattr(context, 'agent_name'),
    }
    
    # Add specific metadata based on context type
    if hasattr(context, 'invocation_id'):
        metadata["invocation_id"] = getattr(context, 'invocation_id', 'unknown')
    
    if hasattr(context, 'agent_name'):
        metadata["agent_name"] = getattr(context, 'agent_name', 'unknown')
    
    if hasattr(context, 'function_call_id'):
        metadata["function_call_id"] = getattr(context, 'function_call_id', 'unknown')
    
    if hasattr(context, 'state'):
        try:
            state = context.state
            if hasattr(state, 'keys'):
                metadata["state_keys"] = list(state.keys())
            elif hasattr(state, '__dict__'):
                metadata["state_keys"] = list(state.__dict__.keys())
            else:
                metadata["state_keys"] = []
        except:
            metadata["state_keys"] = []
    
    return metadata


def safe_context_operation(context: Any,
                         operation: str,
                         *args,
                         logger: Optional[RunLogger] = None,
                         default_result: Any = None,
                         **kwargs) -> Any:
    """
    Safely perform an operation on a context with error handling.
    
    Args:
        context: Context object
        operation: Method name to call
        *args: Arguments for the operation
        logger: Optional logger for error reporting
        default_result: Default result if operation fails
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of operation or default_result if failed
    """
    try:
        if hasattr(context, operation):
            method = getattr(context, operation)
            if callable(method):
                return method(*args, **kwargs)
            else:
                if logger:
                    logger.warning(f"Context operation {operation} is not callable")
                return default_result
        else:
            if logger:
                logger.warning(f"Context does not have operation: {operation}")
            return default_result
    
    except Exception as e:
        if logger:
            logger.error(f"Error performing context operation {operation}: {e}")
        return default_result


# Convenience functions for common context operations

def get_context_state_safely(context: Any, 
                            key: str, 
                            default: Any = None,
                            logger: Optional[RunLogger] = None) -> Any:
    """Safely get a state value from any context."""
    return safe_context_operation(
        context, 'state', 
        logger=logger,
        default_result={}
    ).get(key, default)


def update_context_state_safely(context: Any,
                               key: str,
                               value: Any,
                               logger: Optional[RunLogger] = None) -> bool:
    """Safely update a state value in any context."""
    try:
        if hasattr(context, 'state'):
            context.state[key] = value
            if logger:
                logger.debug(f"Updated context state: {key}")
            return True
        return False
    except Exception as e:
        if logger:
            logger.error(f"Error updating context state {key}: {e}")
        return False


def list_context_artifacts_safely(context: Any,
                                 logger: Optional[RunLogger] = None) -> List[str]:
    """Safely list artifacts from any context."""
    return safe_context_operation(
        context, 'list_artifacts',
        logger=logger,
        default_result=[]
    ) or []