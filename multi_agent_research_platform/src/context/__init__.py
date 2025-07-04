"""
Context Management for Multi-Agent Research Platform

Provides context managers and utilities for working with ADK's different context types:
- InvocationContext: Full session control in agent core implementations
- CallbackContext: State and artifact management in callbacks
- ToolContext: Tool execution with authentication and memory access
- ReadonlyContext: Safe read-only access to basic information

These managers integrate with the platform's logging, services, and configuration systems.
"""

from .managers import (
    ContextManager,
    InvocationContextManager,
    CallbackContextManager,
    ToolContextManager,
    ReadonlyContextManager,
)
from .wrappers import (
    PlatformInvocationContext,
    PlatformCallbackContext,
    PlatformToolContext,
    PlatformReadonlyContext,
)
from .helpers import (
    context_with_logging,
    context_with_services,
    context_with_performance_tracking,
    create_context_manager,
)
from .patterns import (
    ContextPattern,
    AgentContextPattern,
    ToolContextPattern,
    CallbackContextPattern,
    StateManagementPattern,
    ArtifactManagementPattern,
    MemoryAccessPattern,
    AuthenticationPattern,
)

__all__ = [
    # Core context managers
    "ContextManager",
    "InvocationContextManager",
    "CallbackContextManager", 
    "ToolContextManager",
    "ReadonlyContextManager",
    
    # Platform-enhanced context wrappers
    "PlatformInvocationContext",
    "PlatformCallbackContext",
    "PlatformToolContext", 
    "PlatformReadonlyContext",
    
    # Helper functions
    "context_with_logging",
    "context_with_services",
    "context_with_performance_tracking",
    "create_context_manager",
    
    # Context patterns
    "ContextPattern",
    "AgentContextPattern",
    "ToolContextPattern",
    "CallbackContextPattern",
    "StateManagementPattern",
    "ArtifactManagementPattern",
    "MemoryAccessPattern",
    "AuthenticationPattern",
]