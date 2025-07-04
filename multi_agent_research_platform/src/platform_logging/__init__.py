"""
Multi-Agent Research Platform Logging System

Centralized, failure-safe logging with LLM-ready output formatting.
Provides per-run organization and multiple log levels.
"""

from .logger import (
    PlatformLogger,
    RunLogger,
    get_logger,
    setup_logging,
    create_run_logger,
)
from .models import (
    LogConfig,
    LogLevel,
    LogFormat,
    RunContext,
    EventLog,
    StateChangeLog,
    ToolCallLog,
    PerformanceLog,
)
from .formatters import (
    LLMReadyFormatter,
    StructuredFormatter,
    PlainFormatter,
)
from .handlers import (
    FailSafeFileHandler,
    RunDirectoryHandler,
    PerformanceTracker,
)

__all__ = [
    # Core logging
    "PlatformLogger",
    "RunLogger",
    "get_logger", 
    "setup_logging",
    "create_run_logger",
    
    # Models
    "LogConfig",
    "LogLevel",
    "LogFormat", 
    "RunContext",
    "EventLog",
    "StateChangeLog",
    "ToolCallLog",
    "PerformanceLog",
    
    # Formatters
    "LLMReadyFormatter",
    "StructuredFormatter", 
    "PlainFormatter",
    
    # Handlers
    "FailSafeFileHandler",
    "RunDirectoryHandler",
    "PerformanceTracker",
]