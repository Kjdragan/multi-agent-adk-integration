"""
Main logging system implementation with failure-safe operation.
"""

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

from .models import LogConfig, RunContext, LogSummary, EventLog, StateChangeLog, ToolCallLog, PerformanceLog
from .handlers import RunDirectoryHandler, PerformanceTracker
from .formatters import LLMReadyFormatter, StructuredFormatter, PlainFormatter


class PlatformLogger:
    """
    Main logging system for the multi-agent research platform.
    
    Provides failure-safe, per-run organized logging with LLM-ready output.
    """
    
    def __init__(self, config: LogConfig):
        self.config = config
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for run contexts
        self._local = threading.local()
        
        # Global logger setup
        self._setup_global_logger()
        
        # Run management
        self._active_runs: Dict[str, RunDirectoryHandler] = {}
        self._performance_trackers: Dict[str, PerformanceTracker] = {}
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_runs": 0,
            "total_events": 0,
            "total_errors": 0,
            "start_time": datetime.utcnow()
        }
    
    def _setup_global_logger(self) -> None:
        """Set up global logging configuration."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level.value)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.log_level.value)
        
        if self.config.log_format.value == "json":
            console_handler.setFormatter(StructuredFormatter())
        elif self.config.log_format.value == "llm_ready":
            console_handler.setFormatter(LLMReadyFormatter())
        else:
            console_handler.setFormatter(PlainFormatter())
        
        root_logger.addHandler(console_handler)
        
        # Add global error log
        try:
            error_log_path = self.config.log_dir / "global_errors.log"
            error_handler = logging.FileHandler(str(error_log_path))
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(error_handler)
        except Exception:
            # If we can't create error log, continue without it
            pass
    
    @contextmanager
    def create_run_logger(self, invocation_id: str, session_id: str, 
                         user_id: str, initial_query: Optional[str] = None):
        """
        Create a logger for a specific run with automatic cleanup.
        
        Usage:
            with logger.create_run_logger(inv_id, sess_id, user_id) as run_logger:
                run_logger.info("Starting research...")
        """
        run_id = str(uuid.uuid4())
        
        run_context = RunContext(
            run_id=run_id,
            invocation_id=invocation_id,
            session_id=session_id,
            user_id=user_id,
            initial_query=initial_query
        )
        
        run_handler = None
        try:
            # Create run handler
            run_handler = RunDirectoryHandler(self.config.log_dir, run_context)
            
            # Create performance tracker
            perf_tracker = PerformanceTracker(run_handler)
            
            with self._lock:
                self._active_runs[run_id] = run_handler
                self._performance_trackers[run_id] = perf_tracker
                self._stats["total_runs"] += 1
            
            # Set thread-local context
            self._local.run_id = run_id
            self._local.run_context = run_context
            self._local.run_handler = run_handler
            self._local.performance_tracker = perf_tracker
            
            # Create run-specific logger
            run_logger = RunLogger(self, run_id)
            
            yield run_logger
            
        except Exception as e:
            # Log the error but don't crash
            logging.error(f"Failed to create run logger: {e}", exc_info=True)
            # Return a fallback logger that logs to global handlers
            yield FallbackLogger(self)
            
        finally:
            # Clean up
            try:
                if run_handler:
                    # Create run summary
                    summary = self._create_run_summary(run_context, run_handler, perf_tracker)
                    run_handler.create_summary(summary)
                    
                    # Close handler
                    run_handler.close()
                
                # Clean up thread-local
                if hasattr(self._local, 'run_id'):
                    delattr(self._local, 'run_id')
                if hasattr(self._local, 'run_context'):
                    delattr(self._local, 'run_context')
                if hasattr(self._local, 'run_handler'):
                    delattr(self._local, 'run_handler')
                if hasattr(self._local, 'performance_tracker'):
                    delattr(self._local, 'performance_tracker')
                
                # Remove from active runs
                with self._lock:
                    self._active_runs.pop(run_id, None)
                    self._performance_trackers.pop(run_id, None)
                    
            except Exception as e:
                logging.error(f"Error during run logger cleanup: {e}", exc_info=True)
    
    def _create_run_summary(self, run_context: RunContext, run_handler: RunDirectoryHandler, 
                           perf_tracker: PerformanceTracker) -> LogSummary:
        """Create a summary of the completed run."""
        run_context.end_time = datetime.utcnow()
        
        # Calculate duration
        duration_ms = (run_context.end_time - run_context.start_time).total_seconds() * 1000
        
        # Get performance metrics
        perf_metrics = perf_tracker.get_metrics_summary()
        
        # Create summary
        summary = LogSummary(
            run_context=run_context,
            total_events=len(run_context.agent_names),  # Approximation
            total_tool_calls=len(run_context.tool_names),  # Approximation
            total_state_changes=0,  # Would need to count from logs
            total_duration_ms=duration_ms,
            average_event_processing_ms=duration_ms / max(len(run_context.agent_names), 1),
            total_tokens_used=0,  # Would need to aggregate from events
            success_rate=100.0 if run_context.error_count == 0 else 90.0,  # Simplified
            agent_performance=perf_metrics,
            tool_performance=perf_metrics,
            key_insights=[
                f"Run completed in {duration_ms:.2f}ms",
                f"Involved {len(run_context.agent_names)} agents",
                f"Used {len(run_context.tool_names)} tools",
                f"Encountered {run_context.error_count} errors"
            ]
        )
        
        return summary
    
    def get_current_run_logger(self) -> Optional['RunLogger']:
        """Get the logger for the current thread's run."""
        if hasattr(self._local, 'run_id'):
            return RunLogger(self, self._local.run_id)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get platform logging statistics."""
        uptime = (datetime.utcnow() - self._stats["start_time"]).total_seconds()
        
        return {
            **self._stats,
            "uptime_seconds": uptime,
            "active_runs": len(self._active_runs),
            "runs_per_minute": self._stats["total_runs"] / (uptime / 60) if uptime > 0 else 0
        }


class RunLogger:
    """
    Logger for a specific run/invocation.
    
    Provides methods for logging events, state changes, tool calls, and performance.
    """
    
    def __init__(self, platform_logger: PlatformLogger, run_id: str):
        self.platform_logger = platform_logger
        self.run_id = run_id
        self._logger = logging.getLogger(f"run.{run_id[:8]}")
        
        # Add run-specific handlers if available
        with self.platform_logger._lock:
            if run_id in self.platform_logger._active_runs:
                run_handler = self.platform_logger._active_runs[run_id]
                
                # Add handlers to logger
                for handler in run_handler.handlers.values():
                    self._logger.addHandler(handler)
    
    @property
    def _run_handler(self) -> Optional[RunDirectoryHandler]:
        """Get the run handler for this logger."""
        with self.platform_logger._lock:
            return self.platform_logger._active_runs.get(self.run_id)
    
    @property
    def _performance_tracker(self) -> Optional[PerformanceTracker]:
        """Get the performance tracker for this logger."""
        with self.platform_logger._lock:
            return self.platform_logger._performance_trackers.get(self.run_id)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self._logger.debug(message, extra=self._get_extra(**kwargs))
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self._logger.info(message, extra=self._get_extra(**kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self._logger.warning(message, extra=self._get_extra(**kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self._logger.error(message, extra=self._get_extra(**kwargs))
        
        # Update error count
        if hasattr(self.platform_logger._local, 'run_context'):
            self.platform_logger._local.run_context.error_count += 1
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self._logger.critical(message, extra=self._get_extra(**kwargs))
        
        # Update error count
        if hasattr(self.platform_logger._local, 'run_context'):
            self.platform_logger._local.run_context.error_count += 1
    
    def log_event(self, event_log: EventLog) -> None:
        """Log an ADK event."""
        handler = self._run_handler
        if handler:
            handler.log_event(event_log)
        
        # Also log as regular message
        self.info(f"Event: {event_log.event_type} from {event_log.author}", 
                 event_id=event_log.event_id)
    
    def log_state_change(self, state_log: StateChangeLog) -> None:
        """Log a state change."""
        handler = self._run_handler
        if handler:
            handler.log_state_change(state_log)
        
        # Also log as regular message
        self.debug(f"State change: {state_log.state_key} = {state_log.new_value}",
                  agent_name=state_log.agent_name)
    
    def log_tool_call(self, tool_log: ToolCallLog) -> None:
        """Log a tool call."""
        handler = self._run_handler
        if handler:
            handler.log_tool_call(tool_log)
        
        # Also log as regular message
        self.info(f"Tool call: {tool_log.tool_name} by {tool_log.agent_name}",
                 tool_name=tool_log.tool_name, agent_name=tool_log.agent_name)
    
    def log_performance(self, perf_log: PerformanceLog) -> None:
        """Log performance metrics."""
        handler = self._run_handler
        if handler:
            handler.log_performance(perf_log)
    
    def start_timing(self, component_name: str) -> None:
        """Start timing a component."""
        tracker = self._performance_tracker
        if tracker:
            tracker.start_timing(component_name)
    
    def end_timing(self, component_name: str, component_type: str = "unknown", **kwargs) -> None:
        """End timing and log performance."""
        tracker = self._performance_tracker
        if tracker:
            invocation_id = getattr(self.platform_logger._local, 'run_context', None)
            invocation_id = invocation_id.invocation_id if invocation_id else ""
            tracker.end_timing(component_name, component_type, invocation_id, **kwargs)
    
    def _get_extra(self, **kwargs) -> Dict[str, Any]:
        """Get extra context for log messages."""
        extra = {}
        
        if hasattr(self.platform_logger._local, 'run_context'):
            context = self.platform_logger._local.run_context
            extra.update({
                'run_id': context.run_id,
                'invocation_id': context.invocation_id,
                'session_id': context.session_id,
                'user_id': context.user_id
            })
        
        extra.update(kwargs)
        return extra


class FallbackLogger:
    """
    Fallback logger when run-specific logging fails.
    
    Logs to global handlers to ensure no logs are lost.
    """
    
    def __init__(self, platform_logger: PlatformLogger):
        self.platform_logger = platform_logger
        self._logger = logging.getLogger("fallback")
    
    def debug(self, message: str, **kwargs) -> None:
        self._logger.debug(f"[FALLBACK] {message}", extra=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self._logger.info(f"[FALLBACK] {message}", extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self._logger.warning(f"[FALLBACK] {message}", extra=kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self._logger.error(f"[FALLBACK] {message}", extra=kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        self._logger.critical(f"[FALLBACK] {message}", extra=kwargs)
    
    def log_event(self, event_log: EventLog) -> None:
        self.info(f"Event: {event_log.event_type} from {event_log.author}")
    
    def log_state_change(self, state_log: StateChangeLog) -> None:
        self.debug(f"State change: {state_log.state_key} = {state_log.new_value}")
    
    def log_tool_call(self, tool_log: ToolCallLog) -> None:
        self.info(f"Tool call: {tool_log.tool_name} by {tool_log.agent_name}")
    
    def log_performance(self, perf_log: PerformanceLog) -> None:
        self.debug(f"Performance: {perf_log.component_name} took {perf_log.execution_time_ms}ms")
    
    def start_timing(self, component_name: str) -> None:
        pass  # No-op for fallback
    
    def end_timing(self, component_name: str, component_type: str = "unknown", **kwargs) -> None:
        pass  # No-op for fallback


# Global instance management
_platform_logger: Optional[PlatformLogger] = None


def setup_logging(config: LogConfig) -> PlatformLogger:
    """Set up the global platform logger."""
    global _platform_logger
    _platform_logger = PlatformLogger(config)
    return _platform_logger


def get_logger() -> Optional[PlatformLogger]:
    """Get the global platform logger."""
    return _platform_logger


def create_run_logger(invocation_id: str, session_id: str, user_id: str, 
                     initial_query: Optional[str] = None):
    """Create a run logger (convenience function)."""
    if _platform_logger:
        return _platform_logger.create_run_logger(invocation_id, session_id, user_id, initial_query)
    else:
        # Return a no-op context manager if no logger is set up
        from contextlib import nullcontext
        return nullcontext(FallbackLogger(None))