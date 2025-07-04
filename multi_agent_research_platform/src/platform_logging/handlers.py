"""
Custom logging handlers for failure-safe operation and run organization.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
from .models import RunContext, LogSummary, EventLog, StateChangeLog, ToolCallLog, PerformanceLog
from .formatters import EventLogFormatter


class FailSafeFileHandler(logging.FileHandler):
    """
    File handler that continues operating even if primary log file fails.
    
    Implements multiple backup strategies to ensure logs are never lost.
    """
    
    def __init__(self, filename: str, mode: str = 'a', encoding: Optional[str] = None, 
                 delay: bool = False, backup_dir: Optional[Path] = None):
        self.backup_dir = backup_dir or Path(filename).parent / "backup"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_counter = 0
        self.fallback_stream: Optional[TextIO] = None
        
        super().__init__(filename, mode, encoding, delay)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with failure safety."""
        try:
            # Try primary handler
            super().emit(record)
        except Exception as e:
            # Primary failed, try backup strategies
            self._emit_to_backup(record, e)
    
    def _emit_to_backup(self, record: logging.LogRecord, original_error: Exception) -> None:
        """Emit to backup location when primary fails."""
        try:
            # Strategy 1: Write to backup directory
            self.backup_counter += 1
            backup_filename = self.backup_dir / f"backup_{self.backup_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(backup_filename, 'a', encoding=self.encoding) as backup_file:
                backup_file.write(f"[BACKUP LOG - PRIMARY FAILED: {original_error}]\n")
                backup_file.write(self.format(record) + '\n')
                backup_file.flush()
                
        except Exception:
            # Strategy 2: Write to stderr if all else fails
            try:
                import sys
                sys.stderr.write(f"[EMERGENCY LOG]: {self.format(record)}\n")
                sys.stderr.write(f"[EMERGENCY LOG ERROR]: Primary log failed: {original_error}\n")
                sys.stderr.flush()
            except Exception:
                # Strategy 3: Silent failure - at least don't crash the application
                pass


class RunDirectoryHandler:
    """
    Handler that creates and manages per-run log directories.
    
    Organizes logs by run ID with multiple log levels and specialized files.
    """
    
    def __init__(self, base_log_dir: Path, run_context: RunContext):
        self.base_log_dir = base_log_dir
        self.run_context = run_context
        self.run_dir = self._create_run_directory()
        self.handlers: Dict[str, logging.Handler] = {}
        self.specialized_files: Dict[str, TextIO] = {}
        
        self._setup_handlers()
        self._setup_specialized_files()
    
    def _create_run_directory(self) -> Path:
        """Create directory for this run's logs."""
        timestamp = self.run_context.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        run_id_short = self.run_context.run_id[:8]
        
        run_dir = self.base_log_dir / "runs" / f"{timestamp}_{run_id_short}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlink to latest run
        latest_link = self.base_log_dir / "runs" / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
        
        return run_dir
    
    def _setup_handlers(self) -> None:
        """Set up handlers for different log levels."""
        from .formatters import StructuredFormatter, LLMReadyFormatter, PlainFormatter
        
        # Debug log - detailed information
        debug_handler = FailSafeFileHandler(
            str(self.run_dir / "debug.log"),
            backup_dir=self.run_dir / "backup"
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(StructuredFormatter())
        self.handlers["debug"] = debug_handler
        
        # Info log - general operation
        info_handler = FailSafeFileHandler(
            str(self.run_dir / "info.log"),
            backup_dir=self.run_dir / "backup"
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(LLMReadyFormatter())
        self.handlers["info"] = info_handler
        
        # Error log - errors and warnings only
        error_handler = FailSafeFileHandler(
            str(self.run_dir / "error.log"),
            backup_dir=self.run_dir / "backup"
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(PlainFormatter())
        self.handlers["error"] = error_handler
    
    def _setup_specialized_files(self) -> None:
        """Set up specialized log files for different data types."""
        try:
            self.specialized_files["events"] = open(self.run_dir / "events.jsonl", "a")
            self.specialized_files["state_changes"] = open(self.run_dir / "state_changes.jsonl", "a") 
            self.specialized_files["tool_calls"] = open(self.run_dir / "tool_calls.jsonl", "a")
            self.specialized_files["performance"] = open(self.run_dir / "performance.jsonl", "a")
        except Exception as e:
            # Use backup strategy
            backup_dir = self.run_dir / "backup"
            backup_dir.mkdir(exist_ok=True)
            for name in ["events", "state_changes", "tool_calls", "performance"]:
                try:
                    self.specialized_files[name] = open(backup_dir / f"{name}.jsonl", "a")
                except Exception:
                    # Ultimate fallback - None will cause methods to skip specialized logging
                    self.specialized_files[name] = None
    
    def get_handler(self, level: str) -> Optional[logging.Handler]:
        """Get handler for specific log level."""
        return self.handlers.get(level)
    
    def log_event(self, event_log: EventLog) -> None:
        """Log an ADK event to specialized file."""
        if self.specialized_files.get("events"):
            try:
                formatted = EventLogFormatter.format_event_log(event_log)
                self.specialized_files["events"].write(json.dumps(formatted) + "\n")
                self.specialized_files["events"].flush()
            except Exception:
                pass  # Fail silently for specialized logs
    
    def log_state_change(self, state_log: StateChangeLog) -> None:
        """Log a state change to specialized file."""
        if self.specialized_files.get("state_changes"):
            try:
                formatted = EventLogFormatter.format_state_change_log(state_log)
                self.specialized_files["state_changes"].write(json.dumps(formatted) + "\n")
                self.specialized_files["state_changes"].flush()
            except Exception:
                pass
    
    def log_tool_call(self, tool_log: ToolCallLog) -> None:
        """Log a tool call to specialized file."""
        if self.specialized_files.get("tool_calls"):
            try:
                formatted = EventLogFormatter.format_tool_call_log(tool_log)
                self.specialized_files["tool_calls"].write(json.dumps(formatted) + "\n")
                self.specialized_files["tool_calls"].flush()
            except Exception:
                pass
    
    def log_performance(self, perf_log: PerformanceLog) -> None:
        """Log performance metrics to specialized file."""
        if self.specialized_files.get("performance"):
            try:
                formatted = EventLogFormatter.format_performance_log(perf_log)
                self.specialized_files["performance"].write(json.dumps(formatted) + "\n")
                self.specialized_files["performance"].flush()
            except Exception:
                pass
    
    def create_summary(self, summary: LogSummary) -> None:
        """Create a run summary for LLM analysis."""
        try:
            summary_file = self.run_dir / "summary.json"
            with open(summary_file, "w") as f:
                f.write(summary.model_dump_json(indent=2))
        except Exception:
            # Try backup location
            try:
                backup_dir = self.run_dir / "backup"
                backup_dir.mkdir(exist_ok=True)
                with open(backup_dir / "summary.json", "w") as f:
                    f.write(summary.model_dump_json(indent=2))
            except Exception:
                pass  # Fail silently
    
    def close(self) -> None:
        """Close all handlers and files."""
        for handler in self.handlers.values():
            handler.close()
        
        for file_handle in self.specialized_files.values():
            if file_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass


class PerformanceTracker:
    """
    Tracks performance metrics and logs them periodically.
    """
    
    def __init__(self, run_handler: RunDirectoryHandler):
        self.run_handler = run_handler
        self.metrics: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timing(self, component_name: str) -> None:
        """Start timing a component."""
        self.start_times[component_name] = time.time()
    
    def end_timing(self, component_name: str, component_type: str = "unknown", 
                   invocation_id: str = "", **kwargs) -> None:
        """End timing and log performance."""
        if component_name in self.start_times:
            duration_ms = (time.time() - self.start_times[component_name]) * 1000
            
            # Track metric
            if component_name not in self.metrics:
                self.metrics[component_name] = []
            self.metrics[component_name].append(duration_ms)
            
            # Create performance log
            perf_log = PerformanceLog(
                invocation_id=invocation_id,
                component_type=component_type,
                component_name=component_name,
                execution_time_ms=duration_ms,
                **kwargs
            )
            
            self.run_handler.log_performance(perf_log)
            
            # Clean up
            del self.start_times[component_name]
    
    def get_average_time(self, component_name: str) -> Optional[float]:
        """Get average execution time for a component."""
        if component_name in self.metrics and self.metrics[component_name]:
            return sum(self.metrics[component_name]) / len(self.metrics[component_name])
        return None
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all tracked metrics."""
        summary = {}
        for component, times in self.metrics.items():
            if times:
                summary[component] = {
                    "average_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "total_calls": len(times),
                    "total_time_ms": sum(times)
                }
        return summary