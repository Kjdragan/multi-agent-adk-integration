"""
Log formatters for different output formats.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from .models import LogLevel, EventLog, StateChangeLog, ToolCallLog, PerformanceLog


class LLMReadyFormatter(logging.Formatter):
    """
    Formatter that creates LLM-friendly log output.
    
    Designed to be easily parseable by LLMs for analysis and debugging.
    """
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for LLM consumption."""
        
        # Basic structure
        formatted = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": getattr(record, 'component', record.name),
            "message": record.getMessage(),
        }
        
        # Add invocation context if available
        if hasattr(record, 'invocation_id'):
            formatted["invocation_id"] = record.invocation_id
        
        if hasattr(record, 'agent_name'):
            formatted["agent"] = record.agent_name
            
        if hasattr(record, 'tool_name'):
            formatted["tool"] = record.tool_name
        
        # Add structured data if available
        if hasattr(record, 'data') and record.data:
            formatted["data"] = record.data
        
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            formatted["duration_ms"] = record.duration_ms
            
        if hasattr(record, 'tokens_used'):
            formatted["tokens_used"] = record.tokens_used
        
        # Add error context if this is an error
        if record.levelno >= logging.ERROR:
            if hasattr(record, 'exc_info') and record.exc_info:
                formatted["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
                formatted["error_details"] = str(record.exc_info[1]) if record.exc_info[1] else None
        
        # Format as readable text for LLM
        text_parts = [
            f"[{formatted['timestamp']}] {formatted['level']}: {formatted['message']}"
        ]
        
        if 'component' in formatted and formatted['component'] != 'root':
            text_parts.append(f"Component: {formatted['component']}")
        
        if 'invocation_id' in formatted:
            text_parts.append(f"Invocation: {formatted['invocation_id'][:8]}...")
        
        if 'agent' in formatted:
            text_parts.append(f"Agent: {formatted['agent']}")
            
        if 'tool' in formatted:
            text_parts.append(f"Tool: {formatted['tool']}")
        
        if 'duration_ms' in formatted:
            text_parts.append(f"Duration: {formatted['duration_ms']:.2f}ms")
            
        if 'tokens_used' in formatted:
            text_parts.append(f"Tokens: {formatted['tokens_used']}")
        
        if 'data' in formatted:
            text_parts.append(f"Data: {json.dumps(formatted['data'], indent=2)}")
        
        if 'error_type' in formatted:
            text_parts.append(f"Error: {formatted['error_type']} - {formatted['error_details']}")
        
        return " | ".join(text_parts)


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs.
    
    Suitable for log aggregation systems and programmatic analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'stack_info']:
                log_entry[key] = value
        
        # Handle exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))


class PlainFormatter(logging.Formatter):
    """
    Simple plain text formatter for human-readable output.
    """
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class EventLogFormatter:
    """Formatter specifically for ADK events."""
    
    @staticmethod
    def format_event_log(event_log: EventLog) -> Dict[str, Any]:
        """Format an EventLog for output."""
        return {
            "type": "event",
            "timestamp": event_log.timestamp.isoformat(),
            "event_id": event_log.event_id,
            "invocation_id": event_log.invocation_id,
            "author": event_log.author,
            "event_type": event_log.event_type,
            "partial": event_log.partial,
            "content_summary": event_log.content_summary,
            "function_calls": event_log.function_calls,
            "function_responses": event_log.function_responses,
            "state_changes": event_log.state_changes,
            "artifacts_saved": event_log.artifacts_saved,
            "control_signals": event_log.control_signals,
            "processing_time_ms": event_log.processing_time_ms,
            "tokens_used": event_log.tokens_used,
        }
    
    @staticmethod
    def format_state_change_log(state_log: StateChangeLog) -> Dict[str, Any]:
        """Format a StateChangeLog for output."""
        return {
            "type": "state_change",
            "timestamp": state_log.timestamp.isoformat(),
            "invocation_id": state_log.invocation_id,
            "event_id": state_log.event_id,
            "state_key": state_log.state_key,
            "old_value": state_log.old_value,
            "new_value": state_log.new_value,
            "change_type": state_log.change_type,
            "agent_name": state_log.agent_name,
            "context_type": state_log.context_type,
        }
    
    @staticmethod
    def format_tool_call_log(tool_log: ToolCallLog) -> Dict[str, Any]:
        """Format a ToolCallLog for output."""
        return {
            "type": "tool_call",
            "timestamp": tool_log.timestamp.isoformat(),
            "invocation_id": tool_log.invocation_id,
            "function_call_id": tool_log.function_call_id,
            "tool_name": tool_log.tool_name,
            "agent_name": tool_log.agent_name,
            "start_time": tool_log.start_time.isoformat(),
            "end_time": tool_log.end_time.isoformat() if tool_log.end_time else None,
            "status": tool_log.status,
            "arguments": tool_log.arguments,
            "result": tool_log.result,
            "error_message": tool_log.error_message,
            "execution_time_ms": tool_log.execution_time_ms,
            "memory_usage_mb": tool_log.memory_usage_mb,
            "required_auth": tool_log.required_auth,
            "auth_success": tool_log.auth_success,
        }
    
    @staticmethod
    def format_performance_log(perf_log: PerformanceLog) -> Dict[str, Any]:
        """Format a PerformanceLog for output."""
        return {
            "type": "performance",
            "timestamp": perf_log.timestamp.isoformat(),
            "invocation_id": perf_log.invocation_id,
            "component_type": perf_log.component_type,
            "component_name": perf_log.component_name,
            "execution_time_ms": perf_log.execution_time_ms,
            "memory_usage_mb": perf_log.memory_usage_mb,
            "cpu_usage_percent": perf_log.cpu_usage_percent,
            "tokens_per_second": perf_log.tokens_per_second,
            "requests_per_second": perf_log.requests_per_second,
            "success_rate": perf_log.success_rate,
            "error_rate": perf_log.error_rate,
            "metadata": perf_log.metadata,
        }