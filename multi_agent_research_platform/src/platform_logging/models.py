"""
Logging data models and configuration.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class LogLevel(str, Enum):
    """Log levels supported by the platform."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    JSON = "json"
    PLAIN = "plain"
    LLM_READY = "llm_ready"


class LogConfig(BaseModel):
    """Logging system configuration."""
    model_config = ConfigDict(extra="forbid")
    
    # Basic configuration
    log_dir: Path = Field(default=Path("logs"), description="Base logging directory")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Minimum log level")
    log_format: LogFormat = Field(default=LogFormat.JSON, description="Log output format")
    
    # Retention and organization
    retention_days: int = Field(default=30, description="Log retention period in days")
    max_log_size_mb: int = Field(default=100, description="Maximum log file size in MB")
    
    # Feature flags
    enable_performance_tracking: bool = Field(default=True, description="Enable performance metrics")
    enable_structured_logging: bool = Field(default=True, description="Enable structured log format")
    enable_debug_logging: bool = Field(default=True, description="Enable debug level logging")
    
    # Failure safety
    backup_log_handlers: bool = Field(default=True, description="Enable backup log handlers")
    fail_on_log_error: bool = Field(default=False, description="Fail application on log errors")
    
    # LLM integration
    create_llm_summaries: bool = Field(default=True, description="Create LLM-ready summaries")
    summary_frequency: int = Field(default=100, description="Events per summary generation")


class RunContext(BaseModel):
    """Context information for a specific run/invocation."""
    model_config = ConfigDict(extra="forbid")
    
    # Identifiers
    run_id: str = Field(description="Unique run identifier")
    invocation_id: str = Field(description="ADK invocation identifier")
    session_id: str = Field(description="Session identifier")
    user_id: str = Field(description="User identifier")
    
    # Timing
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(default=None)
    
    # Context
    initial_query: Optional[str] = Field(default=None, description="Initial user query")
    agent_names: List[str] = Field(default_factory=list, description="Agents involved in run")
    tool_names: List[str] = Field(default_factory=list, description="Tools used in run")
    
    # Status
    status: str = Field(default="running", description="Run status: running, completed, failed")
    error_count: int = Field(default=0, description="Number of errors encountered")
    warning_count: int = Field(default=0, description="Number of warnings encountered")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional run metadata")


class EventLog(BaseModel):
    """Log entry for ADK events."""
    model_config = ConfigDict(extra="forbid")
    
    # Basic info
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str = Field(description="Unique event identifier")
    invocation_id: str = Field(description="Associated invocation ID")
    
    # Event details
    author: str = Field(description="Event author (user, agent name, etc.)")
    event_type: str = Field(description="Type of event (text, function_call, etc.)")
    partial: bool = Field(default=False, description="Whether event is partial/streaming")
    
    # Content
    content_summary: Optional[str] = Field(default=None, description="Summary of event content")
    function_calls: List[str] = Field(default_factory=list, description="Function calls in event")
    function_responses: List[str] = Field(default_factory=list, description="Function responses")
    
    # Actions
    state_changes: Dict[str, Any] = Field(default_factory=dict, description="State changes")
    artifacts_saved: List[str] = Field(default_factory=list, description="Artifacts saved")
    control_signals: Dict[str, Any] = Field(default_factory=dict, description="Control flow signals")
    
    # Performance
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")


class StateChangeLog(BaseModel):
    """Log entry for state changes."""
    model_config = ConfigDict(extra="forbid")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    invocation_id: str = Field(description="Associated invocation ID")
    event_id: str = Field(description="Associated event ID")
    
    # Change details
    state_key: str = Field(description="State key that changed")
    old_value: Optional[Any] = Field(default=None, description="Previous value")
    new_value: Any = Field(description="New value")
    change_type: str = Field(description="Type of change: create, update, delete")
    
    # Context
    agent_name: str = Field(description="Agent that made the change")
    context_type: str = Field(description="Context type: session, user, app, temp")


class ToolCallLog(BaseModel):
    """Log entry for tool execution."""
    model_config = ConfigDict(extra="forbid")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    invocation_id: str = Field(description="Associated invocation ID")
    function_call_id: str = Field(description="Function call identifier")
    
    # Tool details
    tool_name: str = Field(description="Name of the tool")
    agent_name: str = Field(description="Agent that called the tool")
    
    # Execution
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(default=None)
    status: str = Field(default="started", description="Execution status")
    
    # Input/Output
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Tool result")
    error_message: Optional[str] = Field(default=None, description="Error if failed")
    
    # Performance
    execution_time_ms: Optional[float] = Field(default=None)
    memory_usage_mb: Optional[float] = Field(default=None)
    
    # Authentication
    required_auth: bool = Field(default=False, description="Whether tool required authentication")
    auth_success: bool = Field(default=True, description="Whether authentication succeeded")


class PerformanceLog(BaseModel):
    """Log entry for performance metrics."""
    model_config = ConfigDict(extra="forbid")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    invocation_id: str = Field(description="Associated invocation ID")
    
    # Component performance
    component_type: str = Field(description="Type: agent, tool, service")
    component_name: str = Field(description="Name of component")
    
    # Metrics
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage")
    cpu_usage_percent: Optional[float] = Field(default=None, description="CPU usage")
    
    # Throughput
    tokens_per_second: Optional[float] = Field(default=None, description="Token processing rate")
    requests_per_second: Optional[float] = Field(default=None, description="Request processing rate")
    
    # Quality metrics
    success_rate: Optional[float] = Field(default=None, description="Success rate percentage")
    error_rate: Optional[float] = Field(default=None, description="Error rate percentage")
    
    # Context
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")


class LogSummary(BaseModel):
    """Summary of a completed run for LLM analysis."""
    model_config = ConfigDict(extra="forbid")
    
    # Run info
    run_context: RunContext = Field(description="Run context information")
    
    # Statistics
    total_events: int = Field(description="Total number of events")
    total_tool_calls: int = Field(description="Total number of tool calls")
    total_state_changes: int = Field(description="Total state changes")
    
    # Performance
    total_duration_ms: float = Field(description="Total run duration")
    average_event_processing_ms: float = Field(description="Average event processing time")
    total_tokens_used: int = Field(description="Total tokens consumed")
    
    # Quality
    success_rate: float = Field(description="Overall success rate")
    error_summary: List[str] = Field(default_factory=list, description="Error summaries")
    warning_summary: List[str] = Field(default_factory=list, description="Warning summaries")
    
    # Agent effectiveness
    agent_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Performance metrics by agent"
    )
    
    # Tool effectiveness  
    tool_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Performance metrics by tool"
    )
    
    # Insights for LLM
    key_insights: List[str] = Field(default_factory=list, description="Key insights from run")
    improvement_suggestions: List[str] = Field(
        default_factory=list, 
        description="Suggestions for improvement"
    )
    notable_patterns: List[str] = Field(
        default_factory=list,
        description="Notable patterns observed"
    )