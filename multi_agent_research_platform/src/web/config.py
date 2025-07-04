"""
Web Interface Configuration

Configuration classes for the ADK web interface, debugging capabilities,
and monitoring dashboards.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from ..config.base import BaseConfig


class WebInterfaceMode(str, Enum):
    """Web interface operating modes."""
    DEVELOPMENT = "development"
    DEBUG = "debug"
    PRODUCTION = "production"
    MONITORING = "monitoring"


class LogLevel(str, Enum):
    """Log levels for web interface."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class WebConfig(BaseConfig):
    """Main web interface configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from .env
        validate_assignment=True,
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8081, description="Server port")
    reload: bool = Field(default=True, description="Auto-reload on changes")
    
    # Interface mode
    mode: WebInterfaceMode = Field(default=WebInterfaceMode.DEBUG, description="Interface mode")
    
    # Security settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: str = Field(default="*", description="CORS origins (comma-separated)")
    api_key_required: bool = Field(default=False, description="Require API key")
    api_key: Optional[str] = Field(default=None, description="API key")
    
    # Static files
    static_files_enabled: bool = Field(default=True, description="Enable static files")
    static_files_path: str = Field(default="/static", description="Static files path")
    templates_path: str = Field(default="templates", description="Templates path")
    
    # Session settings
    session_secret_key: str = Field(default="multi-agent-research-platform-secret-key", description="Session secret")
    session_timeout_minutes: int = Field(default=30, description="Session timeout")
    
    # ADK integration
    adk_web_enabled: bool = Field(default=True, description="Enable ADK web interface")
    adk_web_port: Optional[int] = Field(default=None, description="ADK web port")
    adk_debug_mode: bool = Field(default=True, description="ADK debug mode")
    
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]
        return self.cors_origins if isinstance(self.cors_origins, list) else [self.cors_origins]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "mode": self.mode.value,
            "enable_cors": self.enable_cors,
            "cors_origins": self.get_cors_origins_list(),
            "api_key_required": self.api_key_required,
            "static_files_enabled": self.static_files_enabled,
            "static_files_path": self.static_files_path,
            "templates_path": self.templates_path,
            "session_timeout_minutes": self.session_timeout_minutes,
            "adk_web_enabled": self.adk_web_enabled,
            "adk_web_port": self.adk_web_port,
            "adk_debug_mode": self.adk_debug_mode,
        }


class DebugConfig(BaseConfig):
    """Debug interface configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )
    
    # Debug features
    enable_step_debugging: bool = Field(default=True, description="Enable step debugging")
    enable_agent_inspection: bool = Field(default=True, description="Enable agent inspection")
    enable_live_logs: bool = Field(default=True, description="Enable live logs")
    enable_performance_profiling: bool = Field(default=True, description="Enable performance profiling")
    
    # Log settings
    log_level: LogLevel = Field(default=LogLevel.DEBUG, description="Log level")
    max_log_entries: int = Field(default=1000, description="Max log entries")
    log_retention_hours: int = Field(default=24, description="Log retention hours")
    
    # Agent debugging
    agent_state_tracking: bool = Field(default=True, description="Agent state tracking")
    agent_memory_inspection: bool = Field(default=True, description="Agent memory inspection")
    agent_tool_monitoring: bool = Field(default=True, description="Agent tool monitoring")
    
    # Orchestration debugging
    task_flow_visualization: bool = Field(default=True, description="Task flow visualization")
    agent_interaction_tracking: bool = Field(default=True, description="Agent interaction tracking")
    consensus_building_details: bool = Field(default=True, description="Consensus building details")
    
    # Performance debugging
    execution_time_tracking: bool = Field(default=True, description="Execution time tracking")
    memory_usage_monitoring: bool = Field(default=True, description="Memory usage monitoring")
    api_call_monitoring: bool = Field(default=True, description="API call monitoring")
    
    # Error handling
    detailed_error_reporting: bool = Field(default=True, description="Detailed error reporting")
    error_context_capture: bool = Field(default=True, description="Error context capture")
    automatic_error_classification: bool = Field(default=True, description="Automatic error classification")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enable_step_debugging": self.enable_step_debugging,
            "enable_agent_inspection": self.enable_agent_inspection,
            "enable_live_logs": self.enable_live_logs,
            "enable_performance_profiling": self.enable_performance_profiling,
            "log_level": self.log_level.value,
            "max_log_entries": self.max_log_entries,
            "log_retention_hours": self.log_retention_hours,
            "agent_state_tracking": self.agent_state_tracking,
            "agent_memory_inspection": self.agent_memory_inspection,
            "agent_tool_monitoring": self.agent_tool_monitoring,
            "task_flow_visualization": self.task_flow_visualization,
            "agent_interaction_tracking": self.agent_interaction_tracking,
            "consensus_building_details": self.consensus_building_details,
            "execution_time_tracking": self.execution_time_tracking,
            "memory_usage_monitoring": self.memory_usage_monitoring,
            "api_call_monitoring": self.api_call_monitoring,
            "detailed_error_reporting": self.detailed_error_reporting,
            "error_context_capture": self.error_context_capture,
            "automatic_error_classification": self.automatic_error_classification,
        }


class MonitoringConfig(BaseConfig):
    """Monitoring interface configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )
    
    # Real-time monitoring
    enable_real_time_updates: bool = Field(default=True, description="Enable real-time updates")
    update_interval_seconds: int = Field(default=5, description="Update interval seconds")
    websocket_enabled: bool = Field(default=True, description="WebSocket enabled")
    
    # Metrics collection
    collect_performance_metrics: bool = Field(default=True, description="Collect performance metrics")
    collect_usage_metrics: bool = Field(default=True, description="Collect usage metrics")
    collect_error_metrics: bool = Field(default=True, description="Collect error metrics")
    metrics_retention_days: int = Field(default=7, description="Metrics retention days")
    
    # Dashboard settings
    default_dashboard: str = Field(default="overview", description="Default dashboard")
    auto_refresh_dashboards: bool = Field(default=True, description="Auto refresh dashboards")
    dashboard_themes: List[str] = Field(default=["light", "dark"], description="Dashboard themes")
    default_theme: str = Field(default="light", description="Default theme")
    
    # Alerts and notifications
    enable_alerts: bool = Field(default=True, description="Enable alerts")
    alert_thresholds: Dict[str, float] = Field(default={
        "error_rate": 0.1,  # 10% error rate
        "response_time_ms": 5000,  # 5 second response time
        "memory_usage_mb": 1024,  # 1GB memory usage
        "agent_failure_rate": 0.05,  # 5% agent failure rate
    }, description="Alert thresholds")
    
    # Export and reporting
    enable_data_export: bool = Field(default=True, description="Enable data export")
    export_formats: List[str] = Field(default=["json", "csv", "excel"], description="Export formats")
    automated_reports: bool = Field(default=False, description="Automated reports")
    report_schedule: str = Field(default="daily", description="Report schedule")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enable_real_time_updates": self.enable_real_time_updates,
            "update_interval_seconds": self.update_interval_seconds,
            "websocket_enabled": self.websocket_enabled,
            "collect_performance_metrics": self.collect_performance_metrics,
            "collect_usage_metrics": self.collect_usage_metrics,
            "collect_error_metrics": self.collect_error_metrics,
            "metrics_retention_days": self.metrics_retention_days,
            "default_dashboard": self.default_dashboard,
            "auto_refresh_dashboards": self.auto_refresh_dashboards,
            "dashboard_themes": self.dashboard_themes,
            "default_theme": self.default_theme,
            "enable_alerts": self.enable_alerts,
            "alert_thresholds": self.alert_thresholds,
            "enable_data_export": self.enable_data_export,
            "export_formats": self.export_formats,
            "automated_reports": self.automated_reports,
            "report_schedule": self.report_schedule,
        }


class APIConfig(BaseConfig):
    """API configuration for web interface."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )
    
    # API settings
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    enable_openapi: bool = Field(default=True, description="Enable OpenAPI")
    openapi_title: str = Field(default="Multi-Agent Research Platform API", description="OpenAPI title")
    openapi_version: str = Field(default="1.0.0", description="OpenAPI version")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute")
    rate_limit_per_hour: int = Field(default=1000, description="Rate limit per hour")
    
    # Request/Response settings
    max_request_size_mb: int = Field(default=10, description="Max request size MB")
    request_timeout_seconds: int = Field(default=30, description="Request timeout seconds")
    response_compression: bool = Field(default=True, description="Response compression")
    
    # Authentication
    enable_authentication: bool = Field(default=False, description="Enable authentication")
    auth_type: str = Field(default="api_key", description="Auth type")  # api_key, oauth, jwt
    token_expiry_hours: int = Field(default=24, description="Token expiry hours")
    
    # Validation
    strict_validation: bool = Field(default=True, description="Strict validation")
    validate_responses: bool = Field(default=True, description="Validate responses")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "api_prefix": self.api_prefix,
            "enable_openapi": self.enable_openapi,
            "openapi_title": self.openapi_title,
            "openapi_version": self.openapi_version,
            "enable_rate_limiting": self.enable_rate_limiting,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "max_request_size_mb": self.max_request_size_mb,
            "request_timeout_seconds": self.request_timeout_seconds,
            "response_compression": self.response_compression,
            "enable_authentication": self.enable_authentication,
            "auth_type": self.auth_type,
            "token_expiry_hours": self.token_expiry_hours,
            "strict_validation": self.strict_validation,
            "validate_responses": self.validate_responses,
        }


class WebSocketConfig(BaseConfig):
    """WebSocket configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )
    
    # WebSocket settings
    enabled: bool = Field(default=True, description="WebSocket enabled")
    path: str = Field(default="/ws", description="WebSocket path")
    max_connections: int = Field(default=100, description="Max connections")
    
    # Message settings
    max_message_size: int = Field(default=1024 * 1024, description="Max message size")  # 1MB
    ping_interval_seconds: int = Field(default=30, description="Ping interval seconds")
    ping_timeout_seconds: int = Field(default=10, description="Ping timeout seconds")
    
    # Event types
    supported_events: List[str] = Field(default=[
        "agent_status_update",
        "task_progress_update", 
        "orchestration_event",
        "performance_metric_update",
        "error_notification",
        "log_entry",
        "debug_event",
    ], description="Supported events")
    
    # Broadcasting
    enable_broadcasting: bool = Field(default=True, description="Enable broadcasting")
    channel_separation: bool = Field(default=True, description="Channel separation")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "path": self.path,
            "max_connections": self.max_connections,
            "max_message_size": self.max_message_size,
            "ping_interval_seconds": self.ping_interval_seconds,
            "ping_timeout_seconds": self.ping_timeout_seconds,
            "supported_events": self.supported_events,
            "enable_broadcasting": self.enable_broadcasting,
            "channel_separation": self.channel_separation,
        }


class WebConfigFactory:
    """Factory for creating web configuration instances."""
    
    @classmethod
    def create_development_config(cls) -> Dict[str, Any]:
        """Create configuration for development environment."""
        return {
            "web": WebConfig(
                reload=True,
                mode=WebInterfaceMode.DEVELOPMENT,
                adk_debug_mode=True,
            ),
            "debug": DebugConfig(
                enable_step_debugging=True,
                enable_agent_inspection=True,
                enable_live_logs=True,
                log_level=LogLevel.DEBUG,
            ),
            "monitoring": MonitoringConfig(
                enable_real_time_updates=True,
                update_interval_seconds=2,  # More frequent updates
                enable_alerts=True,
            ),
            "api": APIConfig(
                enable_openapi=True,
                enable_rate_limiting=False,  # Disabled for development
            ),
            "websocket": WebSocketConfig(
                enabled=True,
                ping_interval_seconds=15,  # More frequent pings
            ),
        }
    
    @classmethod
    def create_debug_config(cls) -> Dict[str, Any]:
        """Create configuration optimized for debugging."""
        return {
            "web": WebConfig(
                mode=WebInterfaceMode.DEBUG,
                adk_debug_mode=True,
            ),
            "debug": DebugConfig(
                enable_step_debugging=True,
                enable_agent_inspection=True,
                enable_performance_profiling=True,
                log_level=LogLevel.DEBUG,
                detailed_error_reporting=True,
            ),
            "monitoring": MonitoringConfig(
                enable_real_time_updates=True,
                collect_performance_metrics=True,
                collect_error_metrics=True,
            ),
            "api": APIConfig(
                enable_openapi=True,
                strict_validation=True,
            ),
            "websocket": WebSocketConfig(
                enabled=True,
                max_connections=50,
            ),
        }
    
    @classmethod
    def create_production_config(cls) -> Dict[str, Any]:
        """Create configuration for production environment."""
        return {
            "web": WebConfig(
                reload=False,
                mode=WebInterfaceMode.PRODUCTION,
                api_key_required=True,
                adk_debug_mode=False,
            ),
            "debug": DebugConfig(
                enable_step_debugging=False,
                enable_agent_inspection=False,
                log_level=LogLevel.INFO,
                max_log_entries=500,
            ),
            "monitoring": MonitoringConfig(
                enable_real_time_updates=True,
                update_interval_seconds=10,
                enable_alerts=True,
                automated_reports=True,
            ),
            "api": APIConfig(
                enable_authentication=True,
                enable_rate_limiting=True,
                strict_validation=True,
            ),
            "websocket": WebSocketConfig(
                enabled=True,
                max_connections=200,
            ),
        }
    
    @classmethod
    def create_monitoring_config(cls) -> Dict[str, Any]:
        """Create configuration optimized for monitoring."""
        return {
            "web": WebConfig(
                mode=WebInterfaceMode.MONITORING,
                adk_debug_mode=False,
            ),
            "debug": DebugConfig(
                enable_step_debugging=False,
                enable_performance_profiling=True,
                log_level=LogLevel.INFO,
            ),
            "monitoring": MonitoringConfig(
                enable_real_time_updates=True,
                collect_performance_metrics=True,
                collect_usage_metrics=True,
                collect_error_metrics=True,
                enable_alerts=True,
                enable_data_export=True,
            ),
            "api": APIConfig(
                enable_rate_limiting=True,
            ),
            "websocket": WebSocketConfig(
                enabled=True,
            ),
        }


def get_config_for_environment(environment: str = "development") -> Dict[str, Any]:
    """
    Get web configuration for specific environment.
    
    Args:
        environment: Environment name (development, debug, production, monitoring)
        
    Returns:
        Configuration dictionary
    """
    if environment == "development":
        return WebConfigFactory.create_development_config()
    elif environment == "debug":
        return WebConfigFactory.create_debug_config()
    elif environment == "production":
        return WebConfigFactory.create_production_config()
    elif environment == "monitoring":
        return WebConfigFactory.create_monitoring_config()
    else:
        raise ValueError(f"Unknown environment: {environment}")