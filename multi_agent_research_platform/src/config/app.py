"""
Application-level configuration.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator

from .base import BaseConfig, SecuritySettings, FeatureFlags, APIKeySettings

# Handle import context - works both as relative and absolute import
try:
    from ..platform_logging.models import LogConfig
except ImportError:
    from src.platform_logging.models import LogConfig


class ServerConfig(BaseConfig):
    """Server configuration settings."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8080, description="Server port")
    streamlit_port: int = Field(default=8501, description="Streamlit server port")
    
    # Worker settings
    max_workers: int = Field(default=4, description="Maximum worker processes")
    worker_timeout: int = Field(default=300, description="Worker timeout in seconds")
    
    # Request handling
    max_request_size: int = Field(default=100 * 1024 * 1024, description="Max request size in bytes (100MB)")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # Health check settings
    health_check_path: str = Field(default="/health", description="Health check endpoint")
    readiness_check_path: str = Field(default="/ready", description="Readiness check endpoint")
    
    @field_validator('port', 'streamlit_port')
    @classmethod
    def validate_port(cls, v):
        if not 1024 <= v <= 65535:
            raise ValueError('Port must be between 1024 and 65535')
        return v


class PerformanceConfig(BaseConfig):
    """Performance and resource management configuration."""
    
    # Concurrent processing limits
    max_concurrent_agents: int = Field(default=5, description="Maximum concurrent agents")
    max_concurrent_tools: int = Field(default=10, description="Maximum concurrent tools")
    max_parallel_research_tasks: int = Field(default=3, description="Maximum parallel research tasks")
    
    # Timeout settings (seconds)
    agent_timeout: int = Field(default=300, description="Agent execution timeout")
    tool_timeout: int = Field(default=60, description="Tool execution timeout")
    llm_timeout: int = Field(default=30, description="LLM request timeout")
    
    # Memory management
    max_session_memory_mb: int = Field(default=512, description="Maximum session memory in MB")
    max_artifact_size_mb: int = Field(default=100, description="Maximum artifact size in MB")
    
    # Caching
    enable_response_cache: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size_mb: int = Field(default=1024, description="Maximum cache size in MB")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=100, description="Requests per minute limit")
    rate_limit_burst_size: int = Field(default=10, description="Rate limit burst size")


class SecurityConfig(BaseConfig):
    """Security configuration."""
    
    # Authentication
    enable_authentication: bool = Field(default=False, description="Enable authentication")
    auth_provider: str = Field(default="google", description="Authentication provider")
    auth_secret_key: str = Field(default="dev-secret-key", description="Authentication secret key")
    
    # Authorization
    enable_rbac: bool = Field(default=False, description="Enable role-based access control")
    default_user_role: str = Field(default="user", description="Default user role")
    admin_users: List[str] = Field(default_factory=list, description="Admin user emails")
    
    # CORS
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8080", 
            "http://localhost:8501"
        ],
        description="CORS allowed origins"
    )
    cors_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="CORS allowed methods"
    )
    cors_headers: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed headers"
    )
    
    # API Security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    require_api_key: bool = Field(default=False, description="Require API key for requests")
    allowed_api_keys: List[str] = Field(default_factory=list, description="Allowed API keys")
    
    # Session Security
    session_cookie_secure: bool = Field(default=False, description="Secure session cookies")
    session_cookie_httponly: bool = Field(default=True, description="HTTP-only session cookies")
    session_cookie_samesite: str = Field(default="lax", description="SameSite cookie policy")
    session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
    
    # Content Security
    max_upload_size_mb: int = Field(default=50, description="Maximum upload size in MB")
    allowed_file_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".json", ".csv"],
        description="Allowed file extensions for uploads"
    )
    
    @field_validator('auth_provider')
    @classmethod
    def validate_auth_provider(cls, v):
        allowed_providers = ["google", "oauth2", "custom", "none"]
        if v not in allowed_providers:
            raise ValueError(f'Auth provider must be one of: {allowed_providers}')
        return v


class AppConfig(BaseConfig):
    """Main application configuration."""
    
    # Basic app info
    app_name: str = Field(default="multi-agent-research-platform", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    app_description: str = Field(
        default="Multi-Agent Research Platform with Google ADK",
        description="Application description"
    )
    
    # Directories
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    temp_dir: Path = Field(default=Path("temp"), description="Temporary files directory")
    
    # API Keys
    api_keys: APIKeySettings = Field(default_factory=APIKeySettings, description="API key configuration")
    
    # Component configurations
    server: ServerConfig = Field(default_factory=ServerConfig, description="Server configuration")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    logging: LogConfig = Field(default_factory=LogConfig, description="Logging configuration")
    features: FeatureFlags = Field(default_factory=FeatureFlags, description="Feature flags")
    
    # Google Cloud settings
    google_cloud_project: Optional[str] = Field(default=None, description="Google Cloud project ID")
    google_cloud_location: str = Field(default="us-central1", description="Google Cloud location")
    google_genai_use_vertexai: bool = Field(default=True, description="Use Vertex AI for Gemini")
    
    # Vertex AI specific settings
    vertex_ai_search_data_store: Optional[str] = Field(
        default=None, 
        description="Vertex AI Search data store ID"
    )
    vertex_ai_rag_corpus: Optional[str] = Field(
        default=None,
        description="Vertex AI RAG corpus for memory service"
    )
    
    # BigQuery settings
    bigquery_dataset: Optional[str] = Field(default=None, description="BigQuery dataset name")
    bigquery_table_prefix: str = Field(default="research_", description="BigQuery table prefix")
    
    # Development settings
    development_mode: bool = Field(default=True, description="Enable development mode features")
    reload_on_change: bool = Field(default=False, description="Reload on file changes")
    mock_external_services: bool = Field(default=False, description="Mock external services")
    
    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True) 
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Update logging config with our logs directory
        self.logging.log_dir = self.logs_dir
    
    def validate_cloud_config(self) -> None:
        """Validate cloud-specific configuration."""
        if self.google_genai_use_vertexai:
            if not self.google_cloud_project:
                raise ValueError("google_cloud_project is required when using Vertex AI")
    
    def get_database_url(self, service: str = "session") -> str:
        """Get database URL for a specific service."""
        if self.environment.value == "production":
            # In production, use environment-specific URLs
            return f"postgresql://user:pass@host:port/{service}_db"
        else:
            # In development, use SQLite
            return f"sqlite:///{self.data_dir}/{service}.db"
    
    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL if caching is enabled."""
        if self.performance.enable_response_cache:
            if self.environment.value == "production":
                return "redis://redis:6379/0"
            else:
                return "redis://localhost:6379/0"
        return None
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.value == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.value == "production"
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration for FastAPI."""
        if not self.security.cors_enabled:
            return {}
        
        return {
            "allow_origins": self.security.cors_origins,
            "allow_methods": self.security.cors_methods,
            "allow_headers": self.security.cors_headers,
            "allow_credentials": True,
        }