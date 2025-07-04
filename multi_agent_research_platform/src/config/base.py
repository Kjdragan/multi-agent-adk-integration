"""
Base configuration classes and utilities.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class BaseConfig(BaseSettings):
    """
    Base configuration class with common settings.
    
    All configuration classes should inherit from this to get
    consistent environment variable loading and validation.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
        validate_assignment=True,
    )
    
    # Core environment settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    app_name: str = Field(
        default="multi-agent-research-platform",
        description="Application name"
    )


class DatabaseSettings(BaseModel):
    """Database connection settings."""
    model_config = ConfigDict(extra="forbid")
    
    url: str = Field(description="Database connection URL")
    echo: bool = Field(default=False, description="Enable SQL logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")


class RedisSettings(BaseModel):
    """Redis connection settings."""
    model_config = ConfigDict(extra="forbid")
    
    url: str = Field(description="Redis connection URL")
    decode_responses: bool = Field(default=True, description="Decode responses to strings")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    socket_connect_timeout: int = Field(default=5, description="Socket connect timeout")
    socket_keepalive: bool = Field(default=True, description="Enable socket keepalive")
    socket_keepalive_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Socket keepalive options"
    )


class APIKeySettings(BaseModel):
    """API key configuration."""
    model_config = ConfigDict(extra="forbid")
    
    # Google/Gemini APIs
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API key")
    
    # MCP Server APIs
    perplexity_api_key: Optional[str] = Field(default=None, description="Perplexity API key")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key")
    brave_api_key: Optional[str] = Field(default=None, description="Brave Search API key")
    
    # Additional services
    openweather_api_key: Optional[str] = Field(default=None, description="OpenWeather API key")
    
    def get_available_keys(self) -> Dict[str, str]:
        """Get all non-None API keys."""
        return {
            key: value for key, value in self.model_dump().items()
            if value is not None
        }
    
    def validate_required_keys(self, required: list[str]) -> None:
        """Validate that required API keys are present."""
        available = self.get_available_keys()
        missing = [key for key in required if key not in available]
        
        if missing:
            raise ValueError(f"Missing required API keys: {', '.join(missing)}")


class SecuritySettings(BaseModel):
    """Security configuration."""
    model_config = ConfigDict(extra="forbid")
    
    secret_key: str = Field(description="Secret key for session management")
    enable_auth: bool = Field(default=False, description="Enable authentication")
    auth_provider: str = Field(default="google", description="Authentication provider")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080", "http://localhost:8501"],
        description="CORS allowed origins"
    )
    
    # Rate limiting
    requests_per_minute: int = Field(default=100, description="Requests per minute limit")
    tokens_per_minute: int = Field(default=10000, description="Tokens per minute limit")
    
    # Session configuration
    session_timeout_minutes: int = Field(default=120, description="Session timeout in minutes")
    remember_me_days: int = Field(default=30, description="Remember me duration in days")


class FeatureFlags(BaseModel):
    """Feature flag configuration."""
    model_config = ConfigDict(extra="forbid")
    
    # Interface features
    enable_web_interface: bool = Field(default=True, description="Enable ADK web interface")
    enable_streamlit_interface: bool = Field(default=True, description="Enable Streamlit interface")
    
    # Integration features
    enable_mcp_integration: bool = Field(default=True, description="Enable MCP server integration")
    enable_memory_service: bool = Field(default=True, description="Enable memory service")
    enable_artifact_service: bool = Field(default=True, description="Enable artifact service")
    
    # Agent features
    enable_code_execution: bool = Field(default=True, description="Enable code execution")
    enable_google_search: bool = Field(default=True, description="Enable Google Search")
    enable_vertex_ai_search: bool = Field(default=True, description="Enable Vertex AI Search")
    enable_bigquery_tools: bool = Field(default=True, description="Enable BigQuery tools")
    
    # Performance features
    enable_performance_tracking: bool = Field(default=True, description="Enable performance tracking")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    
    # Development features
    mock_external_apis: bool = Field(default=False, description="Mock external APIs for testing")
    log_requests: bool = Field(default=False, description="Log all requests")
    auto_reload: bool = Field(default=False, description="Auto-reload on file changes")


# Global configuration instance
_config_instance: Optional[BaseConfig] = None

ConfigType = TypeVar('ConfigType', bound=BaseConfig)


def load_config(config_class: Type[ConfigType], 
               config_file: Optional[Path] = None) -> ConfigType:
    """
    Load configuration from environment variables and optional config file.
    
    Args:
        config_class: Configuration class to instantiate
        config_file: Optional path to .env file
        
    Returns:
        Configured instance
    """
    # Set environment file if provided
    if config_file and config_file.exists():
        os.environ.setdefault("PYDANTIC_SETTINGS_ENV_FILE", str(config_file))
    
    # Create and validate configuration
    config = config_class()
    
    # Store global instance if it's a main config
    global _config_instance
    if hasattr(config, 'app_name'):
        _config_instance = config
    
    return config


def get_config() -> Optional[BaseConfig]:
    """Get the global configuration instance."""
    return _config_instance


def validate_environment() -> None:
    """Validate that the current environment is properly configured."""
    if not _config_instance:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    
    # Validate based on environment
    if _config_instance.environment == Environment.PRODUCTION:
        # Production-specific validation
        if _config_instance.debug:
            raise ValueError("Debug mode should not be enabled in production")
    
    elif _config_instance.environment == Environment.DEVELOPMENT:
        # Development-specific validation
        pass  # More lenient in development
    
    # Common validation
    # Add any cross-cutting validation logic here


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def get_data_dir() -> Path:
    """Get the data directory for the application."""
    return get_project_root() / "data"


def get_logs_dir() -> Path:
    """Get the logs directory for the application."""
    return get_project_root() / "logs"


def get_config_dir() -> Path:
    """Get the configuration directory."""
    return get_project_root() / "config"