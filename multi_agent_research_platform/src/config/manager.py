"""
Centralized Configuration Manager

Provides consistent configuration loading across the entire platform.
Eliminates inconsistent patterns and ensures proper validation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import lru_cache
import logging

from .base import BaseConfig, Environment
from .app import AppConfig
from .services import ServicesConfig
from .agents import AgentConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigurationManager:
    """
    Centralized configuration manager ensuring consistent loading patterns.
    
    This manager:
    - Loads configuration from .env files consistently
    - Validates all required configuration at startup
    - Provides type-safe access to configuration values
    - Handles environment-specific overrides
    - Prevents direct os.getenv usage anti-patterns
    """
    
    def __init__(self):
        self._configs: Dict[str, BaseConfig] = {}
        self._validated = False
        self._logger = logging.getLogger("ConfigManager")
        
    def initialize(self, env_file_path: Optional[Path] = None) -> None:
        """
        Initialize all configurations with validation.
        
        Args:
            env_file_path: Optional path to .env file. If None, uses project root.
        """
        try:
            # Set environment file path
            if env_file_path is None:
                project_root = Path(__file__).parent.parent.parent
                env_file_path = project_root / ".env"
            
            # Ensure .env file exists
            if not env_file_path.exists():
                self._logger.warning(f"No .env file found at {env_file_path}")
                self._logger.warning("Using environment variables and defaults only")
            
            # Load all configurations
            self._load_configurations()
            
            # Validate all required settings
            self._validate_configurations()
            
            self._validated = True
            self._logger.info("Configuration loaded and validated successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {e}")
    
    def _load_configurations(self) -> None:
        """Load all configuration classes."""
        self._configs = {
            'app': AppConfig(),
            'services': ServicesConfig(),
            'agents': AgentConfig()
        }
    
    def _validate_configurations(self) -> None:
        """Validate all configurations and check for required settings."""
        validation_errors = []
        
        # Validate Google AI API configuration
        app_config = self._configs['app']
        api_errors = self._validate_google_api_configuration(app_config)
        validation_errors.extend(api_errors)
        
        # Validate environment-specific requirements
        env_errors = self._validate_environment_configuration(app_config)
        validation_errors.extend(env_errors)
        
        # Validate service configurations
        services_config = self._configs['services']
        service_errors = self._validate_service_configuration(services_config)
        validation_errors.extend(service_errors)
        
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ConfigurationError(error_msg)
    
    def _validate_google_api_configuration(self, app_config) -> List[str]:
        """Validate Google AI API configuration thoroughly."""
        errors = []
        
        # Check API key configuration
        api_key = self.get_google_api_key()
        vertex_ai_enabled = app_config.google_genai_use_vertexai
        
        if not api_key and not vertex_ai_enabled:
            errors.append(
                "GOOGLE_API_KEY is required when GOOGLE_GENAI_USE_VERTEXAI=False"
            )
        
        # Validate API key format if provided
        if api_key:
            if not api_key.startswith("AIza") or len(api_key) < 35:
                errors.append(
                    "GOOGLE_API_KEY appears to be invalid (should start with 'AIza' and be 39+ characters)"
                )
        
        # Validate Vertex AI configuration if enabled
        if vertex_ai_enabled:
            vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT")
            if not vertex_project:
                errors.append(
                    "GOOGLE_CLOUD_PROJECT is required when GOOGLE_GENAI_USE_VERTEXAI=True"
                )
            
            # Check for ADC authentication in production
            if app_config.environment == Environment.PRODUCTION:
                google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if not google_creds:
                    self._logger.warning(
                        "GOOGLE_APPLICATION_CREDENTIALS not set in production. "
                        "Ensure Application Default Credentials are configured."
                    )
        
        return errors
    
    def _validate_environment_configuration(self, app_config) -> List[str]:
        """Validate environment-specific configuration."""
        errors = []
        
        if app_config.environment == Environment.PRODUCTION:
            if app_config.debug:
                errors.append("DEBUG must be False in production environment")
            
            if not app_config.session_secret_key or app_config.session_secret_key == "your_secret_key_here_change_in_production":
                errors.append("SESSION_SECRET_KEY must be set in production")
            
            # Check log level in production
            if app_config.log_level == "DEBUG":
                self._logger.warning("DEBUG log level not recommended in production")
        
        return errors
    
    def _validate_service_configuration(self, services_config) -> List[str]:
        """Validate service configuration."""
        errors = []
        
        # Validate memory service configuration
        if services_config.memory.service_type == "database":
            if not services_config.memory.database_url:
                errors.append("DATABASE_URL required for database memory service")
        
        if services_config.memory.service_type == "vertex_ai_rag":
            if not services_config.memory.vertex_ai_project:
                errors.append("VERTEX_AI_PROJECT required for Vertex AI RAG memory service")
            if not services_config.memory.rag_corpus_name:
                errors.append("RAG_CORPUS_NAME required for Vertex AI RAG memory service")
        
        # Validate session service configuration
        if services_config.session.service_type in ["database", "vertex_ai"]:
            if not services_config.session.database_url:
                errors.append("DATABASE_URL required for database/vertex_ai session service")
        
        # Validate artifact service configuration
        if services_config.artifact.service_type == "gcs":
            if not services_config.artifact.gcs_bucket_name:
                errors.append("GCS_BUCKET_NAME required for GCS artifact service")
            if not services_config.artifact.gcs_project:
                errors.append("GCS_PROJECT required for GCS artifact service")
        
        elif services_config.artifact.service_type == "s3":
            if not services_config.artifact.s3_bucket_name:
                errors.append("S3_BUCKET_NAME required for S3 artifact service")
        
        return errors
    
    def get_config(self, config_name: str) -> BaseConfig:
        """
        Get a configuration object by name.
        
        Args:
            config_name: Name of configuration ('app', 'services', 'agents')
            
        Returns:
            Configuration object
            
        Raises:
            ConfigurationError: If configuration not found or not initialized
        """
        if not self._validated:
            raise ConfigurationError("Configuration manager not initialized. Call initialize() first.")
        
        if config_name not in self._configs:
            available = list(self._configs.keys())
            raise ConfigurationError(f"Configuration '{config_name}' not found. Available: {available}")
        
        return self._configs[config_name]
    
    def get_google_api_key(self) -> Optional[str]:
        """Get Google API key with proper validation."""
        key = os.getenv("GOOGLE_API_KEY")
        if key and key != "your_google_api_key_here":
            return key
        return None
    
    def get_environment_variable(self, var_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Safe environment variable access with validation.
        
        Args:
            var_name: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
            
        Raises:
            ConfigurationError: If required variable is missing
        """
        value = os.getenv(var_name, default)
        
        if required and not value:
            raise ConfigurationError(f"Required environment variable {var_name} is not set")
        
        return value
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        app_config = self.get_config('app')
        return app_config.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        app_config = self.get_config('app')
        return app_config.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        app_config = self.get_config('app')
        return app_config.environment == Environment.TESTING
    
    def get_all_configs(self) -> Dict[str, BaseConfig]:
        """Get all loaded configurations."""
        if not self._validated:
            raise ConfigurationError("Configuration manager not initialized")
        return self._configs.copy()


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


@lru_cache(maxsize=1)
def get_config_manager() -> ConfigurationManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Initialized configuration manager
        
    Raises:
        ConfigurationError: If configuration is not initialized
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()
        _config_manager.initialize()
    
    return _config_manager


def get_app_config() -> AppConfig:
    """Convenience function to get app configuration."""
    return get_config_manager().get_config('app')


def get_services_config() -> ServicesConfig:
    """Convenience function to get services configuration."""
    return get_config_manager().get_config('services')


def get_agents_config() -> AgentConfig:
    """Convenience function to get agents configuration."""
    return get_config_manager().get_config('agents')


def require_google_api_key() -> str:
    """
    Get Google API key with validation and fallback handling.
    
    Returns:
        Valid Google API key
        
    Raises:
        ConfigurationError: If API key is not properly configured
    """
    try:
        config_manager = get_config_manager()
        key = config_manager.get_google_api_key()
        
        if not key:
            # Check if Vertex AI is configured as fallback
            app_config = config_manager.get_config('app')
            if app_config.google_genai_use_vertexai:
                # Try to get project from environment for Vertex AI
                project = os.getenv("GOOGLE_CLOUD_PROJECT")
                if project:
                    raise ConfigurationError(
                        "Vertex AI is configured but API key is required for this operation. "
                        "Please set GOOGLE_API_KEY for direct API access."
                    )
            
            raise ConfigurationError(
                "Google API key not configured. Please set GOOGLE_API_KEY environment variable "
                "or configure Vertex AI with GOOGLE_GENAI_USE_VERTEXAI=True and GOOGLE_CLOUD_PROJECT"
            )
        
        return key
        
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to get Google API key: {e}")

def test_google_api_connection() -> Dict[str, Any]:
    """
    Test Google AI API connection and return status.
    
    Returns:
        Dictionary with connection test results
    """
    result = {
        "api_key_configured": False,
        "vertex_ai_configured": False,
        "connection_test": "not_tested",
        "error": None
    }
    
    try:
        config_manager = get_config_manager()
        app_config = config_manager.get_config('app')
        
        # Check API key configuration
        api_key = config_manager.get_google_api_key()
        result["api_key_configured"] = bool(api_key)
        
        # Check Vertex AI configuration
        result["vertex_ai_configured"] = (
            app_config.google_genai_use_vertexai and 
            bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
        )
        
        # Basic validation test
        if api_key:
            if api_key.startswith("AIza") and len(api_key) >= 35:
                result["connection_test"] = "format_valid"
            else:
                result["connection_test"] = "format_invalid"
                result["error"] = "API key format appears invalid"
        elif result["vertex_ai_configured"]:
            result["connection_test"] = "vertex_ai_ready"
        else:
            result["connection_test"] = "no_configuration"
            result["error"] = "Neither API key nor Vertex AI is properly configured"
        
    except Exception as e:
        result["error"] = str(e)
        result["connection_test"] = "error"
    
    return result


def validate_startup_configuration() -> None:
    """
    Validate configuration at application startup.
    Call this early in main() to catch configuration issues immediately.
    """
    try:
        get_config_manager()  # This triggers initialization and validation
        logging.info("✅ Configuration validation passed")
    except ConfigurationError as e:
        logging.error(f"❌ Configuration validation failed: {e}")
        sys.exit(1)