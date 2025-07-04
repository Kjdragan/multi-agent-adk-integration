"""
Service configuration models for Session, Memory, and Artifact services.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base import BaseConfig, DatabaseSettings


class ServiceType(str, Enum):
    """Types of service implementations."""
    IN_MEMORY = "in_memory"
    DATABASE = "database"
    VERTEX_AI = "vertex_ai"
    REDIS = "redis"
    GCS = "gcs"
    S3 = "s3"
    LOCAL_FILE = "local_file"


class SessionServiceType(str, Enum):
    """Session service implementation types."""
    IN_MEMORY = "in_memory"
    DATABASE = "database"
    VERTEX_AI = "vertex_ai"


class MemoryServiceType(str, Enum):
    """Memory service implementation types."""
    IN_MEMORY = "in_memory"
    VERTEX_AI_RAG = "vertex_ai_rag"
    DATABASE = "database"


class ArtifactServiceType(str, Enum):
    """Artifact service implementation types."""
    IN_MEMORY = "in_memory"
    LOCAL_FILE = "local_file"
    GCS = "gcs"
    S3 = "s3"
    DATABASE = "database"


class SessionServiceConfig(BaseModel):
    """Configuration for session management service."""
    model_config = ConfigDict(extra="ignore")
    
    # Service type
    service_type: SessionServiceType = Field(
        default=SessionServiceType.IN_MEMORY,
        description="Session service implementation type"
    )
    
    # Database configuration (for DATABASE and VERTEX_AI types)
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL"
    )
    database_settings: Optional[DatabaseSettings] = Field(
        default=None,
        description="Database connection settings"
    )
    
    # Table/collection names
    sessions_table: str = Field(
        default="sessions",
        description="Sessions table/collection name"
    )
    events_table: str = Field(
        default="session_events",
        description="Events table/collection name"
    )
    state_table: str = Field(
        default="session_state",
        description="State table/collection name"
    )
    
    # Session management
    default_session_timeout: int = Field(
        default=3600,
        description="Default session timeout in seconds"
    )
    cleanup_interval: int = Field(
        default=300,
        description="Session cleanup interval in seconds"
    )
    max_sessions_per_user: int = Field(
        default=10,
        description="Maximum sessions per user"
    )
    
    # Performance settings
    enable_compression: bool = Field(
        default=True,
        description="Enable session data compression"
    )
    compression_algorithm: str = Field(
        default="gzip",
        description="Compression algorithm"
    )
    
    # State management
    state_prefix_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "user": {"persistent": True, "scope": "user"},
            "app": {"persistent": True, "scope": "application"},
            "temp": {"persistent": False, "scope": "session"},
            "": {"persistent": True, "scope": "session"}  # Default
        },
        description="State prefix configuration"
    )
    
    # Vertex AI specific settings
    vertex_ai_project: Optional[str] = Field(
        default=None,
        description="Vertex AI project ID"
    )
    vertex_ai_location: str = Field(
        default="us-central1",
        description="Vertex AI location"
    )
    
    @field_validator('service_type')
    @classmethod
    def validate_service_config(cls, v, info):
        """Validate service-specific configuration."""
        # Note: info.data may be incomplete during validation, so we only validate
        # what we can and defer complete validation to model_post_init
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation with complete data."""
        # Validate service-specific requirements with complete data
        if self.service_type in [SessionServiceType.DATABASE, SessionServiceType.VERTEX_AI]:
            if not self.database_url:
                raise ValueError(f"{self.service_type} service type requires database_url")
        
        if self.service_type == SessionServiceType.VERTEX_AI:
            if not self.vertex_ai_project:
                raise ValueError("Vertex AI service requires vertex_ai_project")


class MemoryServiceConfig(BaseModel):
    """Configuration for memory/knowledge management service."""
    model_config = ConfigDict(extra="ignore")
    
    # Service type
    service_type: MemoryServiceType = Field(
        default=MemoryServiceType.IN_MEMORY,
        description="Memory service implementation type"
    )
    
    # Vertex AI RAG configuration
    vertex_ai_project: Optional[str] = Field(
        default=None,
        description="Vertex AI project ID"
    )
    vertex_ai_location: str = Field(
        default="us-central1",
        description="Vertex AI location"
    )
    rag_corpus_name: Optional[str] = Field(
        default=None,
        description="Vertex AI RAG corpus resource name"
    )
    
    # Search configuration
    similarity_top_k: int = Field(
        default=5,
        description="Number of top similar results to return"
    )
    vector_distance_threshold: float = Field(
        default=0.7,
        description="Vector distance threshold for relevance"
    )
    
    # Database configuration (for DATABASE type)
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL for memory storage"
    )
    memory_table: str = Field(
        default="memory_entries",
        description="Memory entries table name"
    )
    embeddings_table: str = Field(
        default="memory_embeddings",
        description="Embeddings table name"
    )
    
    # Memory management
    max_memory_entries: int = Field(
        default=10000,
        description="Maximum memory entries to store"
    )
    memory_retention_days: int = Field(
        default=90,
        description="Memory retention period in days"
    )
    auto_cleanup_enabled: bool = Field(
        default=True,
        description="Enable automatic cleanup of old memories"
    )
    
    # Session ingestion settings
    auto_ingest_sessions: bool = Field(
        default=True,
        description="Automatically ingest completed sessions"
    )
    ingestion_criteria: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_events": 5,
            "min_duration_seconds": 30,
            "success_required": True,
            "exclude_error_sessions": True
        },
        description="Criteria for session ingestion"
    )
    
    # Content processing
    extract_summaries: bool = Field(
        default=True,
        description="Extract summaries for memory entries"
    )
    extract_keywords: bool = Field(
        default=True,
        description="Extract keywords from content"
    )
    content_chunking: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": True,
            "chunk_size": 1000,
            "overlap_size": 200,
            "strategy": "semantic"
        },
        description="Content chunking configuration"
    )
    
    @field_validator('service_type')
    @classmethod
    def validate_memory_config(cls, v, info):
        """Validate memory service configuration."""
        # Note: info.data may be incomplete during validation, so we only validate
        # what we can and defer complete validation to model_post_init
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation with complete data."""
        # Validate service-specific requirements with complete data
        if self.service_type == MemoryServiceType.VERTEX_AI_RAG:
            required_fields = []
            if not self.vertex_ai_project:
                required_fields.append('vertex_ai_project')
            if not self.rag_corpus_name:
                required_fields.append('rag_corpus_name')
            if required_fields:
                raise ValueError(f"Vertex AI RAG requires: {', '.join(required_fields)}")
        
        if self.service_type == MemoryServiceType.DATABASE:
            if not self.database_url:
                raise ValueError("Database memory service requires database_url")


class ArtifactServiceConfig(BaseModel):
    """Configuration for artifact storage service."""
    model_config = ConfigDict(extra="ignore")
    
    # Service type
    service_type: ArtifactServiceType = Field(
        default=ArtifactServiceType.LOCAL_FILE,
        description="Artifact service implementation type"
    )
    
    # Local file storage
    local_storage_path: Path = Field(
        default=Path("data/artifacts"),
        description="Local storage directory path"
    )
    
    # Google Cloud Storage
    gcs_bucket_name: Optional[str] = Field(
        default=None,
        description="GCS bucket name"
    )
    gcs_project: Optional[str] = Field(
        default=None,
        description="GCS project ID"
    )
    gcs_prefix: str = Field(
        default="artifacts/",
        description="GCS object prefix"
    )
    
    # Amazon S3
    s3_bucket_name: Optional[str] = Field(
        default=None,
        description="S3 bucket name"
    )
    s3_region: str = Field(
        default="us-east-1",
        description="S3 region"
    )
    s3_prefix: str = Field(
        default="artifacts/",
        description="S3 key prefix"
    )
    
    # Database storage (for metadata)
    database_url: Optional[str] = Field(
        default=None,
        description="Database URL for artifact metadata"
    )
    artifacts_table: str = Field(
        default="artifacts",
        description="Artifacts metadata table name"
    )
    
    # Storage settings
    max_artifact_size_mb: int = Field(
        default=100,
        description="Maximum artifact size in MB"
    )
    allowed_mime_types: List[str] = Field(
        default_factory=lambda: [
            "text/plain",
            "text/markdown",
            "application/json",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "image/png",
            "image/jpeg",
            "text/csv"
        ],
        description="Allowed MIME types for artifacts"
    )
    
    # Versioning
    enable_versioning: bool = Field(
        default=True,
        description="Enable artifact versioning"
    )
    max_versions_per_artifact: int = Field(
        default=10,
        description="Maximum versions per artifact"
    )
    
    # Cleanup
    cleanup_enabled: bool = Field(
        default=True,
        description="Enable automatic cleanup"
    )
    retention_days: int = Field(
        default=30,
        description="Artifact retention period in days"
    )
    cleanup_interval_hours: int = Field(
        default=24,
        description="Cleanup interval in hours"
    )
    
    # Performance
    enable_compression: bool = Field(
        default=True,
        description="Enable artifact compression"
    )
    compression_threshold_kb: int = Field(
        default=1024,
        description="Compression threshold in KB"
    )
    
    # Security
    encrypt_at_rest: bool = Field(
        default=True,
        description="Encrypt artifacts at rest"
    )
    encryption_key: Optional[str] = Field(
        default=None,
        description="Encryption key (if not using cloud provider encryption)"
    )
    
    # model_post_init moved to after field_validator to handle all validation
    
    @field_validator('service_type')
    @classmethod
    def validate_artifact_config(cls, v, info):
        """Validate artifact service configuration."""
        # Note: info.data may be incomplete during validation, so we only validate
        # what we can and defer complete validation to model_post_init
        return v
    
    def model_post_init(self, __context) -> None:
        """Post-initialization validation with complete data."""
        # Create storage directory for local file storage
        if self.service_type == ArtifactServiceType.LOCAL_FILE:
            self.local_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Validate service-specific requirements with complete data
        elif self.service_type == ArtifactServiceType.GCS:
            required = []
            if not self.gcs_bucket_name:
                required.append('gcs_bucket_name')
            if not self.gcs_project:
                required.append('gcs_project')
            if required:
                raise ValueError(f"GCS artifact service requires: {', '.join(required)}")
        
        elif self.service_type == ArtifactServiceType.S3:
            if not self.s3_bucket_name:
                raise ValueError("S3 artifact service requires s3_bucket_name")
        
        elif self.service_type == ArtifactServiceType.DATABASE:
            if not self.database_url:
                raise ValueError("Database artifact service requires database_url")


class ServicesConfig(BaseConfig):
    """Combined configuration for all services."""
    
    # Service configurations
    session: SessionServiceConfig = Field(
        default_factory=SessionServiceConfig,
        description="Session service configuration"
    )
    memory: MemoryServiceConfig = Field(
        default_factory=MemoryServiceConfig,
        description="Memory service configuration"
    )
    artifact: ArtifactServiceConfig = Field(
        default_factory=ArtifactServiceConfig,
        description="Artifact service configuration"
    )
    
    # Cross-service settings
    enable_service_health_checks: bool = Field(
        default=True,
        description="Enable service health monitoring"
    )
    health_check_interval: int = Field(
        default=60,
        description="Health check interval in seconds"
    )
    
    # Development vs Production configurations
    development_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "use_in_memory_services": True,
            "enable_debug_logging": True,
            "disable_cleanup": True
        },
        description="Development-specific service configuration"
    )
    
    production_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "use_persistent_services": True,
            "enable_monitoring": True,
            "enable_encryption": True,
            "enable_backup": True
        },
        description="Production-specific service configuration"
    )
    
    def configure_for_environment(self, environment: str) -> None:
        """Configure services based on environment."""
        if environment == "development":
            # Use in-memory services for fast development
            self.session.service_type = SessionServiceType.IN_MEMORY
            self.memory.service_type = MemoryServiceType.IN_MEMORY
            self.artifact.service_type = ArtifactServiceType.LOCAL_FILE
            
            # Disable cleanup for debugging
            self.session.cleanup_interval = 0
            self.memory.auto_cleanup_enabled = False
            self.artifact.cleanup_enabled = False
            
        elif environment == "production":
            # Use persistent services
            self.session.service_type = SessionServiceType.DATABASE
            self.memory.service_type = MemoryServiceType.VERTEX_AI_RAG
            self.artifact.service_type = ArtifactServiceType.GCS
            
            # Enable security and monitoring
            self.artifact.encrypt_at_rest = True
            self.enable_service_health_checks = True
    
    def get_database_urls(self, base_database_url: str) -> Dict[str, str]:
        """Generate database URLs for each service."""
        return {
            "session": f"{base_database_url}_sessions",
            "memory": f"{base_database_url}_memory",
            "artifact_metadata": f"{base_database_url}_artifacts"
        }
    
    def validate_service_compatibility(self) -> None:
        """Validate that service configurations are compatible."""
        # Check if all services requiring the same database can share it
        database_services = []
        
        if self.session.service_type in [SessionServiceType.DATABASE, SessionServiceType.VERTEX_AI]:
            database_services.append("session")
        
        if self.memory.service_type == MemoryServiceType.DATABASE:
            database_services.append("memory")
        
        if self.artifact.service_type == ArtifactServiceType.DATABASE:
            database_services.append("artifact")
        
        # Validate consistent database configuration
        if len(database_services) > 1:
            database_urls = [
                self.session.database_url,
                self.memory.database_url,
                self.artifact.database_url
            ]
            database_urls = [url for url in database_urls if url]
            
            if len(set(database_urls)) > 1:
                raise ValueError(
                    "Services using database storage should use the same database_url base"
                )
    
    def create_development_config(self) -> 'ServicesConfig':
        """Create a development-optimized configuration."""
        config = self.model_copy()
        config.configure_for_environment("development")
        return config
    
    def create_production_config(self, 
                               database_url: str,
                               vertex_ai_project: str,
                               gcs_bucket: str) -> 'ServicesConfig':
        """Create a production-optimized configuration."""
        config = self.model_copy()
        config.configure_for_environment("production")
        
        # Set production-specific values
        config.session.database_url = f"{database_url}_sessions"
        config.memory.database_url = f"{database_url}_memory"
        config.memory.vertex_ai_project = vertex_ai_project
        config.artifact.database_url = f"{database_url}_artifacts"
        config.artifact.gcs_bucket_name = gcs_bucket
        config.artifact.gcs_project = vertex_ai_project
        
        return config