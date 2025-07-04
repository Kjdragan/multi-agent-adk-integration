"""
Deployment configuration models for Cloud Run and other deployment targets.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base import BaseConfig, DatabaseSettings


class DeploymentTarget(str, Enum):
    """Supported deployment targets."""
    LOCAL = "local"
    CLOUD_RUN = "cloud_run"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    COMPUTE_ENGINE = "compute_engine"


class DatabaseType(str, Enum):
    """Supported database types for production."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    CLOUD_SQL = "cloud_sql"


class MonitoringProvider(str, Enum):
    """Supported monitoring providers."""
    GOOGLE_CLOUD = "google_cloud"
    PROMETHEUS = "prometheus"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    NONE = "none"


class CloudRunConfig(BaseModel):
    """Google Cloud Run specific configuration."""
    model_config = ConfigDict(extra="forbid")
    
    # Basic service settings
    service_name: str = Field(
        default="multi-agent-research-platform",
        description="Cloud Run service name"
    )
    region: str = Field(
        default="us-central1",
        description="Cloud Run region"
    )
    platform: str = Field(
        default="managed",
        description="Cloud Run platform (managed or gke)"
    )
    
    # Resource allocation
    cpu: str = Field(
        default="1000m",
        description="CPU allocation (e.g., '1000m' = 1 vCPU)"
    )
    memory: str = Field(
        default="2Gi",
        description="Memory allocation (e.g., '2Gi' = 2 GB)"
    )
    max_instances: int = Field(
        default=10,
        description="Maximum number of instances"
    )
    min_instances: int = Field(
        default=0,
        description="Minimum number of instances"
    )
    
    # Request handling
    concurrency: int = Field(
        default=80,
        description="Maximum concurrent requests per instance"
    )
    timeout: int = Field(
        default=300,
        description="Request timeout in seconds"
    )
    
    # Networking
    allow_unauthenticated: bool = Field(
        default=False,
        description="Allow unauthenticated requests"
    )
    vpc_connector: Optional[str] = Field(
        default=None,
        description="VPC connector for private networking"
    )
    
    # Environment variables
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the service"
    )
    
    # Health checks
    startup_probe_path: str = Field(
        default="/health",
        description="Startup probe endpoint"
    )
    liveness_probe_path: str = Field(
        default="/health",
        description="Liveness probe endpoint"
    )
    
    # Deployment settings
    revision_suffix: Optional[str] = Field(
        default=None,
        description="Revision suffix for deployments"
    )
    traffic_allocation: Dict[str, int] = Field(
        default_factory=lambda: {"LATEST": 100},
        description="Traffic allocation between revisions"
    )
    
    @field_validator('cpu')
    @classmethod
    def validate_cpu(cls, v):
        """Validate CPU allocation format."""
        if not (v.endswith('m') or v.isdigit()):
            raise ValueError("CPU must be in format '1000m' or '1'")
        return v
    
    @field_validator('memory')
    @classmethod
    def validate_memory(cls, v):
        """Validate memory allocation format."""
        if not (v.endswith(('Gi', 'Mi', 'G', 'M'))):
            raise ValueError("Memory must be in format '2Gi', '512Mi', etc.")
        return v


class DatabaseConfig(BaseModel):
    """Database configuration for production deployments."""
    model_config = ConfigDict(extra="forbid")
    
    # Database type
    database_type: DatabaseType = Field(
        default=DatabaseType.POSTGRESQL,
        description="Database type"
    )
    
    # Connection settings
    host: Optional[str] = Field(default=None, description="Database host")
    port: Optional[int] = Field(default=None, description="Database port")
    database_name: str = Field(
        default="research_platform",
        description="Database name"
    )
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    
    # Cloud SQL specific
    cloud_sql_instance: Optional[str] = Field(
        default=None,
        description="Cloud SQL instance connection name"
    )
    use_private_ip: bool = Field(
        default=True,
        description="Use private IP for Cloud SQL"
    )
    
    # Connection pooling
    pool_size: int = Field(
        default=20,
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default=30,
        description="Maximum pool overflow"
    )
    pool_timeout: int = Field(
        default=30,
        description="Pool timeout in seconds"
    )
    pool_recycle: int = Field(
        default=3600,
        description="Pool recycle time in seconds"
    )
    
    # SSL/TLS
    ssl_mode: str = Field(
        default="require",
        description="SSL mode (require, prefer, disable)"
    )
    ssl_cert_path: Optional[str] = Field(
        default=None,
        description="Path to SSL certificate"
    )
    
    # Backup and maintenance
    enable_backups: bool = Field(
        default=True,
        description="Enable automated backups"
    )
    backup_retention_days: int = Field(
        default=7,
        description="Backup retention period in days"
    )
    maintenance_window: Optional[str] = Field(
        default=None,
        description="Maintenance window (e.g., 'sun:05:00-sun:06:00')"
    )
    
    def get_connection_url(self) -> str:
        """Generate database connection URL."""
        if self.database_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.database_name}.db"
        
        if self.database_type == DatabaseType.CLOUD_SQL:
            if self.cloud_sql_instance:
                return f"postgresql+pg8000://{self.username}:{self.password}@/{self.database_name}?unix_sock=/cloudsql/{self.cloud_sql_instance}/.s.PGSQL.5432"
            
        # Standard PostgreSQL/MySQL
        protocol = "postgresql" if self.database_type == DatabaseType.POSTGRESQL else "mysql"
        port = self.port or (5432 if self.database_type == DatabaseType.POSTGRESQL else 3306)
        
        return f"{protocol}://{self.username}:{self.password}@{self.host}:{port}/{self.database_name}"
    
    @field_validator('database_type')
    @classmethod
    def validate_database_config(cls, v, info):
        """Validate database-specific configuration."""
        values = info.data if info else {}
        
        if v == DatabaseType.CLOUD_SQL:
            if not values.get('cloud_sql_instance'):
                raise ValueError("Cloud SQL requires cloud_sql_instance")
        
        if v in [DatabaseType.POSTGRESQL, DatabaseType.MYSQL]:
            required = ['host', 'username', 'password']
            missing = [f for f in required if not values.get(f)]
            if missing:
                raise ValueError(f"{v} database requires: {', '.join(missing)}")
        
        return v


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    model_config = ConfigDict(extra="forbid")
    
    # Monitoring provider
    provider: MonitoringProvider = Field(
        default=MonitoringProvider.GOOGLE_CLOUD,
        description="Monitoring provider"
    )
    
    # Metrics collection
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        description="Metrics endpoint port"
    )
    metrics_path: str = Field(
        default="/metrics",
        description="Metrics endpoint path"
    )
    
    # Logging
    enable_structured_logging: bool = Field(
        default=True,
        description="Enable structured logging"
    )
    log_level: str = Field(
        default="INFO",
        description="Log level"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json, plain)"
    )
    
    # Tracing
    enable_tracing: bool = Field(
        default=True,
        description="Enable distributed tracing"
    )
    trace_sample_rate: float = Field(
        default=0.1,
        description="Trace sampling rate (0.0 to 1.0)"
    )
    
    # Health checks
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health checks"
    )
    health_check_interval: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    # Alerting
    enable_alerting: bool = Field(
        default=True,
        description="Enable alerting"
    )
    alert_channels: List[str] = Field(
        default_factory=list,
        description="Alert notification channels"
    )
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    performance_sample_rate: float = Field(
        default=1.0,
        description="Performance monitoring sample rate"
    )
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metrics configuration"
    )


class SecurityConfig(BaseModel):
    """Security configuration for deployment."""
    model_config = ConfigDict(extra="forbid")
    
    # Authentication
    enable_authentication: bool = Field(
        default=True,
        description="Enable authentication"
    )
    auth_provider: str = Field(
        default="google_iap",
        description="Authentication provider"
    )
    
    # Authorization
    enable_rbac: bool = Field(
        default=True,
        description="Enable role-based access control"
    )
    admin_users: List[str] = Field(
        default_factory=list,
        description="Admin user emails"
    )
    
    # Network security
    enable_https_only: bool = Field(
        default=True,
        description="Enforce HTTPS only"
    )
    allowed_origins: List[str] = Field(
        default_factory=list,
        description="Allowed CORS origins"
    )
    
    # API security
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_per_minute: int = Field(
        default=1000,
        description="Rate limit per minute per user"
    )
    
    # Data encryption
    enable_encryption_at_rest: bool = Field(
        default=True,
        description="Enable encryption at rest"
    )
    enable_encryption_in_transit: bool = Field(
        default=True,
        description="Enable encryption in transit"
    )
    
    # Secrets management
    use_secret_manager: bool = Field(
        default=True,
        description="Use Google Secret Manager"
    )
    secrets_project: Optional[str] = Field(
        default=None,
        description="Project for secrets storage"
    )
    
    # Audit logging
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    audit_log_retention_days: int = Field(
        default=90,
        description="Audit log retention in days"
    )


class DeploymentConfig(BaseConfig):
    """Main deployment configuration."""
    
    # Deployment target
    target: DeploymentTarget = Field(
        default=DeploymentTarget.LOCAL,
        description="Deployment target"
    )
    
    # Environment settings
    environment_name: str = Field(
        default="development",
        description="Environment name"
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Google Cloud project ID"
    )
    
    # Component configurations
    cloud_run: CloudRunConfig = Field(
        default_factory=CloudRunConfig,
        description="Cloud Run configuration"
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration"
    )
    
    # Container settings
    container_image: str = Field(
        default="gcr.io/PROJECT_ID/multi-agent-research-platform:latest",
        description="Container image URL"
    )
    container_registry: str = Field(
        default="gcr.io",
        description="Container registry"
    )
    
    # Build settings
    build_context: str = Field(
        default=".",
        description="Docker build context"
    )
    dockerfile_path: str = Field(
        default="Dockerfile",
        description="Path to Dockerfile"
    )
    
    # Environment variables
    environment_variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Deployment environment variables"
    )
    
    # Scaling settings
    enable_autoscaling: bool = Field(
        default=True,
        description="Enable autoscaling"
    )
    min_replicas: int = Field(
        default=1,
        description="Minimum replicas"
    )
    max_replicas: int = Field(
        default=10,
        description="Maximum replicas"
    )
    target_cpu_utilization: int = Field(
        default=70,
        description="Target CPU utilization percentage"
    )
    
    def configure_for_cloud_run(self, project_id: str, region: str = "us-central1") -> None:
        """Configure for Google Cloud Run deployment."""
        self.target = DeploymentTarget.CLOUD_RUN
        self.project_id = project_id
        
        # Update Cloud Run config
        self.cloud_run.region = region
        self.cloud_run.allow_unauthenticated = False  # Use IAP
        
        # Update container image
        self.container_image = f"gcr.io/{project_id}/multi-agent-research-platform:latest"
        
        # Configure database for Cloud SQL
        self.database.database_type = DatabaseType.CLOUD_SQL
        self.database.cloud_sql_instance = f"{project_id}:us-central1:research-platform-db"
        
        # Configure monitoring for Google Cloud
        self.monitoring.provider = MonitoringProvider.GOOGLE_CLOUD
        
        # Configure security
        self.security.auth_provider = "google_iap"
        self.security.use_secret_manager = True
        self.security.secrets_project = project_id
    
    def get_deployment_command(self) -> str:
        """Generate deployment command based on target."""
        if self.target == DeploymentTarget.CLOUD_RUN:
            return f"""
gcloud run deploy {self.cloud_run.service_name} \\
    --image {self.container_image} \\
    --region {self.cloud_run.region} \\
    --platform managed \\
    --memory {self.cloud_run.memory} \\
    --cpu {self.cloud_run.cpu} \\
    --max-instances {self.cloud_run.max_instances} \\
    --min-instances {self.cloud_run.min_instances} \\
    --concurrency {self.cloud_run.concurrency} \\
    --timeout {self.cloud_run.timeout} \\
    --no-allow-unauthenticated \\
    --project {self.project_id}
""".strip()
        
        elif self.target == DeploymentTarget.DOCKER:
            return f"""
docker build -t {self.container_image} {self.build_context}
docker run -p 8080:8080 {self.container_image}
""".strip()
        
        return "# No deployment command available for this target"
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get all environment variables for deployment."""
        env_vars = self.environment_variables.copy()
        
        # Add common environment variables
        env_vars.update({
            "ENVIRONMENT": self.environment_name,
            "GOOGLE_CLOUD_PROJECT": self.project_id or "",
            "DATABASE_URL": self.database.get_connection_url(),
            "ENABLE_MONITORING": str(self.monitoring.enable_metrics),
            "LOG_LEVEL": self.monitoring.log_level,
        })
        
        return env_vars
    
    def validate_deployment_config(self) -> None:
        """Validate deployment configuration."""
        if self.target == DeploymentTarget.CLOUD_RUN:
            if not self.project_id:
                raise ValueError("project_id is required for Cloud Run deployment")
            
            if self.database.database_type == DatabaseType.CLOUD_SQL:
                if not self.database.cloud_sql_instance:
                    raise ValueError("cloud_sql_instance required for Cloud SQL")
        
        # Validate container image format
        if self.target != DeploymentTarget.LOCAL:
            if "PROJECT_ID" in self.container_image and not self.project_id:
                raise ValueError("project_id required to resolve container image")
    
    def create_production_config(self, project_id: str) -> 'DeploymentConfig':
        """Create production deployment configuration."""
        config = self.model_copy()
        config.environment_name = "production"
        config.configure_for_cloud_run(project_id)
        
        # Production-specific settings
        config.cloud_run.min_instances = 2
        config.cloud_run.max_instances = 20
        config.security.enable_authentication = True
        config.security.enable_rbac = True
        config.monitoring.enable_alerting = True
        
        return config