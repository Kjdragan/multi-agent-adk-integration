"""
Multi-Agent Research Platform Services

Provides Session, State, Memory, and Artifact management services with
configurable backends for development and production environments.
"""

from .base import (
    BaseService,
    ServiceStatus,
    ServiceHealth,
    ServiceRegistry,
)
from .session import (
    SessionService,
    InMemorySessionService,
    DatabaseSessionService,
    VertexAISessionService,
)
from .memory import (
    MemoryService,
    InMemoryMemoryService,
    VertexAIRagMemoryService,
    DatabaseMemoryService,
)
from .artifact import (
    ArtifactService,
    InMemoryArtifactService,
    LocalFileArtifactService,
    GCSArtifactService,
    S3ArtifactService,
)
from .factory import (
    ServiceFactory,
    create_services,
    create_development_services,
    create_production_services,
)

__all__ = [
    # Base service infrastructure
    "BaseService",
    "ServiceStatus",
    "ServiceHealth", 
    "ServiceRegistry",
    
    # Session services
    "SessionService",
    "InMemorySessionService",
    "DatabaseSessionService",
    "VertexAISessionService",
    
    # Memory services
    "MemoryService",
    "InMemoryMemoryService",
    "VertexAIRagMemoryService",
    "DatabaseMemoryService",
    
    # Artifact services
    "ArtifactService",
    "InMemoryArtifactService", 
    "LocalFileArtifactService",
    "GCSArtifactService",
    "S3ArtifactService",
    
    # Service factory
    "ServiceFactory",
    "create_services",
    "create_development_services",
    "create_production_services",
]