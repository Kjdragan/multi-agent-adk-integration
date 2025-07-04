"""
Service factory for creating and configuring platform services.
"""

from typing import Dict, Optional

from .base import ServiceRegistry
from .session import SessionService, InMemorySessionService, DatabaseSessionService, VertexAISessionService
from .memory import MemoryService, InMemoryMemoryService, DatabaseMemoryService, VertexAIRagMemoryService
from .artifact import (
    ArtifactService, InMemoryArtifactService, LocalFileArtifactService, 
    GCSArtifactService, S3ArtifactService
)
from ..config.services import ServicesConfig, SessionServiceType, MemoryServiceType, ArtifactServiceType
from ..platform_logging import RunLogger


class ServiceFactory:
    """
    Factory for creating and configuring platform services.
    
    Handles service instantiation based on configuration and environment.
    """
    
    @staticmethod
    def create_session_service(config: ServicesConfig) -> SessionService:
        """Create a session service based on configuration."""
        session_config = config.session
        
        if session_config.service_type == SessionServiceType.IN_MEMORY:
            return InMemorySessionService(session_config)
        
        elif session_config.service_type == SessionServiceType.DATABASE:
            return DatabaseSessionService(session_config)
        
        elif session_config.service_type == SessionServiceType.VERTEX_AI:
            return VertexAISessionService(session_config)
        
        else:
            raise ValueError(f"Unknown session service type: {session_config.service_type}")
    
    @staticmethod
    def create_memory_service(config: ServicesConfig) -> MemoryService:
        """Create a memory service based on configuration."""
        memory_config = config.memory
        
        if memory_config.service_type == MemoryServiceType.IN_MEMORY:
            return InMemoryMemoryService(memory_config)
        
        elif memory_config.service_type == MemoryServiceType.DATABASE:
            return DatabaseMemoryService(memory_config)
        
        elif memory_config.service_type == MemoryServiceType.VERTEX_AI_RAG:
            return VertexAIRagMemoryService(memory_config)
        
        else:
            raise ValueError(f"Unknown memory service type: {memory_config.service_type}")
    
    @staticmethod
    def create_artifact_service(config: ServicesConfig) -> ArtifactService:
        """Create an artifact service based on configuration."""
        artifact_config = config.artifact
        
        if artifact_config.service_type == ArtifactServiceType.IN_MEMORY:
            return InMemoryArtifactService(artifact_config)
        
        elif artifact_config.service_type == ArtifactServiceType.LOCAL_FILE:
            return LocalFileArtifactService(artifact_config)
        
        elif artifact_config.service_type == ArtifactServiceType.GCS:
            return GCSArtifactService(artifact_config)
        
        elif artifact_config.service_type == ArtifactServiceType.S3:
            return S3ArtifactService(artifact_config)
        
        elif artifact_config.service_type == ArtifactServiceType.DATABASE:
            # Use local file service with database metadata
            return LocalFileArtifactService(artifact_config)
        
        else:
            raise ValueError(f"Unknown artifact service type: {artifact_config.service_type}")
    
    @staticmethod
    def create_service_registry(config: ServicesConfig, 
                              logger: Optional[RunLogger] = None) -> ServiceRegistry:
        """Create a complete service registry with all services."""
        registry = ServiceRegistry()
        
        if logger:
            registry.set_logger(logger)
        
        # Create services in dependency order
        # Session service has no dependencies
        session_service = ServiceFactory.create_session_service(config)
        registry.register(session_service, startup_order=1)
        
        # Memory service may depend on session service
        memory_service = ServiceFactory.create_memory_service(config)
        registry.register(memory_service, startup_order=2)
        
        # Artifact service is independent
        artifact_service = ServiceFactory.create_artifact_service(config)
        registry.register(artifact_service, startup_order=3)
        
        return registry


def create_services(config: ServicesConfig, 
                   logger: Optional[RunLogger] = None) -> Dict[str, any]:
    """
    Create all platform services based on configuration.
    
    Returns a dictionary with service instances for easy access.
    """
    services = {}
    
    # Create individual services
    services['session'] = ServiceFactory.create_session_service(config)
    services['memory'] = ServiceFactory.create_memory_service(config)
    services['artifact'] = ServiceFactory.create_artifact_service(config)
    
    # Set logger on all services
    if logger:
        for service in services.values():
            service.set_logger(logger)
    
    return services


def create_development_services(logger: Optional[RunLogger] = None) -> Dict[str, any]:
    """
    Create services optimized for development.
    
    Uses in-memory implementations for fast iteration and easy debugging.
    """
    config = ServicesConfig()
    config.configure_for_environment("development")
    
    return create_services(config, logger)


def create_production_services(database_url: str,
                             vertex_ai_project: str,
                             gcs_bucket: str,
                             logger: Optional[RunLogger] = None) -> Dict[str, any]:
    """
    Create services optimized for production.
    
    Uses persistent storage with proper scaling and monitoring.
    """
    config = ServicesConfig()
    production_config = config.create_production_config(
        database_url=database_url,
        vertex_ai_project=vertex_ai_project,
        gcs_bucket=gcs_bucket
    )
    
    return create_services(production_config, logger)


async def initialize_services(services: Dict[str, any]) -> None:
    """Initialize all services in the correct order."""
    # Initialize in dependency order
    initialization_order = ['session', 'artifact', 'memory']
    
    for service_name in initialization_order:
        if service_name in services:
            service = services[service_name]
            await service.start()


async def shutdown_services(services: Dict[str, any]) -> None:
    """Shutdown all services in reverse order."""
    # Shutdown in reverse dependency order
    shutdown_order = ['memory', 'artifact', 'session']
    
    for service_name in shutdown_order:
        if service_name in services:
            service = services[service_name]
            try:
                await service.stop()
            except Exception as e:
                # Log error but continue shutting down other services
                print(f"Error shutting down {service_name} service: {e}")


class ServiceManager:
    """
    High-level service manager for the platform.
    
    Provides convenient methods for service lifecycle management.
    """
    
    def __init__(self, config: ServicesConfig, logger: Optional[RunLogger] = None):
        self.config = config
        self.logger = logger
        self.registry: Optional[ServiceRegistry] = None
        self.services: Dict[str, any] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all services."""
        if self._initialized:
            return
        
        # Create service registry
        self.registry = ServiceFactory.create_service_registry(self.config, self.logger)
        
        # Create individual services for easy access
        self.services = create_services(self.config, self.logger)
        
        # Start all services
        await self.registry.start_all()
        
        self._initialized = True
        
        if self.logger:
            self.logger.info("All services initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown all services."""
        if not self._initialized:
            return
        
        if self.registry:
            await self.registry.stop_all()
        
        self.services.clear()
        self.registry = None
        self._initialized = False
        
        if self.logger:
            self.logger.info("All services shut down")
    
    def get_session_service(self) -> Optional[SessionService]:
        """Get the session service."""
        return self.services.get('session')
    
    def get_memory_service(self) -> Optional[MemoryService]:
        """Get the memory service."""
        return self.services.get('memory')
    
    def get_artifact_service(self) -> Optional[ArtifactService]:
        """Get the artifact service."""
        return self.services.get('artifact')
    
    async def health_check(self) -> Dict[str, Dict[str, any]]:
        """Perform health checks on all services."""
        if not self.registry:
            return {}
        
        return await self.registry.health_check_all()
    
    @property
    def is_healthy(self) -> bool:
        """Check if all services are healthy."""
        if not self.registry:
            return False
        
        return self.registry.all_healthy
    
    def get_unhealthy_services(self) -> list[str]:
        """Get list of unhealthy service names."""
        if not self.registry:
            return []
        
        return self.registry.get_unhealthy_services()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# Convenience functions for common service creation patterns

def create_minimal_services(logger: Optional[RunLogger] = None) -> ServiceManager:
    """Create minimal services for testing or simple applications."""
    config = ServicesConfig()
    # Use all in-memory services
    config.session.service_type = SessionServiceType.IN_MEMORY
    config.memory.service_type = MemoryServiceType.IN_MEMORY
    config.artifact.service_type = ArtifactServiceType.IN_MEMORY
    
    return ServiceManager(config, logger)


def create_local_file_services(data_dir: str = "data", 
                             logger: Optional[RunLogger] = None) -> ServiceManager:
    """Create services using local file storage."""
    from pathlib import Path
    
    config = ServicesConfig()
    
    # Use database session service with local SQLite
    config.session.service_type = SessionServiceType.DATABASE
    config.session.database_url = f"sqlite:///{data_dir}/sessions.db"
    
    # Use database memory service with local SQLite
    config.memory.service_type = MemoryServiceType.DATABASE
    config.memory.database_url = f"sqlite:///{data_dir}/memory.db"
    
    # Use local file artifact service
    config.artifact.service_type = ArtifactServiceType.LOCAL_FILE
    config.artifact.local_storage_path = Path(data_dir) / "artifacts"
    
    return ServiceManager(config, logger)


def create_cloud_services(project_id: str,
                         database_url: str,
                         gcs_bucket: str,
                         rag_corpus: str,
                         logger: Optional[RunLogger] = None) -> ServiceManager:
    """Create services using Google Cloud infrastructure."""
    config = ServicesConfig()
    
    # Use Vertex AI session service
    config.session.service_type = SessionServiceType.VERTEX_AI
    config.session.database_url = database_url
    config.session.vertex_ai_project = project_id
    
    # Use Vertex AI RAG memory service
    config.memory.service_type = MemoryServiceType.VERTEX_AI_RAG
    config.memory.vertex_ai_project = project_id
    config.memory.rag_corpus_name = rag_corpus
    
    # Use GCS artifact service
    config.artifact.service_type = ArtifactServiceType.GCS
    config.artifact.gcs_project = project_id
    config.artifact.gcs_bucket_name = gcs_bucket
    
    return ServiceManager(config, logger)