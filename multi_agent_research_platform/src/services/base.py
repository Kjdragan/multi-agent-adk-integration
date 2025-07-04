"""
Base service classes and infrastructure.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field

from ..platform_logging import get_logger, RunLogger


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class ServiceHealth(BaseModel):
    """Service health information."""
    service_name: str = Field(description="Service name")
    status: ServiceStatus = Field(description="Current service status")
    last_check: datetime = Field(description="Last health check time")
    checks_passed: int = Field(default=0, description="Number of successful health checks")
    checks_failed: int = Field(default=0, description="Number of failed health checks")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")
    error_message: Optional[str] = Field(default=None, description="Last error message")


ServiceType = TypeVar('ServiceType', bound='BaseService')


class BaseService(ABC):
    """
    Base class for all platform services.
    
    Provides common functionality for health monitoring, lifecycle management,
    and error handling that all services should inherit.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
        # Service state
        self._status = ServiceStatus.STARTING
        self._start_time = time.time()
        self._last_health_check = datetime.utcnow()
        self._health_checks_passed = 0
        self._health_checks_failed = 0
        self._last_error: Optional[str] = None
        
        # Health monitoring
        self._health_check_interval = self.config.get('health_check_interval', 60)
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Logging
        self._logger: Optional[RunLogger] = None
    
    async def start(self) -> None:
        """Start the service and begin health monitoring."""
        try:
            self._status = ServiceStatus.STARTING
            await self._start_impl()
            self._status = ServiceStatus.HEALTHY
            
            # Start health monitoring if enabled
            if self._health_check_interval > 0:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self._log_info(f"Service {self.name} started successfully")
            
        except Exception as e:
            self._status = ServiceStatus.UNHEALTHY
            self._last_error = str(e)
            self._log_error(f"Failed to start service {self.name}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the service and cleanup resources."""
        try:
            self._status = ServiceStatus.STOPPED
            
            # Stop health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            await self._stop_impl()
            self._log_info(f"Service {self.name} stopped successfully")
            
        except Exception as e:
            self._last_error = str(e)
            self._log_error(f"Error stopping service {self.name}: {e}")
            raise
    
    async def health_check(self) -> ServiceHealth:
        """Perform a health check and return status."""
        try:
            self._last_health_check = datetime.utcnow()
            
            # Perform service-specific health check
            is_healthy, details = await self._health_check_impl()
            
            if is_healthy:
                if self._status == ServiceStatus.UNHEALTHY:
                    self._status = ServiceStatus.HEALTHY
                self._health_checks_passed += 1
                self._last_error = None
            else:
                self._status = ServiceStatus.UNHEALTHY
                self._health_checks_failed += 1
            
            return ServiceHealth(
                service_name=self.name,
                status=self._status,
                last_check=self._last_health_check,
                checks_passed=self._health_checks_passed,
                checks_failed=self._health_checks_failed,
                uptime_seconds=time.time() - self._start_time,
                details=details,
                error_message=self._last_error
            )
            
        except Exception as e:
            self._status = ServiceStatus.UNHEALTHY
            self._health_checks_failed += 1
            self._last_error = str(e)
            
            return ServiceHealth(
                service_name=self.name,
                status=self._status,
                last_check=self._last_health_check,
                checks_passed=self._health_checks_passed,
                checks_failed=self._health_checks_failed,
                uptime_seconds=time.time() - self._start_time,
                details={},
                error_message=str(e)
            )
    
    @property
    def status(self) -> ServiceStatus:
        """Get current service status."""
        return self._status
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def set_logger(self, logger: RunLogger) -> None:
        """Set the logger for this service."""
        self._logger = logger
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        if self._logger:
            self._logger.debug(message, service=self.name, **kwargs)
    
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message."""
        if self._logger:
            self._logger.info(message, service=self.name, **kwargs)
    
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        if self._logger:
            self._logger.warning(message, service=self.name, **kwargs)
    
    def _log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        if self._logger:
            self._logger.error(message, service=self.name, **kwargs)
    
    async def _health_check_loop(self) -> None:
        """Continuous health check loop."""
        while self._status != ServiceStatus.STOPPED:
            try:
                await asyncio.sleep(self._health_check_interval)
                if self._status != ServiceStatus.STOPPED:
                    await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_error(f"Health check loop error: {e}")
    
    @abstractmethod
    async def _start_impl(self) -> None:
        """Service-specific startup implementation."""
        pass
    
    @abstractmethod
    async def _stop_impl(self) -> None:
        """Service-specific shutdown implementation."""
        pass
    
    @abstractmethod
    async def _health_check_impl(self) -> tuple[bool, Dict[str, Any]]:
        """
        Service-specific health check implementation.
        
        Returns:
            Tuple of (is_healthy, details_dict)
        """
        pass


class ServiceRegistry:
    """
    Registry for managing multiple services.
    
    Provides centralized service lifecycle management and health monitoring.
    """
    
    def __init__(self):
        self._services: Dict[str, BaseService] = {}
        self._startup_order: List[str] = []
        self._logger: Optional[RunLogger] = None
    
    def register(self, service: BaseService, startup_order: int = 0) -> None:
        """Register a service with optional startup order priority."""
        self._services[service.name] = service
        
        # Insert in startup order (lower numbers start first)
        inserted = False
        for i, existing_name in enumerate(self._startup_order):
            existing_service = self._services[existing_name]
            existing_order = getattr(existing_service, '_startup_order', 0)
            if startup_order < existing_order:
                self._startup_order.insert(i, service.name)
                inserted = True
                break
        
        if not inserted:
            self._startup_order.append(service.name)
        
        # Set startup order on service for future reference
        service._startup_order = startup_order
        
        # Set logger if available
        if self._logger:
            service.set_logger(self._logger)
    
    def get_service(self, name: str) -> Optional[BaseService]:
        """Get a service by name."""
        return self._services.get(name)
    
    def get_service_typed(self, name: str, service_type: Type[ServiceType]) -> Optional[ServiceType]:
        """Get a service by name with type checking."""
        service = self._services.get(name)
        if service and isinstance(service, service_type):
            return service
        return None
    
    def set_logger(self, logger: RunLogger) -> None:
        """Set logger for registry and all registered services."""
        self._logger = logger
        for service in self._services.values():
            service.set_logger(logger)
    
    async def start_all(self) -> None:
        """Start all registered services in startup order."""
        for service_name in self._startup_order:
            service = self._services[service_name]
            try:
                await service.start()
                if self._logger:
                    self._logger.info(f"Started service: {service_name}")
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Failed to start service {service_name}: {e}")
                raise
    
    async def stop_all(self) -> None:
        """Stop all registered services in reverse startup order."""
        for service_name in reversed(self._startup_order):
            service = self._services[service_name]
            try:
                await service.stop()
                if self._logger:
                    self._logger.info(f"Stopped service: {service_name}")
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Error stopping service {service_name}: {e}")
                # Continue stopping other services even if one fails
    
    async def health_check_all(self) -> Dict[str, ServiceHealth]:
        """Perform health checks on all services."""
        health_results = {}
        
        for service_name, service in self._services.items():
            try:
                health = await service.health_check()
                health_results[service_name] = health
            except Exception as e:
                # Create error health status
                health_results[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    last_check=datetime.utcnow(),
                    uptime_seconds=time.time() - service._start_time,
                    error_message=str(e)
                )
        
        return health_results
    
    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy service names."""
        return [
            name for name, service in self._services.items()
            if not service.is_healthy
        ]
    
    @property
    def all_healthy(self) -> bool:
        """Check if all services are healthy."""
        return all(service.is_healthy for service in self._services.values())
    
    @property
    def service_count(self) -> int:
        """Get number of registered services."""
        return len(self._services)
    
    def list_services(self) -> List[str]:
        """Get list of all registered service names."""
        return list(self._services.keys())