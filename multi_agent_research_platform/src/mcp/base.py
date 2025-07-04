"""
Base classes for MCP (Model Context Protocol) server integration.

Provides common functionality for all MCP servers including authentication,
rate limiting, error handling, and integration with platform services.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json
from datetime import datetime, timedelta

from ..context import ToolContextPattern, MemoryAccessPattern
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


class MCPServerStatus(str, Enum):
    """Status of MCP server connection and operations."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class MCPOperationType(str, Enum):
    """Types of MCP operations."""
    SEARCH = "search"
    QUERY = "query"
    RETRIEVE = "retrieve"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"


@dataclass
class MCPAuthConfig:
    """Authentication configuration for MCP servers."""
    auth_type: str  # "api_key", "oauth", "token", etc.
    api_key: Optional[str] = None
    token: Optional[str] = None
    oauth_config: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    endpoint: Optional[str] = None
    timeout_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "auth_type": self.auth_type,
            "api_key": self.api_key,
            "token": self.token,
            "oauth_config": self.oauth_config or {},
            "headers": self.headers or {},
            "endpoint": self.endpoint,
            "timeout": self.timeout_seconds,
        }
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for HTTP requests."""
        headers = self.headers.copy() if self.headers else {}
        
        if self.auth_type == "api_key" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_type == "token" and self.token:
            headers["Authorization"] = f"Token {self.token}"
        
        return headers


@dataclass
class MCPServerConfig:
    """Base configuration for MCP servers."""
    server_name: str
    server_type: str
    base_url: str
    auth_config: MCPAuthConfig
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: int = 30
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    custom_headers: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "server_name": self.server_name,
            "server_type": self.server_type,
            "base_url": self.base_url,
            "auth_config": self.auth_config.to_dict(),
            "rate_limits": {
                "requests_per_minute": self.rate_limit_requests_per_minute,
                "requests_per_hour": self.rate_limit_requests_per_hour,
            },
            "retry_config": {
                "max_retries": self.max_retries,
                "retry_delay_seconds": self.retry_delay_seconds,
            },
            "timeout_seconds": self.timeout_seconds,
            "caching": {
                "enabled": self.enable_caching,
                "ttl_seconds": self.cache_ttl_seconds,
            },
            "custom_headers": self.custom_headers or {},
        }


@dataclass
class MCPServerResult:
    """Result from MCP server operation."""
    server_name: str
    operation_type: MCPOperationType
    status: MCPServerStatus
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    cached: bool = False
    rate_limited: bool = False
    retry_count: int = 0
    
    @property
    def success(self) -> bool:
        """Check if operation was successful."""
        return self.status in [MCPServerStatus.CONNECTED, MCPServerStatus.AUTHENTICATED] and self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "server_name": self.server_name,
            "operation_type": self.operation_type.value,
            "status": self.status.value,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "cached": self.cached,
            "rate_limited": self.rate_limited,
            "retry_count": self.retry_count,
        }


class RateLimiter:
    """Rate limiter for MCP server requests."""
    
    def __init__(self, 
                 requests_per_minute: int = 60,
                 requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Track requests
        self.minute_requests: List[datetime] = []
        self.hour_requests: List[datetime] = []
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits."""
        async with self._lock:
            now = datetime.utcnow()
            
            # Clean old requests
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            
            self.minute_requests = [req for req in self.minute_requests if req > minute_ago]
            self.hour_requests = [req for req in self.hour_requests if req > hour_ago]
            
            # Check limits
            minute_ok = len(self.minute_requests) < self.requests_per_minute
            hour_ok = len(self.hour_requests) < self.requests_per_hour
            
            return minute_ok and hour_ok
    
    async def record_request(self) -> None:
        """Record a successful request."""
        async with self._lock:
            now = datetime.utcnow()
            self.minute_requests.append(now)
            self.hour_requests.append(now)
    
    async def wait_for_rate_limit(self) -> float:
        """Wait until a request can be made, return wait time."""
        start_time = time.time()
        
        while not await self.can_make_request():
            await asyncio.sleep(0.1)
        
        return time.time() - start_time


class MCPCache:
    """Simple in-memory cache for MCP server responses."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, server_name: str, operation: str, params: Dict[str, Any]) -> str:
        """Generate cache key from request parameters."""
        import hashlib
        key_data = f"{server_name}:{operation}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    async def get(self, server_name: str, operation: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and not expired."""
        async with self._lock:
            key = self._generate_key(server_name, operation, params)
            
            if key in self.cache:
                cached_item = self.cache[key]
                
                # Check if expired
                if time.time() - cached_item["timestamp"] < self.ttl_seconds:
                    return cached_item["data"]
                else:
                    # Remove expired item
                    del self.cache[key]
            
            return None
    
    async def set(self, server_name: str, operation: str, params: Dict[str, Any], data: Any) -> None:
        """Cache result data."""
        async with self._lock:
            key = self._generate_key(server_name, operation, params)
            
            self.cache[key] = {
                "data": data,
                "timestamp": time.time(),
                "server_name": server_name,
                "operation": operation,
            }
    
    async def clear_expired(self) -> int:
        """Clear expired cache entries, return count cleared."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, item in self.cache.items()
                if now - item["timestamp"] >= self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)


class MCPServer(ABC):
    """
    Abstract base class for MCP (Model Context Protocol) servers.
    
    Provides common functionality including authentication, rate limiting,
    caching, error handling, and integration with platform services.
    """
    
    def __init__(self,
                 config: MCPServerConfig,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        self.config = config
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Server state
        self.status = MCPServerStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        self.connection_time: Optional[datetime] = None
        
        # Rate limiting and caching
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests_per_minute,
            config.rate_limit_requests_per_hour
        )
        self.cache = MCPCache(config.cache_ttl_seconds) if config.enable_caching else None
        
        # Context patterns for integration
        self.tool_pattern = ToolContextPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        self.memory_pattern = MemoryAccessPattern(
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
        )
        
        # Request session for HTTP calls
        self._session: Optional[Any] = None
    
    async def connect(self) -> bool:
        """Connect and authenticate with the MCP server."""
        try:
            self.status = MCPServerStatus.CONNECTING
            
            # Initialize HTTP session
            await self._init_session()
            
            # Perform server-specific connection
            connection_result = await self._connect_server()
            
            if connection_result:
                # Authenticate if needed
                auth_result = await self._authenticate()
                
                if auth_result:
                    self.status = MCPServerStatus.AUTHENTICATED
                    self.connection_time = datetime.utcnow()
                    
                    if self.logger:
                        self.logger.info(f"Successfully connected to {self.config.server_name}")
                    
                    return True
                else:
                    self.status = MCPServerStatus.ERROR
                    self.last_error = "Authentication failed"
            else:
                self.status = MCPServerStatus.ERROR
                self.last_error = "Connection failed"
            
        except Exception as e:
            self.status = MCPServerStatus.ERROR
            self.last_error = str(e)
            
            if self.logger:
                self.logger.error(f"Failed to connect to {self.config.server_name}: {e}")
        
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            await self._disconnect_server()
            
            if self._session:
                await self._session.close()
                self._session = None
            
            self.status = MCPServerStatus.DISCONNECTED
            self.connection_time = None
            
            if self.logger:
                self.logger.info(f"Disconnected from {self.config.server_name}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error disconnecting from {self.config.server_name}: {e}")
    
    async def execute_operation(self, 
                              operation_type: MCPOperationType,
                              parameters: Dict[str, Any],
                              use_cache: bool = True) -> MCPServerResult:
        """
        Execute an operation with comprehensive error handling and monitoring.
        
        Args:
            operation_type: Type of operation to perform
            parameters: Operation parameters
            use_cache: Whether to use cached results if available
            
        Returns:
            MCPServerResult with operation results
        """
        start_time = time.time()
        
        result = MCPServerResult(
            server_name=self.config.server_name,
            operation_type=operation_type,
            status=self.status,
        )
        
        try:
            # Check server status
            if self.status not in [MCPServerStatus.CONNECTED, MCPServerStatus.AUTHENTICATED]:
                await self.connect()
                
                if self.status not in [MCPServerStatus.CONNECTED, MCPServerStatus.AUTHENTICATED]:
                    result.error = f"Server not connected: {self.last_error}"
                    return result
            
            # Check cache first
            if use_cache and self.cache:
                cached_data = await self.cache.get(
                    self.config.server_name, 
                    operation_type.value, 
                    parameters
                )
                
                if cached_data is not None:
                    result.data = cached_data
                    result.cached = True
                    result.status = MCPServerStatus.AUTHENTICATED
                    result.execution_time_ms = (time.time() - start_time) * 1000
                    
                    if self.logger:
                        self.logger.debug(f"Cache hit for {self.config.server_name} {operation_type.value}")
                    
                    return result
            
            # Check rate limits
            if not await self.rate_limiter.can_make_request():
                wait_time = await self.rate_limiter.wait_for_rate_limit()
                result.rate_limited = True
                result.metadata["rate_limit_wait_time"] = wait_time
                
                if self.logger:
                    self.logger.warning(f"Rate limited {self.config.server_name}, waited {wait_time:.2f}s")
            
            # Execute operation with retries
            for attempt in range(self.config.max_retries + 1):
                try:
                    # Record request for rate limiting
                    await self.rate_limiter.record_request()
                    
                    # Execute server-specific operation
                    operation_data = await self._execute_operation(operation_type, parameters)
                    
                    result.data = operation_data
                    result.status = MCPServerStatus.AUTHENTICATED
                    result.retry_count = attempt
                    
                    # Cache successful result
                    if use_cache and self.cache and operation_data is not None:
                        await self.cache.set(
                            self.config.server_name,
                            operation_type.value,
                            parameters,
                            operation_data
                        )
                    
                    break
                    
                except Exception as e:
                    result.retry_count = attempt
                    
                    if attempt < self.config.max_retries:
                        if self.logger:
                            self.logger.warning(f"Retry {attempt + 1} for {self.config.server_name}: {e}")
                        
                        await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
                    else:
                        result.error = str(e)
                        result.status = MCPServerStatus.ERROR
                        
                        if self.logger:
                            self.logger.error(f"Failed {self.config.server_name} after {attempt + 1} attempts: {e}")
            
        except Exception as e:
            result.error = str(e)
            result.status = MCPServerStatus.ERROR
            
            if self.logger:
                self.logger.error(f"Unexpected error in {self.config.server_name}: {e}")
        
        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the MCP server."""
        health_info = {
            "server_name": self.config.server_name,
            "status": self.status.value,
            "connected": self.status in [MCPServerStatus.CONNECTED, MCPServerStatus.AUTHENTICATED],
            "connection_time": self.connection_time.isoformat() if self.connection_time else None,
            "last_error": self.last_error,
            "rate_limiter": {
                "minute_requests": len(self.rate_limiter.minute_requests),
                "hour_requests": len(self.rate_limiter.hour_requests),
                "can_make_request": await self.rate_limiter.can_make_request(),
            },
        }
        
        # Add cache info if enabled
        if self.cache:
            health_info["cache"] = {
                "enabled": True,
                "size": len(self.cache.cache),
                "ttl_seconds": self.cache.ttl_seconds,
            }
        
        return health_info
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get comprehensive server information."""
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "capabilities": self.get_capabilities(),
            "connection_time": self.connection_time.isoformat() if self.connection_time else None,
            "last_error": self.last_error,
        }
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _init_session(self) -> None:
        """Initialize HTTP session for the server."""
        pass
    
    @abstractmethod
    async def _connect_server(self) -> bool:
        """Perform server-specific connection logic."""
        pass
    
    @abstractmethod
    async def _authenticate(self) -> bool:
        """Perform server-specific authentication."""
        pass
    
    @abstractmethod
    async def _disconnect_server(self) -> None:
        """Perform server-specific disconnection logic."""
        pass
    
    @abstractmethod
    async def _execute_operation(self, 
                               operation_type: MCPOperationType, 
                               parameters: Dict[str, Any]) -> Any:
        """Execute server-specific operation."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities and supported operations."""
        pass
    
    # Helper methods for subclasses
    
    def log_operation(self, 
                     operation: str, 
                     parameters: Dict[str, Any],
                     result: Optional[MCPServerResult] = None) -> None:
        """Log MCP server operation for monitoring and debugging."""
        if not self.logger:
            return
        
        log_data = {
            "server_name": self.config.server_name,
            "operation": operation,
            "parameter_keys": list(parameters.keys()),
            "parameter_count": len(parameters),
        }
        
        if result:
            log_data.update({
                "result_status": result.status.value,
                "result_success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "cached": result.cached,
                "rate_limited": result.rate_limited,
                "retry_count": result.retry_count,
            })
        
        self.logger.info(f"MCP operation: {self.config.server_name}.{operation}", **log_data)
    
    async def store_results_in_memory(self, 
                                    operation_type: MCPOperationType,
                                    query: str,
                                    results: Any) -> None:
        """Store operation results in memory for future reference."""
        if not self.memory_service:
            return
        
        try:
            memory_text = f"""
MCP Server Result ({self.config.server_name}):
Operation: {operation_type.value}
Query: {query}

Results: {json.dumps(results, indent=2) if results else 'No results'}

Server: {self.config.server_name}
Timestamp: {datetime.utcnow().isoformat()}
            """.strip()
            
            # This would integrate with the actual memory service
            if hasattr(self.memory_service, 'store'):
                await self.memory_service.store(
                    text=memory_text,
                    metadata={
                        "type": "mcp_server_result",
                        "server_name": self.config.server_name,
                        "operation_type": operation_type.value,
                        "query": query,
                    }
                )
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing MCP results in memory: {e}")


class HTTPMCPServer(MCPServer):
    """
    Base class for HTTP-based MCP servers.
    
    Provides common HTTP functionality including session management,
    request handling, and error processing.
    """
    
    async def _init_session(self) -> None:
        """Initialize aiohttp session for HTTP requests."""
        try:
            import aiohttp
            
            # Create session with timeout and headers
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            headers = self.config.auth_config.get_auth_headers()
            
            if self.config.custom_headers:
                headers.update(self.config.custom_headers)
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
            )
            
        except ImportError:
            raise RuntimeError("aiohttp is required for HTTP MCP servers")
    
    async def _disconnect_server(self) -> None:
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def make_http_request(self, 
                              method: str,
                              endpoint: str,
                              params: Optional[Dict[str, Any]] = None,
                              json_data: Optional[Dict[str, Any]] = None,
                              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make HTTP request with error handling and logging.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON body data
            headers: Additional headers
            
        Returns:
            Response data as dictionary
        """
        if not self._session:
            raise RuntimeError("HTTP session not initialized")
        
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Merge headers
        request_headers = {}
        if headers:
            request_headers.update(headers)
        
        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=request_headers
            ) as response:
                
                # Check for rate limiting
                if response.status == 429:
                    raise Exception("Rate limit exceeded")
                
                # Check for authentication errors
                if response.status == 401:
                    raise Exception("Authentication failed")
                
                # Check for general errors
                response.raise_for_status()
                
                # Parse response
                if response.content_type == 'application/json':
                    return await response.json()
                else:
                    return {"text": await response.text()}
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise