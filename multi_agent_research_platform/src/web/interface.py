"""
Core Web Interface Implementation

Main web interface classes for ADK integration, debugging, and monitoring
of the multi-agent research platform.
"""

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

# TODO: Update for ADK v1.5.0 - get_fast_api_app no longer available  
# from google.adk import get_fast_api_app
from fastapi import FastAPI

from .config import WebConfig, DebugConfig, MonitoringConfig, WebSocketConfig
from .api import AgentAPI, OrchestrationAPI, DebugAPI, MonitoringAPI
from .handlers import WebSocketHandler, EventHandler, LogHandler
from .templates import get_template_renderer, get_dashboard_renderer
from ..agents import AgentRegistry, AgentOrchestrator
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


@dataclass
class WebInterfaceState:
    """State tracking for web interface."""
    is_running: bool = False
    start_time: Optional[float] = None
    connected_clients: int = 0
    total_requests: int = 0
    active_sessions: int = 0
    last_activity: Optional[float] = None


class WebInterface:
    """
    Main web interface for the multi-agent research platform.
    
    Integrates with Google ADK web capabilities and provides debugging,
    monitoring, and interaction features for the multi-agent system.
    """
    
    def __init__(self,
                 web_config: WebConfig,
                 debug_config: Optional[DebugConfig] = None,
                 monitoring_config: Optional[MonitoringConfig] = None,
                 websocket_config: Optional[WebSocketConfig] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None):
        
        self.web_config = web_config
        self.debug_config = debug_config or DebugConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.websocket_config = websocket_config or WebSocketConfig()
        
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        self.artifact_service = artifact_service
        
        # Web interface state
        self.state = WebInterfaceState()
        
        # FastAPI app (updated for ADK v1.5.0)
        self.app = FastAPI(title="Multi-Agent Research Platform", version="1.0.0")
        self.server = None
        self.server_thread = None
        
        # Sub-interfaces
        self.debug_interface = None
        self.monitoring_interface = None
        
        # API handlers
        self.agent_api = None
        self.orchestration_api = None
        self.debug_api = None
        self.monitoring_api = None
        
        # Template renderer
        self.template_renderer = None
        self.dashboard_renderer = None
        
        # WebSocket and event handlers
        self.websocket_handler = None
        self.event_handler = None
        self.log_handler = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize web interface components."""
        try:
            # Initialize sub-interfaces
            self.debug_interface = DebugInterface(
                self.debug_config,
                logger=self.logger,
                session_service=self.session_service,
                memory_service=self.memory_service,
            )
            
            self.monitoring_interface = MonitoringInterface(
                self.monitoring_config,
                logger=self.logger,
                session_service=self.session_service,
            )
            
            # Initialize API handlers
            self.agent_api = AgentAPI(
                logger=self.logger,
                session_service=self.session_service,
            )
            
            self.orchestration_api = OrchestrationAPI(
                logger=self.logger,
                session_service=self.session_service,
            )
            
            self.debug_api = DebugAPI(
                self.debug_interface,
                logger=self.logger,
            )
            
            self.monitoring_api = MonitoringAPI(
                self.monitoring_interface,
                logger=self.logger,
            )
            
            # Initialize template renderers
            self.template_renderer = get_template_renderer(logger=self.logger)
            self.dashboard_renderer = get_dashboard_renderer(logger=self.logger)
            
            # Initialize WebSocket and event handlers
            if self.websocket_config.enabled:
                self.websocket_handler = WebSocketHandler(
                    self.websocket_config,
                    logger=self.logger,
                )
                
                self.event_handler = EventHandler(
                    self.websocket_handler,
                    logger=self.logger,
                )
            
            # Initialize log handler
            self.log_handler = LogHandler(
                self.debug_config,
                websocket_handler=self.websocket_handler,
                logger=self.logger,
            )
            
            if self.logger:
                self.logger.info("Web interface components initialized successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize web interface components: {e}")
            raise
    
    async def start(self) -> bool:
        """
        Start the web interface server.
        
        Returns:
            Success status
        """
        try:
            if self.state.is_running:
                if self.logger:
                    self.logger.warning("Web interface is already running")
                return True
            
            if self.logger:
                self.logger.info(f"Starting web interface on {self.web_config.host}:{self.web_config.port}")
            
            # Get FastAPI app from ADK
            self.app = get_fast_api_app()
            
            # Configure CORS if enabled
            if self.web_config.enable_cors:
                from fastapi.middleware.cors import CORSMiddleware
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=self.web_config.cors_origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            
            # Register API routes
            await self._register_api_routes()
            
            # Setup WebSocket endpoints
            if self.websocket_handler:
                await self._setup_websocket_endpoints()
            
            # Setup static files if enabled
            if self.web_config.static_files_enabled:
                self._setup_static_files()
            
            # Start sub-interfaces
            if self.debug_interface:
                await self.debug_interface.start()
            
            if self.monitoring_interface:
                await self.monitoring_interface.start()
            
            # Start the server
            import uvicorn
            config = uvicorn.Config(
                app=self.app,
                host=self.web_config.host,
                port=self.web_config.port,
                reload=self.web_config.reload,
                log_level="debug" if self.web_config.debug else "info",
            )
            
            self.server = uvicorn.Server(config)
            
            # Run server in separate thread to avoid blocking
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            # Wait a moment for server to start
            await asyncio.sleep(1)
            
            # Update state
            self.state.is_running = True
            self.state.start_time = time.time()
            self.state.last_activity = time.time()
            
            if self.logger:
                self.logger.info(f"Web interface started successfully at http://{self.web_config.host}:{self.web_config.port}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start web interface: {e}")
            return False
    
    def _run_server(self) -> None:
        """Run the uvicorn server."""
        try:
            asyncio.run(self.server.serve())
        except Exception as e:
            if self.logger:
                self.logger.error(f"Server error: {e}")
    
    async def stop(self) -> bool:
        """
        Stop the web interface server.
        
        Returns:
            Success status
        """
        try:
            if not self.state.is_running:
                if self.logger:
                    self.logger.warning("Web interface is not running")
                return True
            
            if self.logger:
                self.logger.info("Stopping web interface...")
            
            # Stop sub-interfaces
            if self.debug_interface:
                await self.debug_interface.stop()
            
            if self.monitoring_interface:
                await self.monitoring_interface.stop()
            
            # Stop WebSocket handler
            if self.websocket_handler:
                await self.websocket_handler.stop()
            
            # Stop server
            if self.server:
                self.server.should_exit = True
            
            # Wait for server thread to finish
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)
            
            # Update state
            self.state.is_running = False
            
            if self.logger:
                self.logger.info("Web interface stopped successfully")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to stop web interface: {e}")
            return False
    
    async def _register_api_routes(self) -> None:
        """Register API routes with the FastAPI app."""
        # Agent management routes
        self.app.include_router(
            self.agent_api.router,
            prefix="/api/v1/agents",
            tags=["agents"]
        )
        
        # Orchestration routes
        self.app.include_router(
            self.orchestration_api.router,
            prefix="/api/v1/orchestration",
            tags=["orchestration"]
        )
        
        # Debug routes
        self.app.include_router(
            self.debug_api.router,
            prefix="/api/v1/debug",
            tags=["debug"]
        )
        
        # Monitoring routes
        self.app.include_router(
            self.monitoring_api.router,
            prefix="/api/v1/monitoring",
            tags=["monitoring"]
        )
        
        # Dashboard HTML routes
        @self.app.get("/")
        async def dashboard_home():
            """Serve the main dashboard page."""
            from fastapi.responses import HTMLResponse
            return HTMLResponse(
                self.template_renderer.render_template("dashboard.html")
            )
        
        @self.app.get("/dashboard")
        async def dashboard_page():
            """Serve the dashboard page."""
            from fastapi.responses import HTMLResponse
            return HTMLResponse(
                self.template_renderer.render_template("dashboard.html")
            )
        
        @self.app.get("/api/v1/dashboards/{dashboard_name}")
        async def get_dashboard_data(dashboard_name: str):
            """Get dashboard data for AJAX requests."""
            try:
                if dashboard_name == "agents":
                    agents = AgentRegistry.get_all_agents()
                    return {
                        "agents": [
                            {
                                "agent_id": agent.agent_id,
                                "name": agent.name,
                                "agent_type": agent.agent_type.value,
                                "capabilities": [cap.value for cap in agent.get_capabilities()],
                                "is_active": agent.is_active,
                                "total_tasks_completed": agent.total_tasks_completed,
                                "status": agent.get_status()
                            }
                            for agent in agents
                        ]
                    }
                elif dashboard_name == "overview":
                    return {
                        "system_status": "active",
                        "agents": AgentRegistry.get_registry_status(),
                        "web_interface": self.get_status(),
                        "timestamp": time.time()
                    }
                else:
                    return {"error": f"Unknown dashboard: {dashboard_name}"}
            except Exception as e:
                return {"error": str(e)}
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "1.0.0",
                "mode": self.web_config.mode.value,
            }
        
        # Status endpoint
        @self.app.get("/status")
        async def interface_status():
            return {
                "web_interface": self.get_status(),
                "agents": AgentRegistry.get_registry_status(),
                "debug": self.debug_interface.get_status() if self.debug_interface else None,
                "monitoring": self.monitoring_interface.get_status() if self.monitoring_interface else None,
            }
    
    async def _setup_websocket_endpoints(self) -> None:
        """Setup WebSocket endpoints."""
        if not self.websocket_handler:
            return
        
        @self.app.websocket(self.websocket_config.path)
        async def websocket_endpoint(websocket):
            await self.websocket_handler.handle_connection(websocket)
    
    def _setup_static_files(self) -> None:
        """Setup static file serving."""
        try:
            from fastapi.staticfiles import StaticFiles
            import os
            
            # Create static directory if it doesn't exist
            static_dir = os.path.join(os.getcwd(), "static")
            os.makedirs(static_dir, exist_ok=True)
            
            self.app.mount(
                self.web_config.static_files_path,
                StaticFiles(directory=static_dir),
                name="static"
            )
            
            if self.logger:
                self.logger.info(f"Static files enabled at {self.web_config.static_files_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to setup static files: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get web interface status."""
        uptime = time.time() - self.state.start_time if self.state.start_time else 0
        
        return {
            "is_running": self.state.is_running,
            "uptime_seconds": uptime,
            "host": self.web_config.host,
            "port": self.web_config.port,
            "mode": self.web_config.mode.value,
            "debug_enabled": self.web_config.debug,
            "connected_clients": self.state.connected_clients,
            "total_requests": self.state.total_requests,
            "active_sessions": self.state.active_sessions,
            "last_activity": self.state.last_activity,
            "websocket_enabled": self.websocket_config.enabled,
            "components": {
                "debug_interface": self.debug_interface is not None,
                "monitoring_interface": self.monitoring_interface is not None,
                "websocket_handler": self.websocket_handler is not None,
                "event_handler": self.event_handler is not None,
            }
        }
    
    async def send_event(self, event_type: str, data: Any) -> None:
        """Send event through WebSocket if available."""
        if self.event_handler:
            await self.event_handler.send_event(event_type, data)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.state.last_activity = time.time()
        self.state.total_requests += 1


class DebugInterface:
    """Debug interface for multi-agent system debugging."""
    
    def __init__(self,
                 config: DebugConfig,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None):
        
        self.config = config
        self.logger = logger
        self.session_service = session_service
        self.memory_service = memory_service
        
        # Debug state
        self.is_active = False
        self.debug_sessions = {}
        self.breakpoints = {}
        self.step_mode_enabled = False
        
        # Performance profiling
        self.performance_data = {}
        self.execution_traces = []
        
        # Log capture
        self.captured_logs = []
        self.log_filters = {}
    
    async def start(self) -> bool:
        """Start debug interface."""
        try:
            self.is_active = True
            
            if self.logger:
                self.logger.info("Debug interface started")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start debug interface: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop debug interface."""
        try:
            self.is_active = False
            
            # Clear debug state
            self.debug_sessions.clear()
            self.breakpoints.clear()
            self.step_mode_enabled = False
            
            if self.logger:
                self.logger.info("Debug interface stopped")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to stop debug interface: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get debug interface status."""
        return {
            "is_active": self.is_active,
            "step_debugging_enabled": self.config.enable_step_debugging,
            "agent_inspection_enabled": self.config.enable_agent_inspection,
            "live_logs_enabled": self.config.enable_live_logs,
            "performance_profiling_enabled": self.config.enable_performance_profiling,
            "active_debug_sessions": len(self.debug_sessions),
            "active_breakpoints": len(self.breakpoints),
            "step_mode_enabled": self.step_mode_enabled,
            "captured_logs_count": len(self.captured_logs),
            "execution_traces_count": len(self.execution_traces),
        }
    
    async def inspect_agent(self, agent_id: str) -> Dict[str, Any]:
        """Inspect agent state and configuration."""
        agent = AgentRegistry.get_agent(agent_id)
        if not agent:
            return {"error": f"Agent {agent_id} not found"}
        
        inspection_data = {
            "agent_id": agent_id,
            "name": agent.name,
            "type": agent.agent_type.value,
            "status": agent.get_status(),
            "capabilities": [cap.value for cap in agent.get_capabilities()],
            "is_active": agent.is_active,
            "total_tasks_completed": agent.total_tasks_completed,
        }
        
        # Add performance metrics if available
        if hasattr(agent, 'get_performance_metrics'):
            inspection_data["performance_metrics"] = agent.get_performance_metrics()
        
        # Add specialization metrics for custom agents
        if hasattr(agent, 'get_specialization_metrics'):
            inspection_data["specialization_metrics"] = agent.get_specialization_metrics()
        
        return inspection_data
    
    def capture_log(self, log_entry: Dict[str, Any]) -> None:
        """Capture log entry for debugging."""
        if not self.config.enable_live_logs:
            return
        
        # Apply filters
        if self._should_capture_log(log_entry):
            self.captured_logs.append({
                **log_entry,
                "captured_at": time.time(),
            })
            
            # Trim to max entries
            if len(self.captured_logs) > self.config.max_log_entries:
                self.captured_logs = self.captured_logs[-self.config.max_log_entries:]
    
    def _should_capture_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if log entry should be captured based on filters."""
        # Check log level
        log_level = log_entry.get("level", "INFO").upper()
        config_level = self.config.log_level.value.upper()
        
        level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }
        
        if level_priority.get(log_level, 1) < level_priority.get(config_level, 1):
            return False
        
        return True


class MonitoringInterface:
    """Monitoring interface for real-time system monitoring."""
    
    def __init__(self,
                 config: MonitoringConfig,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None):
        
        self.config = config
        self.logger = logger
        self.session_service = session_service
        
        # Monitoring state
        self.is_active = False
        self.monitoring_tasks = []
        
        # Metrics storage
        self.performance_metrics = []
        self.usage_metrics = []
        self.error_metrics = []
        
        # Real-time data
        self.current_metrics = {}
        self.alert_states = {}
        
        # Update tracking
        self.last_update = None
        self.update_counter = 0
    
    async def start(self) -> bool:
        """Start monitoring interface."""
        try:
            self.is_active = True
            
            # Start monitoring tasks
            if self.config.enable_real_time_updates:
                self.monitoring_tasks.append(
                    asyncio.create_task(self._real_time_monitoring_loop())
                )
            
            if self.config.collect_performance_metrics:
                self.monitoring_tasks.append(
                    asyncio.create_task(self._performance_monitoring_loop())
                )
            
            if self.logger:
                self.logger.info("Monitoring interface started")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start monitoring interface: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop monitoring interface."""
        try:
            self.is_active = False
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            self.monitoring_tasks.clear()
            
            if self.logger:
                self.logger.info("Monitoring interface stopped")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to stop monitoring interface: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring interface status."""
        return {
            "is_active": self.is_active,
            "real_time_updates_enabled": self.config.enable_real_time_updates,
            "update_interval_seconds": self.config.update_interval_seconds,
            "active_monitoring_tasks": len(self.monitoring_tasks),
            "performance_metrics_count": len(self.performance_metrics),
            "usage_metrics_count": len(self.usage_metrics),
            "error_metrics_count": len(self.error_metrics),
            "last_update": self.last_update,
            "update_counter": self.update_counter,
            "current_metrics": self.current_metrics,
            "active_alerts": len(self.alert_states),
        }
    
    async def _real_time_monitoring_loop(self) -> None:
        """Real-time monitoring loop."""
        while self.is_active:
            try:
                await self._collect_current_metrics()
                await self._check_alert_conditions()
                
                self.last_update = time.time()
                self.update_counter += 1
                
                await asyncio.sleep(self.config.update_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Real-time monitoring error: {e}")
                await asyncio.sleep(self.config.update_interval_seconds)
    
    async def _performance_monitoring_loop(self) -> None:
        """Performance monitoring loop."""
        while self.is_active:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(self.config.update_interval_seconds * 2)  # Less frequent
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.config.update_interval_seconds * 2)
    
    async def _collect_current_metrics(self) -> None:
        """Collect current system metrics."""
        # Agent metrics
        registry_status = AgentRegistry.get_registry_status()
        
        self.current_metrics.update({
            "timestamp": time.time(),
            "agents": {
                "total": registry_status["total_agents"],
                "active": registry_status["active_agents"],
                "by_type": registry_status["agents_by_type"],
            },
        })
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics."""
        # This would collect detailed performance data
        performance_data = {
            "timestamp": time.time(),
            "memory_usage": 0,  # Would get actual memory usage
            "cpu_usage": 0,     # Would get actual CPU usage
            "response_times": [],
            "error_rates": {},
        }
        
        self.performance_metrics.append(performance_data)
        
        # Trim old metrics
        retention_seconds = self.config.metrics_retention_days * 24 * 3600
        cutoff_time = time.time() - retention_seconds
        
        self.performance_metrics = [
            m for m in self.performance_metrics 
            if m["timestamp"] > cutoff_time
        ]
    
    async def _check_alert_conditions(self) -> None:
        """Check alert conditions and trigger alerts if needed."""
        if not self.config.enable_alerts:
            return
        
        # Check each alert threshold
        for metric_name, threshold in self.config.alert_thresholds.items():
            current_value = self.current_metrics.get(metric_name, 0)
            
            # Simple threshold check
            if current_value > threshold:
                if metric_name not in self.alert_states:
                    self.alert_states[metric_name] = {
                        "triggered_at": time.time(),
                        "threshold": threshold,
                        "current_value": current_value,
                    }
                    
                    # Would trigger actual alert here
                    if self.logger:
                        self.logger.warning(f"Alert triggered: {metric_name} = {current_value} > {threshold}")
            else:
                # Clear alert if it was active
                if metric_name in self.alert_states:
                    del self.alert_states[metric_name]